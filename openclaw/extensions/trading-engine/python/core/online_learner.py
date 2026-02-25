"""Online learning for agent weight adaptation — Regime-Aware Thompson Sampling.

This IS reinforcement learning, just practical for small sample sizes:
- After each trade closes: observe which agent signals were correct
- Update Beta distributions per agent PER REGIME (regime-aware)
- Sample from posteriors to get weights for next decision
- Works with as few as 10-20 completed trades

Unlike deep RL (PPO/SAC), this:
- Has theoretical guarantees (Bayesian regret bounds)
- Works with tiny datasets (Beta prior regularizes)
- Is interpretable (you can see each agent's win rate per regime)
- Adapts in real-time (every closed trade updates weights)
- Regime-aware: momentum gets high weight in trending, lower in ranging

Regime-Aware enhancement (v2, 2026-02-23):
- Per-regime Beta distributions: {agent: {regime: Beta(α,β)}}
- Hierarchical fallback: if regime has <3 trades, use global posterior
- IQC 2025 insight: all-weather strategy pool = different weights per macro state

References:
- Thompson (1933): original paper
- Agrawal & Goyal (2012): near-optimal regret bounds
- CADTS (arXiv:2410.04217): Adaptive Discounted TS for portfolios
- 167-paper meta-analysis (arXiv:2512.10913): implementation > algorithm
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("trading-engine.online_learner")


@dataclass
class AgentBeta:
    """Beta distribution for a single agent's signal quality.

    alpha = successes + prior, beta = failures + prior.
    Mean = alpha / (alpha + beta).
    """
    name: str
    alpha: float = 2.0    # prior: 2 successes (mildly optimistic)
    beta: float = 2.0     # prior: 2 failures
    total_trades: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def sample(self) -> float:
        """Sample from Beta(alpha, beta). Used for Thompson Sampling."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, pnl_pct: float = 0.0, discount: float = 0.95) -> None:
        """Update posterior with discounted Thompson Sampling.

        - Decay existing beliefs (prevents stale priors dominating)
        - Use sigmoid of pnl for continuous reward (not binary win/loss)
        """
        # Discount existing beliefs toward prior
        self.alpha = max(1.0, self.alpha * discount)
        self.beta = max(1.0, self.beta * discount)

        # Continuous reward via sigmoid: maps pnl_pct to (0, 1)
        reward = 1.0 / (1.0 + math.exp(-pnl_pct * 10))
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.total_trades += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "total_trades": self.total_trades,
        }


@dataclass
class TradeRecord:
    """Record of a completed trade for learning."""
    ticker: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    held_hours: float
    agent_signals: Dict[str, float]  # {agent_name: signal_value} at entry
    entry_time: float
    exit_time: float
    market_type: str = "crypto"
    regime: str = "unknown"  # regime at trade entry (for regime-aware learning)


# Default regime when none detected
_GLOBAL_REGIME = "_global"
# Minimum trades per regime before trusting regime-specific weights
_MIN_REGIME_TRADES = 3


class OnlineLearner:
    """Regime-Aware Thompson Sampling agent weight optimizer.

    Maintains per-regime Beta distributions for each agent's predictive
    accuracy. Different market regimes have different optimal agent weights
    (e.g. momentum works in trending, mean-reversion in ranging).

    Hierarchical fallback: if a regime has fewer than 3 trades,
    the global (all-regime) posterior is used instead.

    Usage::

        learner = OnlineLearner(save_path="models/online_learner.json")
        learner.load()  # restore from disk

        # After a trade closes
        learner.record_trade(
            ticker="BTC/USDT",
            entry_price=60000, exit_price=63000,
            pnl_pct=0.05,
            agent_signals={"quant": 0.7, "momentum": 0.3},
            regime="trending",
            ...
        )

        # Get adapted weights for current regime
        weights = learner.sample_weights(regime="trending")
    """

    AGENT_NAMES = ["market", "quant", "sentiment", "momentum", "regime",
                    "funding_rate", "oi_signal", "macro"]

    def __init__(
        self,
        save_path: Optional[str] = None,
        min_trades_to_adapt: int = 5,
        max_window: int = 50,
    ) -> None:
        self._save_path = Path(save_path) if save_path else None
        self._min_trades = min_trades_to_adapt
        self._max_window = max_window  # ADTS-style sliding window cap

        # Per-regime Beta distributions: {regime: {agent_name: AgentBeta}}
        # _GLOBAL_REGIME always updated (for fallback)
        self._regime_agents: Dict[str, Dict[str, AgentBeta]] = {
            _GLOBAL_REGIME: {name: AgentBeta(name=name) for name in self.AGENT_NAMES},
        }

        # Backward compat: self._agents points to global
        self._agents = self._regime_agents[_GLOBAL_REGIME]

        # Trade history for analysis
        self._trades: List[TradeRecord] = []
        self._total_pnl: float = 0.0
        self._regime_trade_counts: Dict[str, int] = {_GLOBAL_REGIME: 0}

    def _get_regime_agents(self, regime: str) -> Dict[str, AgentBeta]:
        """Get or create agent Betas for a specific regime."""
        if regime not in self._regime_agents:
            self._regime_agents[regime] = {
                name: AgentBeta(name=name) for name in self.AGENT_NAMES
            }
            self._regime_trade_counts[regime] = 0
        return self._regime_agents[regime]

    def _effective_agents(self, regime: str) -> Dict[str, AgentBeta]:
        """Get effective agents for a regime, with hierarchical fallback.

        If the regime has fewer than _MIN_REGIME_TRADES, fall back to global.
        """
        count = self._regime_trade_counts.get(regime, 0)
        if count >= _MIN_REGIME_TRADES and regime in self._regime_agents:
            return self._regime_agents[regime]
        return self._agents  # global fallback

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def has_enough_data(self) -> bool:
        return self.total_trades >= self._min_trades

    # ------------------------------------------------------------------
    # Core: record trade outcome + update posteriors
    # ------------------------------------------------------------------

    def record_trade(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        held_hours: float,
        agent_signals: Dict[str, float],
        market_type: str = "crypto",
        regime: str = "unknown",
    ) -> Dict[str, Any]:
        """Record a completed trade and update agent posteriors.

        Updates BOTH the regime-specific posteriors AND the global posteriors.
        This ensures global always has full data for fallback.

        Returns summary of the update.
        """
        now = time.time()
        record = TradeRecord(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            held_hours=held_hours,
            agent_signals=agent_signals,
            entry_time=now - held_hours * 3600,
            exit_time=now,
            market_type=market_type,
            regime=regime,
        )
        self._trades.append(record)
        self._total_pnl += pnl_pct

        # ADTS sliding window: cap trade history to prevent stale data domination
        if len(self._trades) > self._max_window:
            self._trades = self._trades[-self._max_window:]

        # Update both global and regime-specific Betas
        targets = [self._agents]  # always update global
        if regime and regime != "unknown":
            regime_agents = self._get_regime_agents(regime)
            targets.append(regime_agents)
            self._regime_trade_counts[regime] = self._regime_trade_counts.get(regime, 0) + 1
        self._regime_trade_counts[_GLOBAL_REGIME] = self._regime_trade_counts.get(_GLOBAL_REGIME, 0) + 1

        updates = {}
        profitable = pnl_pct > 0

        for agent_name, signal in agent_signals.items():
            if abs(signal) < 0.05:
                continue

            signal_bullish = signal > 0
            aligned = (signal_bullish and profitable) or (not signal_bullish and not profitable)
            effective_pnl = abs(pnl_pct) if aligned else -abs(pnl_pct)

            for agent_dict in targets:
                if agent_name not in agent_dict:
                    agent_dict[agent_name] = AgentBeta(name=agent_name)
                agent_dict[agent_name].update(pnl_pct=effective_pnl)

            updates[agent_name] = {
                "signal": round(signal, 3),
                "aligned": aligned,
                "new_mean": round(self._agents[agent_name].mean, 4),
            }

        logger.info(
            "[online_rl] Trade #%d: %s pnl=%+.2f%% regime=%s → updates: %s",
            self.total_trades, ticker, pnl_pct * 100, regime,
            {k: v["aligned"] for k, v in updates.items()},
        )

        if self._save_path:
            self.save()

        return {
            "trade_number": self.total_trades,
            "ticker": ticker,
            "pnl_pct": pnl_pct,
            "profitable": profitable,
            "regime": regime,
            "agent_updates": updates,
            "cumulative_pnl": self._total_pnl,
        }

    # ------------------------------------------------------------------
    # Weight sampling (Thompson Sampling)
    # ------------------------------------------------------------------

    def sample_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Sample weights from Beta posteriors (Regime-Aware Thompson Sampling).

        Uses regime-specific Betas if enough data, else falls back to global.
        If not enough global trades either, returns default static weights.
        """
        from agents.synthesizer import AGENT_WEIGHTS

        if not self.has_enough_data:
            return dict(AGENT_WEIGHTS)

        agents = self._effective_agents(regime)

        raw_samples = {}
        for name, agent_beta in agents.items():
            raw_samples[name] = agent_beta.sample()

        raw_samples["rl"] = 0.0

        total = sum(raw_samples.values())
        if total <= 0:
            return dict(AGENT_WEIGHTS)

        weights = {k: v / total for k, v in raw_samples.items()}
        return weights

    def get_mean_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Get mean weights for a regime (deterministic, no sampling noise)."""
        from agents.synthesizer import AGENT_WEIGHTS

        if not self.has_enough_data:
            return dict(AGENT_WEIGHTS)

        agents = self._effective_agents(regime)
        raw = {name: ab.mean for name, ab in agents.items()}
        raw["rl"] = 0.0
        total = sum(raw.values())
        if total <= 0:
            return dict(AGENT_WEIGHTS)
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save learner state to disk (including per-regime Betas)."""
        if self._save_path is None:
            return

        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "agents": {name: ab.to_dict() for name, ab in self._agents.items()},
            "regime_agents": {
                regime: {name: ab.to_dict() for name, ab in agents.items()}
                for regime, agents in self._regime_agents.items()
                if regime != _GLOBAL_REGIME
            },
            "regime_trade_counts": dict(self._regime_trade_counts),
            "total_trades": self.total_trades,
            "total_pnl": self._total_pnl,
            "last_trades": [
                {
                    "ticker": t.ticker,
                    "pnl_pct": t.pnl_pct,
                    "held_hours": t.held_hours,
                    "agent_signals": t.agent_signals,
                    "market_type": t.market_type,
                    "exit_time": t.exit_time,
                    "regime": t.regime,
                }
                for t in self._trades[-100:]
            ],
        }
        self._save_path.write_text(json.dumps(state, indent=2))
        logger.debug("[online_rl] Saved state (%d trades, %d regimes)",
                     self.total_trades, len(self._regime_agents) - 1)

    def load(self) -> bool:
        """Load learner state from disk (including per-regime Betas). Returns True if loaded."""
        if self._save_path is None or not self._save_path.exists():
            return False

        try:
            state = json.loads(self._save_path.read_text())

            # Load global agents
            for name, data in state.get("agents", {}).items():
                if name not in self._agents:
                    self._agents[name] = AgentBeta(name=name)
                self._agents[name].alpha = data["alpha"]
                self._agents[name].beta = data["beta"]
                self._agents[name].total_trades = data["total_trades"]

            # Load per-regime agents
            for regime, agents_data in state.get("regime_agents", {}).items():
                regime_agents = self._get_regime_agents(regime)
                for name, data in agents_data.items():
                    if name not in regime_agents:
                        regime_agents[name] = AgentBeta(name=name)
                    regime_agents[name].alpha = data["alpha"]
                    regime_agents[name].beta = data["beta"]
                    regime_agents[name].total_trades = data["total_trades"]

            self._regime_trade_counts = state.get("regime_trade_counts", {_GLOBAL_REGIME: 0})
            self._total_pnl = state.get("total_pnl", 0.0)

            # Restore trade records (for total_trades property)
            self._trades = []
            for t in state.get("last_trades", []):
                self._trades.append(TradeRecord(
                    ticker=t["ticker"],
                    entry_price=0.0,
                    exit_price=0.0,
                    pnl_pct=t["pnl_pct"],
                    held_hours=t.get("held_hours", 0.0),
                    agent_signals=t.get("agent_signals", {}),
                    entry_time=0.0,
                    exit_time=t.get("exit_time", 0.0),
                    market_type=t.get("market_type", "crypto"),
                    regime=t.get("regime", "unknown"),
                ))

            n_regimes = len(self._regime_agents) - 1
            logger.info(
                "[online_rl] Loaded state: %d trades, %d regimes, pnl=%+.2f%%",
                state.get("total_trades", 0), n_regimes, self._total_pnl * 100,
            )
            return True

        except Exception as exc:
            logger.warning("[online_rl] Failed to load state: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        regime_info = {}
        for regime, agents in self._regime_agents.items():
            if regime == _GLOBAL_REGIME:
                continue
            count = self._regime_trade_counts.get(regime, 0)
            regime_info[regime] = {
                "trade_count": count,
                "using_own_weights": count >= _MIN_REGIME_TRADES,
                "agents": {name: ab.to_dict() for name, ab in agents.items()},
            }

        return {
            "total_trades": self.total_trades,
            "has_enough_data": self.has_enough_data,
            "min_trades_to_adapt": self._min_trades,
            "cumulative_pnl_pct": round(self._total_pnl * 100, 2),
            "global_agents": {name: ab.to_dict() for name, ab in self._agents.items()},
            "regime_agents": regime_info,
            "regime_trade_counts": dict(self._regime_trade_counts),
            "mean_weights_global": self.get_mean_weights(),
            "recent_trades": [
                {
                    "ticker": t.ticker,
                    "pnl": f"{t.pnl_pct:+.2%}",
                    "regime": t.regime,
                    "market": t.market_type,
                }
                for t in self._trades[-5:]
            ],
        }
