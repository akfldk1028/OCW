"""Online learning for agent weight adaptation — Hierarchical Thompson Sampling.

Two architectures available:
1. OnlineLearner (v2): Flat 8-arm regime-aware TS (legacy)
2. HierarchicalOnlineLearner (v3): 2-level TS with 6 groups × ~5-7 signals = ~28 arms

v3 Hierarchical TS (2026-02-26):
- Level 1: 6 signal GROUPS (technical_trend, technical_reversion, technical_volume,
  derivatives, sentiment, macro) — converges in ~20 trades
- Level 2: 5-7 individual signals per group — converges in ~20 trades per group
- Final weight = group_weight × within_group_signal_weight
- Different discount rates: 0.98 (group, stable) vs 0.95 (signal, adaptive)
- Hierarchical fallback: regime → global at both levels
- Migration: old 8-arm state auto-mapped to new hierarchy

Theory:
- Flat 28 arms = 1,120 trades to converge (28 × 40)
- Hierarchical = ~140 trades (6×20 + 6×~20) — 5-6x faster
- Carlsson et al. IJCAI 2021: clustered TS significantly improves regret

References:
- Thompson (1933): original paper
- Agrawal & Goyal (2012): near-optimal regret bounds
- Carlsson et al. (IJCAI 2021): Thompson Sampling for Bandits with Clustered Arms
- Zhao et al. (arXiv:2602.15972): Hierarchical Unimodal Thompson Sampling
- CADTS (arXiv:2410.04217): Adaptive Discounted TS for portfolios
- 167-paper meta-analysis (arXiv:2512.10913): implementation > algorithm
"""

from __future__ import annotations

import json
import logging
import math
import os
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
        a = max(1e-6, self.alpha)
        b = max(1e-6, self.beta)
        return random.betavariate(a, b)

    def update(self, pnl_pct: float = 0.0, discount: float = 0.95) -> None:
        """Update posterior with discounted Thompson Sampling.

        - Decay existing beliefs (prevents stale priors dominating)
        - Use sigmoid of pnl for continuous reward (not binary win/loss)
        """
        # Discount existing beliefs toward prior
        self.alpha = max(1.0, self.alpha * discount)
        self.beta = max(1.0, self.beta * discount)

        # Continuous reward via sigmoid: maps pnl_pct to (0, 1)
        # pnl_pct is a fraction (e.g. 0.01 = +1%), scale=100 so:
        #   +1% → sigmoid(1.0) = 0.73 (strong positive)
        #   -1% → sigmoid(-1.0) = 0.27 (strong negative)
        #   +3% → sigmoid(3.0) = 0.95 (very strong positive)
        reward = 1.0 / (1.0 + math.exp(-pnl_pct * 100))
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
# 5 trades = same as min_trades_to_adapt. Below this, regime posterior has
# std ~0.19 which is too noisy — fall back to global (more data = lower variance).
_MIN_REGIME_TRADES = 5


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
        tmp = self._save_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self._save_path)  # atomic on POSIX
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

            # Detect stale posteriors from old sigmoid scale (all means ~0.50)
            # If we have 5+ trades but no agent mean deviates >2% from 0.50,
            # the posteriors are useless — reprocess with current scale.
            if self.total_trades >= 5:
                max_dev = max(abs(ab.mean - 0.5) for ab in self._agents.values())
                if max_dev < 0.02:
                    logger.warning(
                        "[online_rl] Stale posteriors detected (max deviation %.4f "
                        "after %d trades). Reprocessing with corrected reward scale.",
                        max_dev, self.total_trades,
                    )
                    self.reprocess_all_trades()

            return True

        except Exception as exc:
            logger.warning("[online_rl] Failed to load state: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Reprocessing (for scale corrections)
    # ------------------------------------------------------------------

    def reprocess_all_trades(self) -> None:
        """Reset posteriors to fresh priors and re-learn from trade history.

        Use after fixing the sigmoid scale or other reward mapping changes.
        Preserves trade history but rebuilds all Beta distributions from scratch.
        """
        old_trades = list(self._trades)
        old_pnl = self._total_pnl
        n = len(old_trades)

        # Reset all posteriors to fresh priors
        self._regime_agents = {
            _GLOBAL_REGIME: {name: AgentBeta(name=name) for name in self.AGENT_NAMES},
        }
        self._agents = self._regime_agents[_GLOBAL_REGIME]
        self._trades = []
        self._total_pnl = 0.0
        self._regime_trade_counts = {_GLOBAL_REGIME: 0}

        # Re-learn from each trade with the current reward mapping
        for t in old_trades:
            if abs(t.pnl_pct) < 1e-8:
                # Skip 0% PnL trades (broken price data from before the fix)
                continue
            self.record_trade(
                ticker=t.ticker,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pct=t.pnl_pct,
                held_hours=t.held_hours,
                agent_signals=t.agent_signals,
                market_type=t.market_type,
                regime=t.regime,
            )

        logger.info(
            "[online_rl] Reprocessed %d trades → %d valid (skipped 0%% PnL). "
            "New posteriors reflect corrected scale.",
            n, len(self._trades),
        )

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
                    "pnl_pct": t.pnl_pct,
                    "held_hours": t.held_hours,
                    "agent_signals": t.agent_signals,
                    "regime": t.regime,
                    "market": t.market_type,
                }
                for t in self._trades[-10:]
            ],
        }


# ======================================================================
# Hierarchical Thompson Sampling (v3)
# ======================================================================

# Mapping from old flat 8-arm signal names to new hierarchy
_OLD_TO_NEW_MAPPING: Dict[str, List[str]] = {
    "market": ["volume_spike", "vwap_deviation"],
    "quant": ["rsi_signal", "bb_deviation", "support_resistance"],
    "sentiment": ["news_sentiment", "fear_greed"],
    "momentum": ["ema_cross_fast", "trend_strength"],
    "regime": ["market_regime", "volatility_regime"],
    "funding_rate": ["funding_rate"],
    "oi_signal": ["oi_change", "long_short_ratio", "cvd_signal"],
    "macro": ["btc_dominance", "dxy_direction"],
}


class HierarchicalOnlineLearner:
    """3-Level Hierarchical Thompson Sampling for signal weight optimization.

    Level 0: 6 META PARAMS — how to trade in this regime? (position sizing, risk, style)
    Level 1: 6 signal GROUPS — which category of analysis is most reliable?
    Level 2: ~5-7 signals per group — which specific indicator within a group?

    Final weight for signal s in group g:
        w(s) = sample(group_beta[g]) × sample(signal_beta[g][s]) / Z

    This converges 5-6x faster than flat TS with 28 arms because:
    - Group level (6 arms) converges in ~20 trades
    - Signal level (5-7 arms) converges in ~20 trades per group
    - Total ~140 trades vs ~1,120 for flat

    Level 0 meta-parameters learn *how* to trade per regime:
    - position_scale, entry_selectivity, hold_patience, trade_frequency,
      trend_vs_reversion, risk_aversion — all Beta(2,2) priors

    References:
        Carlsson et al. (IJCAI 2021): Thompson Sampling for Bandits with Clustered Arms
        Zhao et al. (arXiv:2602.15972): Hierarchical Unimodal Thompson Sampling
        Meta-Thompson Sampling (Kveton 2021, ICML): inter-task prior transfer
        MARS (2025, arXiv:2508.01173): heterogeneous agent ensemble
        Regime-Aware RL (Nixon 2025): HMM regime + risk aversion γₖ
    """

    # 6 groups × 5-7 signals each = 28 total signals
    SIGNAL_GROUPS: Dict[str, List[str]] = {
        "technical_trend": [
            "ema_cross_fast",       # EMA(9) vs EMA(21) crossover
            "ema_cross_slow",       # EMA(21) vs EMA(50) crossover
            "macd_histogram",       # MACD line vs signal divergence
            "trend_strength",       # ADX > 25 trending detection
            "supertrend",           # ATR-based trend direction
        ],
        "technical_reversion": [
            "rsi_signal",           # RSI(14) overbought/oversold
            "stoch_rsi",            # Stochastic RSI crossovers
            "bb_squeeze",           # Bollinger Band width contraction
            "bb_deviation",         # Price distance from BB middle
            "vwap_deviation",       # Price vs session VWAP
            "support_resistance",   # Key S/R level proximity
        ],
        "technical_volume": [
            "volume_spike",         # Volume > 2x 20-period average
            "cvd_signal",           # Cumulative Volume Delta (buy vs sell pressure)
            "obv_divergence",       # OBV vs price divergence
            "volume_profile",       # Volume at price (POC proximity)
            "mfi_signal",           # Money Flow Index extremes
        ],
        "derivatives": [
            "funding_rate",         # Binance perp funding rate
            "oi_change",            # Open interest change rate
            "long_short_ratio",     # Top trader long/short ratio
            "liquidation_level",    # Nearby liquidation cluster density
            "basis_spread",         # Futures premium vs spot (annualized)
        ],
        "sentiment": [
            "news_sentiment",       # FinBERT / headline NLP score
            "fear_greed",           # Crypto Fear & Greed Index
            "social_buzz",          # Social media volume/sentiment
            "whale_activity",       # Large transfer alerts (>$10M)
            "exchange_flow",        # Net exchange inflow/outflow
        ],
        "macro": [
            "market_regime",        # HMM/rule-based regime label
            "volatility_regime",    # ATR/BB-based vol state
            "btc_dominance",        # BTC.D percentage trend
            "dxy_direction",        # Dollar index direction
            "etf_flow",             # BTC ETF daily net flow
            "stablecoin_flow",      # Stablecoin supply change
        ],
    }

    # Flat list of all signal names (for iteration)
    ALL_SIGNALS: List[str] = []
    for _sigs in SIGNAL_GROUPS.values():
        ALL_SIGNALS.extend(_sigs)

    # Reverse lookup: signal_name → group_name
    SIGNAL_TO_GROUP: Dict[str, str] = {}
    for _g, _sigs in SIGNAL_GROUPS.items():
        for _s in _sigs:
            SIGNAL_TO_GROUP[_s] = _g

    GROUP_NAMES: List[str] = list(SIGNAL_GROUPS.keys())

    # Level 0: Meta-parameters — "how to trade" per regime
    # Each is a Beta(2,2) prior, mean=0.5. Interpretation depends on param:
    #   position_scale:      low mean → size down, high mean → size up
    #   entry_selectivity:   low mean → broad entry, high mean → picky
    #   hold_patience:       low mean → quick exit, high mean → hold longer
    #   trade_frequency:     low mean → sit out, high mean → active
    #   trend_vs_reversion:  low mean → mean-revert, high mean → trend-follow
    #   risk_aversion:       low mean → aggressive, high mean → conservative
    META_PARAMS: List[str] = [
        "position_scale",
        "entry_selectivity",
        "hold_patience",
        "trade_frequency",
        "trend_vs_reversion",
        "risk_aversion",
    ]

    def __init__(
        self,
        save_path: Optional[str] = None,
        min_trades_to_adapt: int = 5,
        max_window: int = 100,
        group_discount: float = 0.98,     # Groups are stable across regimes
        signal_discount: float = 0.95,    # Signals adapt faster
    ) -> None:
        self._save_path = Path(save_path) if save_path else None
        self._min_trades = min_trades_to_adapt
        self._max_window = max_window
        self._group_discount = group_discount
        self._signal_discount = signal_discount

        # Level 0: meta-parameter Betas per regime
        # {regime: {meta_param_name: AgentBeta}}
        self._meta_betas: Dict[str, Dict[str, AgentBeta]] = {
            _GLOBAL_REGIME: {p: AgentBeta(name=p) for p in self.META_PARAMS},
        }

        # Level 1: group-level Betas per regime
        # {regime: {group_name: AgentBeta}}
        self._group_betas: Dict[str, Dict[str, AgentBeta]] = {
            _GLOBAL_REGIME: {g: AgentBeta(name=g) for g in self.GROUP_NAMES},
        }

        # Level 2: signal-level Betas per regime per group
        # {regime: {group_name: {signal_name: AgentBeta}}}
        self._signal_betas: Dict[str, Dict[str, Dict[str, AgentBeta]]] = {
            _GLOBAL_REGIME: {
                g: {s: AgentBeta(name=s) for s in sigs}
                for g, sigs in self.SIGNAL_GROUPS.items()
            },
        }

        self._trades: List[TradeRecord] = []
        self._total_pnl: float = 0.0
        self._regime_trade_counts: Dict[str, int] = {_GLOBAL_REGIME: 0}

    def _ensure_regime(self, regime: str) -> None:
        """Create Beta structures for a new regime if needed."""
        if regime not in self._group_betas:
            self._group_betas[regime] = {
                g: AgentBeta(name=g) for g in self.GROUP_NAMES
            }
            self._signal_betas[regime] = {
                g: {s: AgentBeta(name=s) for s in sigs}
                for g, sigs in self.SIGNAL_GROUPS.items()
            }
            self._meta_betas[regime] = {
                p: AgentBeta(name=p) for p in self.META_PARAMS
            }
            self._regime_trade_counts.setdefault(regime, 0)

    def _effective_regime(self, regime: str) -> str:
        """Return regime key to use, falling back to global if insufficient data."""
        count = self._regime_trade_counts.get(regime, 0)
        if count >= _MIN_REGIME_TRADES and regime in self._group_betas:
            return regime
        return _GLOBAL_REGIME

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def has_enough_data(self) -> bool:
        return self.total_trades >= self._min_trades

    # ------------------------------------------------------------------
    # Core: record trade outcome + update posteriors (both levels)
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
        position_pct_used: float = 0.0,
        confidence_at_entry: float = 0.0,
    ) -> Dict[str, Any]:
        """Record a completed trade and update group + signal + meta posteriors.

        Trades held < 5 minutes are recorded in history but DO NOT update
        posteriors (likely position-restore artifacts, not real decisions).
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

        # Skip posterior update for trades held < 5 minutes (artifacts)
        if held_hours < 5.0 / 60.0:
            logger.info("[h-ts] Trade #%d: %s skipped posterior update (held %.1f min < 5 min threshold)",
                        len(self._trades), ticker, held_hours * 60)
            if self._save_path:
                self.save()
            return {"trade_number": len(self._trades), "ticker": ticker, "pnl_pct": pnl_pct,
                    "skipped": True, "reason": "held < 5 minutes"}

        if len(self._trades) > self._max_window:
            self._trades = self._trades[-self._max_window:]

        # Determine target regimes to update
        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)
            self._regime_trade_counts[regime] = self._regime_trade_counts.get(regime, 0) + 1
        self._regime_trade_counts[_GLOBAL_REGIME] = self._regime_trade_counts.get(_GLOBAL_REGIME, 0) + 1

        profitable = pnl_pct > 0
        updates = {}
        active_groups: Dict[str, float] = {}  # group_name → aggregate pnl for group update

        # Level 2: update individual signal Betas
        for sig_name, signal_value in agent_signals.items():
            if abs(signal_value) < 0.05:
                continue

            group = self.SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                # Unknown signal name — possibly old format, skip
                continue

            signal_bullish = signal_value > 0
            aligned = (signal_bullish and profitable) or (not signal_bullish and not profitable)
            effective_pnl = abs(pnl_pct) if aligned else -abs(pnl_pct)

            for r in target_regimes:
                sig_beta = self._signal_betas[r][group].get(sig_name)
                if sig_beta is None:
                    sig_beta = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sig_beta
                sig_beta.update(pnl_pct=effective_pnl, discount=self._signal_discount)

            # Track which groups were active (for Level 1 update)
            # Use the trade outcome directly — multiple signals in same group
            # should reinforce the group, not dilute via averaging
            if group not in active_groups:
                active_groups[group] = effective_pnl
            # Keep first value (trade outcome); don't average with subsequent signals

            updates[sig_name] = {
                "group": group,
                "signal": round(signal_value, 3),
                "aligned": aligned,
                "new_mean": round(
                    self._signal_betas[_GLOBAL_REGIME][group][sig_name].mean, 4
                ),
            }

        # Level 1: update group Betas
        for group_name, group_pnl in active_groups.items():
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=group_pnl, discount=self._group_discount
                )

        # Level 0: update meta-parameter Betas
        meta_updates = self._update_meta_params(
            pnl_pct=pnl_pct,
            held_hours=held_hours,
            position_pct_used=position_pct_used,
            confidence_at_entry=confidence_at_entry,
            agent_signals=agent_signals,
            target_regimes=target_regimes,
        )

        logger.info(
            "[h-ts] Trade #%d: %s pnl=%+.2f%% regime=%s groups=%s signals=%s meta=%s",
            self.total_trades, ticker, pnl_pct * 100, regime,
            list(active_groups.keys()),
            {k: v["aligned"] for k, v in updates.items()},
            {k: f"{v:+.3f}" for k, v in meta_updates.items()} if meta_updates else "N/A",
        )

        if self._save_path and not getattr(self, '_reprocessing', False):
            self.save()

        return {
            "trade_number": self.total_trades,
            "ticker": ticker,
            "pnl_pct": pnl_pct,
            "profitable": profitable,
            "regime": regime,
            "active_groups": list(active_groups.keys()),
            "signal_updates": updates,
            "cumulative_pnl": self._total_pnl,
        }

    # ------------------------------------------------------------------
    # Counterfactual Learning: learn from missed opportunities
    # When Claude says HOLD but price moves significantly, update posteriors
    # with reduced weight so H-TS learns what it's missing.
    #
    # This breaks selection bias: without this, H-TS only learns from
    # executed trades, so strategies it never tries never get data.
    # ------------------------------------------------------------------
    def record_counterfactual(
        self,
        ticker: str,
        price_at_hold: float,
        price_now: float,
        ta_signals: Dict[str, float],
        regime: str = "unknown",
        discount_factor: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """Record a phantom trade from a missed opportunity.

        Called when Claude decided HOLD but price moved significantly.
        Updates posteriors with reduced weight (discount_factor x actual pnl).

        Args:
            ticker: The asset that moved
            price_at_hold: Price when Claude said HOLD
            price_now: Current price (after the move)
            ta_signals: TA-derived signals at hold time {signal_name: value}
                        Positive = bullish signal was present
            regime: Market regime at hold time
            discount_factor: Weight relative to real trade (default 0.3 = 30%)

        Returns:
            Summary dict or None if move was too small.
        """
        if price_at_hold <= 0:
            return None

        raw_pnl = (price_now - price_at_hold) / price_at_hold
        # Only learn from significant moves (> fee threshold)
        if abs(raw_pnl) < 0.003:  # < 0.3% = noise, skip
            return None

        # Discount the pnl — phantom trades carry less weight than real ones
        effective_pnl = raw_pnl * discount_factor

        # Determine target regimes
        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)

        updates = {}
        active_groups: Dict[str, float] = {}

        for sig_name, signal_value in ta_signals.items():
            if abs(signal_value) < 0.05:
                continue
            group = self.SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                continue

            # For counterfactual: bullish signal + price went up = signal was right
            signal_bullish = signal_value > 0
            price_went_up = raw_pnl > 0
            aligned = (signal_bullish and price_went_up) or (not signal_bullish and not price_went_up)
            sig_effective = abs(effective_pnl) if aligned else -abs(effective_pnl)

            for r in target_regimes:
                sb = self._signal_betas[r][group].get(sig_name)
                if sb is None:
                    sb = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sb
                sb.update(pnl_pct=sig_effective, discount=self._signal_discount)

            if group not in active_groups:
                active_groups[group] = sig_effective
            # Keep first value; don't dilute via averaging

            updates[sig_name] = {
                "group": group, "signal": round(signal_value, 3),
                "aligned": aligned,
            }

        # Group-level update
        for group_name, group_pnl in active_groups.items():
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=group_pnl, discount=self._group_discount
                )

        if updates:
            logger.info(
                "[h-ts] Counterfactual %s: raw_pnl=%+.2f%% effective=%+.2f%% regime=%s signals=%s",
                ticker, raw_pnl * 100, effective_pnl * 100, regime,
                {k: v["aligned"] for k, v in updates.items()},
            )
            if self._save_path:
                self.save()

        return {
            "ticker": ticker,
            "raw_pnl": raw_pnl,
            "effective_pnl": effective_pnl,
            "regime": regime,
            "signal_updates": updates,
        } if updates else None

    # ------------------------------------------------------------------
    # Level 0: Meta-parameter learning — "how to trade" per regime
    # Papers: Meta-TS (Kveton 2021), MARS (2025), Regime-Aware RL (Nixon 2025),
    #         Behaviorally Informed DRL (Sci Reports 2026), Kelly Criterion
    # ------------------------------------------------------------------

    def _update_meta_params(
        self,
        pnl_pct: float,
        held_hours: float,
        position_pct_used: float,
        confidence_at_entry: float,
        agent_signals: Dict[str, float],
        target_regimes: List[str],
    ) -> Dict[str, float]:
        """Update Level 0 meta-parameter Betas based on trade outcome.

        Each meta-param has a custom reward shaping that captures its semantics.
        Uses group_discount (0.98) since meta-params should be stable like groups.

        Returns dict of {param: effective_pnl} for logging.
        """
        if not target_regimes:
            return {}

        meta_updates: Dict[str, float] = {}

        # 1. position_scale: large position + win → alpha, large position + loss → beta
        pos_scale = position_pct_used / 0.10 if position_pct_used > 0 else 1.0
        eff_position = pnl_pct * pos_scale
        meta_updates["position_scale"] = eff_position

        # 2. entry_selectivity: high confidence + win → alpha, high confidence + loss → beta
        conf_scale = confidence_at_entry if confidence_at_entry > 0 else 0.5
        eff_selectivity = pnl_pct * conf_scale
        meta_updates["entry_selectivity"] = eff_selectivity

        # 3. hold_patience: longer hold + win → alpha, longer hold + loss → beta
        hold_scale = min(held_hours / 0.5, 2.0)
        eff_patience = pnl_pct * hold_scale
        meta_updates["hold_patience"] = eff_patience

        # 4. trade_frequency: pure outcome (win → alpha, loss → beta)
        meta_updates["trade_frequency"] = pnl_pct

        # 5. trend_vs_reversion: trend-following signals + win → alpha (trend works)
        #    reversion signals + win → negative effective (reversion works = low mean)
        trend_groups = {"technical_trend"}
        reversion_groups = {"technical_reversion"}
        trend_weight = 0.0
        reversion_weight = 0.0
        for sig_name, sig_val in agent_signals.items():
            group = self.SIGNAL_TO_GROUP.get(sig_name)
            if group in trend_groups and abs(sig_val) >= 0.05:
                trend_weight += abs(sig_val)
            elif group in reversion_groups and abs(sig_val) >= 0.05:
                reversion_weight += abs(sig_val)
        total_tr_weight = trend_weight + reversion_weight
        if total_tr_weight > 0:
            # Net direction: positive = trend-dominated, negative = reversion-dominated
            direction = (trend_weight - reversion_weight) / total_tr_weight
            eff_trend = pnl_pct * direction
        else:
            eff_trend = 0.0
        meta_updates["trend_vs_reversion"] = eff_trend

        # 6. risk_aversion: sign-flipped × position scale
        #    Big loss → alpha++ (learn to be more cautious)
        #    Big win → beta++ (OK to be aggressive)
        eff_risk = -pnl_pct * pos_scale
        meta_updates["risk_aversion"] = eff_risk

        # Apply updates to all target regimes
        for param, eff_pnl in meta_updates.items():
            if abs(eff_pnl) < 1e-8:
                continue
            for r in target_regimes:
                mb = self._meta_betas.get(r, {}).get(param)
                if mb:
                    mb.update(pnl_pct=eff_pnl, discount=self._group_discount)

        return meta_updates

    def sample_meta_params(self, regime: str = "unknown") -> Dict[str, float]:
        """Sample meta-parameters from Beta posteriors (Thompson Sampling).

        Returns {param_name: sampled_value} where values are in (0, 1).
        Uses hierarchical fallback: regime → global.
        """
        eff_regime = self._effective_regime(regime)
        meta = self._meta_betas.get(eff_regime, self._meta_betas[_GLOBAL_REGIME])
        return {p: mb.sample() for p, mb in meta.items()}

    def get_meta_param_means(self, regime: str = "unknown") -> Dict[str, float]:
        """Get deterministic meta-parameter means (no sampling noise). For display."""
        eff_regime = self._effective_regime(regime)
        meta = self._meta_betas.get(eff_regime, self._meta_betas[_GLOBAL_REGIME])
        return {p: round(mb.mean, 4) for p, mb in meta.items()}

    # ------------------------------------------------------------------
    # Weight sampling (2-level Thompson Sampling)
    # ------------------------------------------------------------------

    def sample_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Sample weights via 2-level hierarchical Thompson Sampling.

        1. Sample from Level 1 (group Betas) → group weights
        2. Sample from Level 2 (signal Betas within each group) → signal weights
        3. Final weight = group_weight × within_group_weight (normalized)
        """
        if not self.has_enough_data:
            return self._default_weights()

        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        signal_betas = self._signal_betas[eff_regime]

        # Level 1: sample group weights
        group_samples = {g: gb.sample() for g, gb in group_betas.items()}
        total_group = sum(group_samples.values())
        if total_group <= 0:
            return self._default_weights()

        # Level 2: sample signal weights within each group
        final_weights: Dict[str, float] = {}
        for group_name, sigs in self.SIGNAL_GROUPS.items():
            group_w = group_samples[group_name] / total_group

            sig_samples = {}
            for sig_name in sigs:
                sb = signal_betas[group_name].get(sig_name)
                if sb:
                    sig_samples[sig_name] = sb.sample()
                else:
                    sig_samples[sig_name] = 0.5  # uninformative default

            total_sig = sum(sig_samples.values())
            if total_sig <= 0:
                total_sig = 1.0

            for sig_name, sv in sig_samples.items():
                final_weights[sig_name] = group_w * (sv / total_sig)

        return final_weights

    def get_mean_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Deterministic mean weights (no sampling noise). For display."""
        if not self.has_enough_data:
            return self._default_weights()

        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        signal_betas = self._signal_betas[eff_regime]

        group_means = {g: gb.mean for g, gb in group_betas.items()}
        total_group = sum(group_means.values())
        if total_group <= 0:
            return self._default_weights()

        final: Dict[str, float] = {}
        for group_name, sigs in self.SIGNAL_GROUPS.items():
            group_w = group_means[group_name] / total_group

            sig_means = {}
            for sig_name in sigs:
                sb = signal_betas[group_name].get(sig_name)
                sig_means[sig_name] = sb.mean if sb else 0.5

            total_sig = sum(sig_means.values())
            if total_sig <= 0:
                total_sig = 1.0

            for sig_name, sv in sig_means.items():
                final[sig_name] = group_w * (sv / total_sig)

        return final

    def get_group_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Get group-level mean weights only (for dashboard summary)."""
        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        raw = {g: gb.mean for g, gb in group_betas.items()}
        total = sum(raw.values())
        if total <= 0:
            return {g: 1.0 / len(self.GROUP_NAMES) for g in self.GROUP_NAMES}
        return {g: v / total for g, v in raw.items()}

    @staticmethod
    def _default_weights() -> Dict[str, float]:
        """Uniform weights when no data yet."""
        n = len(HierarchicalOnlineLearner.ALL_SIGNALS)
        return {s: 1.0 / n for s in HierarchicalOnlineLearner.ALL_SIGNALS}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if self._save_path is None:
            return

        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 4,  # v4 = hierarchical + meta-parameters (Level 0)
            "meta_betas": {
                regime: {p: mb.to_dict() for p, mb in params.items()}
                for regime, params in self._meta_betas.items()
            },
            "group_betas": {
                regime: {g: gb.to_dict() for g, gb in groups.items()}
                for regime, groups in self._group_betas.items()
            },
            "signal_betas": {
                regime: {
                    g: {s: sb.to_dict() for s, sb in sigs.items()}
                    for g, sigs in groups.items()
                }
                for regime, groups in self._signal_betas.items()
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
        tmp = self._save_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._save_path)
        logger.debug("[h-ts] Saved state (%d trades, %d regimes, %d signals)",
                     self.total_trades, len(self._group_betas) - 1,
                     len(self.ALL_SIGNALS))

    def load(self) -> bool:
        if self._save_path is None or not self._save_path.exists():
            return False

        try:
            state = json.loads(self._save_path.read_text())
            version = state.get("version", 1)

            if version < 3:
                # Old flat OnlineLearner state → migrate
                return self._migrate_from_v1(state)

            # Load v3/v4 hierarchical state
            # v3→v4 migration: meta_betas absent → default priors (no error)
            for regime, params in state.get("meta_betas", {}).items():
                if regime != _GLOBAL_REGIME:
                    self._ensure_regime(regime)
                for p, data in params.items():
                    if p in self._meta_betas.get(regime, {}):
                        self._meta_betas[regime][p].alpha = data["alpha"]
                        self._meta_betas[regime][p].beta = data["beta"]
                        self._meta_betas[regime][p].total_trades = data["total_trades"]

            for regime, groups in state.get("group_betas", {}).items():
                self._ensure_regime(regime) if regime != _GLOBAL_REGIME else None
                for g, data in groups.items():
                    if g in self._group_betas.get(regime, {}):
                        self._group_betas[regime][g].alpha = data["alpha"]
                        self._group_betas[regime][g].beta = data["beta"]
                        self._group_betas[regime][g].total_trades = data["total_trades"]

            for regime, groups in state.get("signal_betas", {}).items():
                if regime not in self._signal_betas:
                    self._ensure_regime(regime) if regime != _GLOBAL_REGIME else None
                for g, sigs in groups.items():
                    if g not in self._signal_betas.get(regime, {}):
                        continue
                    for s, data in sigs.items():
                        if s not in self._signal_betas[regime][g]:
                            self._signal_betas[regime][g][s] = AgentBeta(name=s)
                        sb = self._signal_betas[regime][g][s]
                        sb.alpha = data["alpha"]
                        sb.beta = data["beta"]
                        sb.total_trades = data["total_trades"]

            self._regime_trade_counts = state.get("regime_trade_counts", {_GLOBAL_REGIME: 0})
            self._total_pnl = state.get("total_pnl", 0.0)

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

            logger.info(
                "[h-ts] Loaded v3 state: %d trades, %d regimes, %d groups, pnl=%+.2f%%",
                self.total_trades, len(self._group_betas) - 1,
                len(self.GROUP_NAMES), self._total_pnl * 100,
            )

            # v3→v4 migration: bootstrap meta betas from existing trades
            if "meta_betas" not in state and self._trades:
                logger.info(
                    "[h-ts] v3→v4: bootstrapping meta-parameters from %d existing trades...",
                    len(self._trades),
                )
                self.reprocess_all_trades()
            else:
                # Detect trades held < 5 minutes that polluted posteriors
                short_trades = sum(1 for t in self._trades if t.held_hours < 5.0 / 60.0 and abs(t.pnl_pct) > 1e-8)
                if short_trades > 0:
                    logger.warning(
                        "[h-ts] Found %d trades held < 5 min (position-restore artifacts). "
                        "Reprocessing to exclude them from posteriors.",
                        short_trades,
                    )
                    self.reprocess_all_trades()

            return True

        except Exception as exc:
            logger.warning("[h-ts] Failed to load state: %s", exc)
            return False

    def _migrate_from_v1(self, state: Dict) -> bool:
        """Migrate old flat 8-arm OnlineLearner state to hierarchical v3.

        Maps old signal names to new ones, reprocesses trade history with
        expanded signal names.
        """
        logger.info("[h-ts] Migrating from v1/v2 flat state to v3 hierarchical...")

        self._total_pnl = state.get("total_pnl", 0.0)
        self._regime_trade_counts = state.get("regime_trade_counts", {_GLOBAL_REGIME: 0})

        # Restore trade records and remap old signal names
        self._trades = []
        for t in state.get("last_trades", []):
            old_signals = t.get("agent_signals", {})
            new_signals = self._remap_old_signals(old_signals)

            self._trades.append(TradeRecord(
                ticker=t["ticker"],
                entry_price=0.0,
                exit_price=0.0,
                pnl_pct=t["pnl_pct"],
                held_hours=t.get("held_hours", 0.0),
                agent_signals=new_signals,
                entry_time=0.0,
                exit_time=t.get("exit_time", 0.0),
                market_type=t.get("market_type", "crypto"),
                regime=t.get("regime", "unknown"),
            ))

        # Reprocess all trades with new signal names
        old_trades = list(self._trades)
        self._trades = []
        self._total_pnl = 0.0
        self._regime_trade_counts = {_GLOBAL_REGIME: 0}

        for t in old_trades:
            if abs(t.pnl_pct) < 1e-8:
                continue
            self.record_trade(
                ticker=t.ticker,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pct=t.pnl_pct,
                held_hours=t.held_hours,
                agent_signals=t.agent_signals,
                market_type=t.market_type,
                regime=t.regime,
            )

        logger.info(
            "[h-ts] Migration complete: %d trades reprocessed with %d signal types",
            len(self._trades), len(self.ALL_SIGNALS),
        )
        return True

    @staticmethod
    def _remap_old_signals(old_signals: Dict[str, float]) -> Dict[str, float]:
        """Map old 8-category signal names to new granular names.

        When the old signal is e.g. {"momentum": 0.7}, we spread it evenly
        across the new signals it maps to: {"price_momentum": 0.7, "ema_cross_fast": 0.7, ...}
        """
        new_signals: Dict[str, float] = {}
        for old_name, value in old_signals.items():
            new_names = _OLD_TO_NEW_MAPPING.get(old_name)
            if new_names:
                for nn in new_names:
                    new_signals[nn] = value
            # If old name IS a valid new signal name, keep it
            elif old_name in HierarchicalOnlineLearner.SIGNAL_TO_GROUP:
                new_signals[old_name] = value
        return new_signals

    # ------------------------------------------------------------------
    # Reprocessing
    # ------------------------------------------------------------------

    def reprocess_all_trades(self) -> None:
        """Reset all posteriors and re-learn from trade history."""
        old_trades = list(self._trades)
        n = len(old_trades)

        # Reset
        self._meta_betas = {
            _GLOBAL_REGIME: {p: AgentBeta(name=p) for p in self.META_PARAMS},
        }
        self._group_betas = {
            _GLOBAL_REGIME: {g: AgentBeta(name=g) for g in self.GROUP_NAMES},
        }
        self._signal_betas = {
            _GLOBAL_REGIME: {
                g: {s: AgentBeta(name=s) for s in sigs}
                for g, sigs in self.SIGNAL_GROUPS.items()
            },
        }
        self._trades = []
        self._total_pnl = 0.0
        self._regime_trade_counts = {_GLOBAL_REGIME: 0}

        for t in old_trades:
            if abs(t.pnl_pct) < 1e-8:
                continue
            self.record_trade(
                ticker=t.ticker,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pct=t.pnl_pct,
                held_hours=t.held_hours,
                agent_signals=t.agent_signals,
                market_type=t.market_type,
                regime=t.regime,
            )

        logger.info("[h-ts] Reprocessed %d trades → %d valid.", n, len(self._trades))

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        global_groups = self._group_betas.get(_GLOBAL_REGIME, {})
        global_signals = self._signal_betas.get(_GLOBAL_REGIME, {})

        # Group-level summary
        group_summary = {}
        for g, gb in global_groups.items():
            signals_in_group = {}
            for s, sb in global_signals.get(g, {}).items():
                signals_in_group[s] = sb.to_dict()
            group_summary[g] = {
                "group_beta": gb.to_dict(),
                "signals": signals_in_group,
            }

        # Meta-parameter summaries
        global_meta = self._meta_betas.get(_GLOBAL_REGIME, {})
        meta_params_global = {p: mb.to_dict() for p, mb in global_meta.items()}

        # Regime info
        regime_info = {}
        for regime in self._group_betas:
            if regime == _GLOBAL_REGIME:
                continue
            count = self._regime_trade_counts.get(regime, 0)
            regime_meta = self._meta_betas.get(regime, {})
            regime_info[regime] = {
                "trade_count": count,
                "using_own_weights": count >= _MIN_REGIME_TRADES,
                "group_means": {
                    g: round(gb.mean, 4)
                    for g, gb in self._group_betas[regime].items()
                },
                "meta_param_means": {
                    p: round(mb.mean, 4)
                    for p, mb in regime_meta.items()
                },
            }

        return {
            "version": 4,
            "total_trades": self.total_trades,
            "has_enough_data": self.has_enough_data,
            "min_trades_to_adapt": self._min_trades,
            "cumulative_pnl_pct": round(self._total_pnl * 100, 2),
            "num_groups": len(self.GROUP_NAMES),
            "num_signals": len(self.ALL_SIGNALS),
            "group_summary": group_summary,
            "meta_params_global": meta_params_global,
            "meta_param_means_global": self.get_meta_param_means(),
            "regime_info": regime_info,
            "regime_trade_counts": dict(self._regime_trade_counts),
            "group_weights_global": self.get_group_weights(),
            "mean_weights_global": self.get_mean_weights(),
            "recent_trades": [
                {
                    "ticker": t.ticker,
                    "pnl_pct": t.pnl_pct,
                    "held_hours": t.held_hours,
                    "agent_signals": t.agent_signals,
                    "regime": t.regime,
                    "market": t.market_type,
                }
                for t in self._trades[-10:]
            ],
        }

    # ------------------------------------------------------------------
    # Diary L3 → H-TS Prior Bridge
    # Papers: TradingGroup (Tian 2025) — similarity-based lesson retrieval
    #         LLM Regret (Park 2024) — regret-weighted prior elicitation
    # ------------------------------------------------------------------
    def apply_diary_lessons(self, lessons: list[dict]) -> int:
        """Inject L3 semantic lessons into H-TS as Bayesian prior adjustments.

        Each lesson has:
            text, regime_scope, signals (list), strength, occurrences, decay_factor

        Mechanism:
            effective = strength * decay_factor
            boost = effective * min(occurrences, 5) * BOOST_FACTOR
            → signal Beta(α += boost), group Beta(α += boost * 0.5)

        Returns number of adjustments made.
        """
        BOOST_FACTOR = 0.15  # ~15% of a real trade update
        MIN_EFFECTIVE = 0.3  # skip weak/decayed lessons
        MAX_BOOST = 0.5      # cap per-signal boost

        adjustments = 0

        for lesson in lessons:
            effective = lesson.get("strength", 0.5) * lesson.get("decay_factor", 1.0)
            if effective < MIN_EFFECTIVE:
                continue

            signals = lesson.get("signals", [])
            if not signals:
                continue

            occurrences = min(lesson.get("occurrences", 1), 5)
            boost = min(effective * occurrences * BOOST_FACTOR, MAX_BOOST)

            regime_scope = lesson.get("regime_scope", "*")

            # Find matching regimes (compare base components, not substrings)
            target_regimes = []
            if regime_scope == "*":
                target_regimes = list(self._group_betas.keys())
            else:
                scope_parts = set(regime_scope.replace("_", " ").split())
                for regime in self._group_betas:
                    regime_parts = set(regime.replace("_", " ").split())
                    # Match if scope's key words are subset of regime
                    if scope_parts & regime_parts:
                        target_regimes.append(regime)
                if not target_regimes:
                    target_regimes = [_GLOBAL_REGIME]

            for regime in target_regimes:
                for sig_name in signals:
                    group = self.SIGNAL_TO_GROUP.get(sig_name)
                    if not group:
                        continue

                    # Ensure regime exists in betas
                    if regime not in self._signal_betas:
                        continue

                    sig_betas = self._signal_betas[regime].get(group, {})
                    sb = sig_betas.get(sig_name)
                    if sb:
                        # Outcome-aware: win lessons boost alpha, loss lessons boost beta
                        outcome = lesson.get("outcome", "unknown")
                        if outcome == "loss":
                            sb.beta += boost
                        else:
                            sb.alpha += boost
                        adjustments += 1

                    # Group-level gets half boost (same direction)
                    gb = self._group_betas.get(regime, {}).get(group)
                    if gb:
                        if outcome == "loss":
                            gb.beta += boost * 0.5
                        else:
                            gb.alpha += boost * 0.5

        if adjustments > 0:
            self.save()
            logger.info(
                "[H-TS] Diary prior injection: %d signal adjustments from %d lessons",
                adjustments, len(lessons),
            )

        return adjustments
