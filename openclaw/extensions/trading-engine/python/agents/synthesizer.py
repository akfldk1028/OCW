"""Multi-agent signal synthesizer — weighted voting (v8).

Combines signals from MarketAgent, QuantAgent (v8), and FinBERT sentiment
into final buy/sell/hold decisions with confidence scores and position
sizing.  QuantAgent's P(top-quartile) is the highest-weighted signal.

v8 changes:
    - QuantAgent now predicts P(top-quartile) instead of P(outperform SPY)
    - Z-scored features preserve signal magnitude
    - 13 features (momentum_21d replaces momentum_30d)
    - Backtest v8: +116% vs SPY +78%, Sharpe 1.28, Payoff 7.93x

Signal weights (RL disabled — see docs/profitability-assessment.md):
    market    0.24 — regime + sector momentum
    quant     0.35 — XGBoost P(top-quartile) [primary signal]
    rl        0.00 — disabled (50K timesteps = noise)
    sentiment 0.12 — FinBERT news sentiment
    momentum  0.18 — raw price momentum
    regime    0.11 — defensive/offensive bias
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import SWING_EXIT_CONFIG, TRANSACTION_COSTS
from core.event_bus import EventBus

logger = logging.getLogger("trading-engine.agents.synthesizer")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthesizer")


# Agent signal weights — RL removed (Cargo Cult ML at 50K timesteps)
# See docs/profitability-assessment.md for evidence
AGENT_WEIGHTS = {
    "market": 0.24,     # regime + sector (was 0.20)
    "quant": 0.35,      # XGBoost factor — primary signal (was 0.30)
    "rl": 0.00,         # RL disabled — noise at 50K timesteps
    "sentiment": 0.12,  # FinBERT (was 0.10)
    "momentum": 0.18,   # raw price momentum (was 0.15)
    "regime": 0.11,     # defensive/offensive bias (was 0.10)
}

# Confidence thresholds — lowered to allow more candidates through
THRESHOLDS = {
    "strong_buy": 0.45,
    "buy": 0.20,
    "hold": 0.10,
    "sell": -0.15,
    "strong_sell": -0.35,
}

# Position sizing — raised to meaningful allocations
MAX_POSITION_PCT = 0.15
MAX_EXPOSURE = 0.80


@dataclass
class AgentSignal:
    """Signal from a single agent."""

    agent: str
    ticker: str
    score: float       # [-1, 1] normalized
    confidence: float  # [0, 1]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Final trading decision for one ticker."""

    ticker: str
    action: str        # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    confidence: float  # [-1, 1]
    size_pct: float    # position size as % of portfolio
    size_usd: float    # position size in USD
    price: float
    reasons: List[str]
    signals: Dict[str, float]  # per-agent signal breakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "position_size_pct": round(self.size_pct, 4),
            "position_size_usd": round(self.size_usd, 2),
            "price": self.price,
            "reasons": self.reasons,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


class Synthesizer:
    """Multi-agent signal synthesizer — weighted voting.

    Aggregates outputs from:
        - MarketAgent.analyze() → regime bias, sector picks
        - QuantAgent.rank_stocks() → P(outperform) rankings
        - FinBERT sentiment scores
        - RiskManager filter

    Usage::

        synth = Synthesizer(event_bus, risk_manager, finbert_scorer)
        decisions = await synth.synthesize(
            market_view=market_view,
            quant_rankings=quant_rankings,
            portfolio_value=100000,
        )
    """

    def __init__(
        self,
        event_bus: EventBus,
        risk_manager: Any,
        finbert_scorer: Any = None,
        sector_scanner: Any = None,
        ensemble_agent: Any = None,
        data_processor: Any = None,
    ) -> None:
        self._bus = event_bus
        self._risk = risk_manager
        self._finbert = finbert_scorer
        self._scanner = sector_scanner
        self._ensemble = ensemble_agent
        self._data_proc = data_processor
        self._last_decisions: List[Decision] = []
        self._rl_signals: Dict[str, float] = {}  # cached RL signals per ticker
        self._scan_miss_counts: Dict[str, int] = {}  # ticker -> consecutive misses

    async def synthesize(
        self,
        market_view: Any,
        quant_rankings: List[Any],
        portfolio_value: float = 100_000.0,
        current_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        include_sentiment: bool = True,
        adaptive_weights: Optional[Dict[str, float]] = None,
    ) -> List[Decision]:
        """Combine all agent signals into final decisions.

        Steps:
            1. Extract sector signal from MarketView
            2. Use QuantAgent P(outperform) as primary signal
            3. Score FinBERT sentiment on news headlines
            4. Weighted combination → final confidence
            5. Position sizing (Kelly fraction)
            6. RiskManager filter
        """
        if current_positions is None:
            current_positions = {}

        loop = asyncio.get_running_loop()

        # Gather candidate tickers from quant rankings
        # v8: P(top-quartile) > 0.40 threshold (calibrated for ~25% base rate)
        candidates = [r for r in quant_rankings if r.p_outperform > 0.40]
        candidate_ticker_set = {c.ticker for c in candidates}

        if not candidates and not current_positions:
            logger.info("Synthesizer: no candidates and no positions to evaluate")
            return []

        candidate_tickers = [c.ticker for c in candidates]

        # Get sentiment scores (if enabled)
        sentiment_map: Dict[str, float] = {}
        if include_sentiment and self._finbert is not None:
            sentiment_map = await loop.run_in_executor(
                _executor,
                self._score_sentiment_batch,
                candidate_tickers,
            )

        # Get RL ensemble signals (PPO+A2C+SAC)
        rl_signal_map: Dict[str, float] = {}
        if self._ensemble is not None and self._ensemble.is_trained:
            rl_signal_map = await loop.run_in_executor(
                _executor,
                self._get_rl_signals,
                candidate_tickers,
            )
            self._rl_signals = rl_signal_map

        # Compute sector signal map
        sector_signal_map = self._build_sector_signals(market_view)

        # Build decisions
        decisions: List[Decision] = []
        for rank in candidates:
            tic = rank.ticker

            # 1. Market signal: sector relative strength
            sector_sig = sector_signal_map.get(tic, 0.0)

            # 2. Quant signal: P(outperform SPY) → normalized to [-1, 1]
            quant_sig = (rank.p_outperform - 0.5) * 2  # 0.5→0, 0.75→0.5, 1.0→1.0

            # 3. Sentiment signal
            sent_sig = sentiment_map.get(tic, 0.0)

            # 4. RL ensemble signal (PPO+A2C+SAC action value)
            rl_sig = rl_signal_map.get(tic, 0.0)

            # 5. Momentum signal (from z-scored features — v8: momentum_21d)
            # Z-scores are already in [-3, 3], normalize to [-1, 1]
            mom_z = rank.rank_features.get("momentum_21d", 0.0)
            mom_sig = max(-1.0, min(1.0, mom_z / 3.0))

            # 6. Regime bias
            regime_sig = self._regime_signal(market_view, tic)

            # Weighted combination
            signals = {
                "market": sector_sig,
                "quant": quant_sig,
                "rl": rl_sig,
                "sentiment": sent_sig,
                "momentum": mom_sig,
                "regime": regime_sig,
            }

            # Use online-learned weights if available, else static
            weights = adaptive_weights if adaptive_weights else AGENT_WEIGHTS
            final_score = sum(
                weights.get(k, 0.0) * signals[k] for k in signals
            )
            final_score = max(-1.0, min(1.0, final_score))

            # Determine action
            action = self._classify_action(final_score)
            if action == "HOLD":
                continue

            # Cost filter: skip low-conviction buys where expected gain
            # barely covers round-trip transaction costs
            tx_cost = TRANSACTION_COSTS.get("default", 0.0015)
            round_trip_cost = tx_cost * 2  # buy + sell
            # Use P(outperform) and average swing return to estimate expected gain
            expected_gain = (rank.p_outperform - 0.5) * 0.08  # excess prob × avg swing
            if "BUY" in action and expected_gain < round_trip_cost * 3:
                logger.debug("Cost filter: %s skipped (expected %.3f%% < %.3f%% threshold)",
                             tic, expected_gain * 100, round_trip_cost * 3 * 100)
                continue

            # Position sizing
            exposure_scale = market_view.exposure_scale if market_view else 1.0
            size_pct = self._compute_position_size(
                final_score, rank.p_outperform, exposure_scale
            )
            size_usd = size_pct * portfolio_value

            # Build reasons
            reasons = self._build_reasons(signals, rank, market_view, sentiment_map.get(tic, 0.0))

            decision = Decision(
                ticker=tic,
                action=action,
                confidence=final_score,
                size_pct=size_pct,
                size_usd=size_usd,
                price=rank.price,
                reasons=reasons,
                signals=signals,
            )
            decisions.append(decision)

        # ---------------------------------------------------------------
        # EXIT management: evaluate held positions not in candidates
        # ---------------------------------------------------------------

        # Update scan miss counts
        if current_positions:
            for tic in current_positions:
                if tic in candidate_ticker_set:
                    self._scan_miss_counts.pop(tic, None)
                else:
                    self._scan_miss_counts[tic] = self._scan_miss_counts.get(tic, 0) + 1
            # Clean up tickers no longer held
            held = set(current_positions.keys())
            for tic in list(self._scan_miss_counts.keys()):
                if tic not in held:
                    del self._scan_miss_counts[tic]

        if current_positions:
            orphan_tickers = set(current_positions.keys()) - candidate_ticker_set
            # Also check held tickers that ARE in candidates but got SELL/HOLD
            decided_tickers = {d.ticker for d in decisions}

            for tic in orphan_tickers:
                if tic in decided_tickers:
                    continue
                pos_info = current_positions[tic]
                exit_decision = self._evaluate_exit(tic, pos_info)
                if exit_decision is not None:
                    decisions.append(exit_decision)

        # Sort: SELLs first (to free capital), then BUYs by confidence descending
        decisions.sort(key=lambda d: (0 if "SELL" in d.action else 1, -d.confidence))

        # Sector concentration limit: max 30% per sector
        MAX_SECTOR_PCT = 0.30
        from agents.quant_agent import TICKER_SECTOR
        sector_alloc: Dict[str, float] = {}
        sector_filtered: List[Decision] = []
        for d in decisions:
            if "SELL" in d.action:
                sector_filtered.append(d)
                continue
            sec = TICKER_SECTOR.get(d.ticker, "unknown")
            current = sector_alloc.get(sec, 0.0)
            if current + d.size_pct > MAX_SECTOR_PCT:
                remaining = MAX_SECTOR_PCT - current
                if remaining > 0.02:
                    d.size_pct = remaining
                    d.size_usd = remaining * portfolio_value
                else:
                    continue
            sector_alloc[sec] = sector_alloc.get(sec, 0.0) + d.size_pct
            sector_filtered.append(d)
        decisions = sector_filtered

        # Cap total exposure
        total_pct = 0.0
        filtered: List[Decision] = []
        for d in decisions:
            if total_pct + d.size_pct > MAX_EXPOSURE:
                remaining = MAX_EXPOSURE - total_pct
                if remaining > 0.02:
                    d.size_pct = remaining
                    d.size_usd = remaining * portfolio_value
                else:
                    continue
            total_pct += d.size_pct
            filtered.append(d)

        self._last_decisions = filtered

        # Publish decisions to EventBus
        for d in filtered:
            await self._bus.publish("decision.signal", d.to_dict())

        await self._bus.publish("agent.status", {
            "stage": "synthesizer",
            "total_decisions": len(filtered),
            "total_exposure": round(total_pct, 4),
            "actions": {d.ticker: d.action for d in filtered},
        })

        logger.info(
            "Synthesizer: %d decisions, exposure=%.1f%%, top=%s",
            len(filtered),
            total_pct * 100,
            [(d.ticker, d.action, f"{d.confidence:.2f}") for d in filtered[:5]],
        )

        return filtered

    @property
    def last_decisions(self) -> List[Decision]:
        return self._last_decisions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_action(self, score: float) -> str:
        if score >= THRESHOLDS["strong_buy"]:
            return "STRONG_BUY"
        elif score >= THRESHOLDS["buy"]:
            return "BUY"
        elif score <= THRESHOLDS["strong_sell"]:
            return "STRONG_SELL"
        elif score <= THRESHOLDS["sell"]:
            return "SELL"
        return "HOLD"

    def _compute_position_size(
        self, confidence: float, p_outperform: float, exposure_scale: float
    ) -> float:
        """Backtest-validated position sizing.

        Uses the same formula as backtest_pipeline.py:
            abs(score) * 0.15, capped at MAX_POSITION_PCT.
        Previous Kelly formula was ~4x too conservative vs backtest
        (1.8% vs 7.5% at conf=0.5).
        """
        sized = min(abs(confidence) * 0.15, MAX_POSITION_PCT) * exposure_scale

        # Boost for high P(outperform) — validated factor
        if p_outperform > 0.60:
            sized *= 1.0 + (p_outperform - 0.60) * 2.0  # 0.80 → 1.4x, 0.90 → 1.6x

        # Floor: minimum 1% allocation for any BUY decision
        if sized < 0.01:
            sized = 0.01

        return min(sized, MAX_POSITION_PCT)

    def _build_sector_signals(self, market_view: Any) -> Dict[str, float]:
        """Map tickers to their sector's relative strength signal."""
        if market_view is None:
            return {}

        from agents.quant_agent import SECTOR_MAP

        # Build sector score lookup
        sector_scores = {}
        for sec_info in (market_view.sector_scores or []):
            name = sec_info.get("sector", "")
            score = sec_info.get("composite_score", sec_info.get("score", 0.0))
            sector_scores[name] = score

        # Map each ticker to its sector signal
        ticker_signals = {}
        for sector_name, info in SECTOR_MAP.items():
            score = sector_scores.get(sector_name, 0.0)
            # Normalize to [-1, 1] roughly (typical scores are -0.1 to +0.1)
            normalized = max(-1.0, min(1.0, score * 5))
            for tic in info["stocks"]:
                ticker_signals[tic] = normalized

        return ticker_signals

    def _regime_signal(self, market_view: Any, ticker: str) -> float:
        """Regime-based bias for a ticker."""
        if market_view is None:
            return 0.0

        from agents.quant_agent import SECTOR_MAP

        # Find ticker's sector
        sector_name = None
        for name, info in SECTOR_MAP.items():
            if ticker in info["stocks"]:
                sector_name = name
                break

        if sector_name is None:
            return 0.0

        # Growth vs defensive
        defensive_sectors = {"Healthcare", "Staples", "Utilities", "RealEstate"}
        growth_sectors = {"Technology", "Semis", "Communication", "ConsDisc", "Financials"}
        speculative_sectors = {"Biotech"}

        regime = market_view.regime if market_view else "low_volatility"
        if regime == "high_volatility":
            if sector_name in defensive_sectors:
                return 0.3
            elif sector_name in speculative_sectors:
                return -0.5
            elif sector_name in growth_sectors:
                return -0.2
            return 0.0
        else:
            # Low vol: growth favoured
            if sector_name in growth_sectors:
                return 0.1
            elif sector_name in defensive_sectors:
                return -0.1
            return 0.0

    def _score_sentiment_batch(self, tickers: List[str]) -> Dict[str, float]:
        """Score FinBERT sentiment for a list of tickers.

        Uses SectorScanner._get_news_sentiment() when available (it fetches
        yfinance news headlines internally), otherwise scores a generic query.
        """
        if self._finbert is None:
            return {}

        sentiment_map = {}
        for tic in tickers:
            try:
                # Reuse SectorScanner's news sentiment method (fetches yfinance headlines + FinBERT)
                if self._scanner and hasattr(self._scanner, '_get_news_sentiment'):
                    sentiment_map[tic] = self._scanner._get_news_sentiment(tic)
                else:
                    # Fallback: score a generic headline
                    score = self._finbert.score_text(f"{tic} stock market")
                    sentiment_map[tic] = score
            except Exception as e:
                logger.debug("Sentiment scoring failed for %s: %s", tic, e)
                sentiment_map[tic] = 0.0

        return sentiment_map

    def _get_rl_signals(self, candidates: List[str]) -> Dict[str, float]:
        """Get RL ensemble (PPO+A2C+SAC) action signals for candidates.

        Only returns signals for tickers the RL model was trained on.
        The model's num_tickers may differ from config (e.g. trained on 5 of 10),
        so we use the model's actual dimensions to construct the observation.

        Returns dict of {ticker: action_value} where action_value is in [-1, 1].
        """
        if self._ensemble is None or not self._ensemble.is_trained:
            return {}
        if self._data_proc is None:
            return {}

        try:
            import numpy as np
            from datetime import datetime, timedelta
            from ensemble_agent import TradingEnv
            from config import TRAIN_CONFIG

            # Use model's actual dimensions, not config
            model_n_tickers = self._ensemble.num_tickers
            model_n_features = self._ensemble.num_features

            rl_tickers = TRAIN_CONFIG.get("tickers", [])[:model_n_tickers]
            if not rl_tickers:
                return {}

            # Fetch latest data for RL tickers
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

            raw_df = self._data_proc.fetch_data(rl_tickers, start, end, save_raw=False)
            enriched_df = self._data_proc.add_technical_indicators(raw_df)
            feature_array, _ = self._data_proc.create_feature_array(enriched_df)

            if feature_array.shape[0] == 0:
                return {}

            # Trim/pad features to match model's expected dimensions
            actual_tickers = feature_array.shape[1]
            actual_features = feature_array.shape[2]

            if actual_tickers != model_n_tickers or actual_features != model_n_features:
                # Pad or trim to match model dimensions
                adjusted = np.zeros(
                    (feature_array.shape[0], model_n_tickers, model_n_features),
                    dtype=np.float32,
                )
                t_dim = min(actual_tickers, model_n_tickers)
                f_dim = min(actual_features, model_n_features)
                adjusted[:, :t_dim, :f_dim] = feature_array[:, :t_dim, :f_dim]
                feature_array = adjusted

            # Build observation by fast-forwarding env to last step
            env = TradingEnv(feature_array)
            obs, _ = env.reset()
            for _ in range(feature_array.shape[0] - 1):
                zero_action = np.zeros(env.num_tickers, dtype=np.float32)
                obs, _, done, _, _ = env.step(zero_action)
                if done:
                    break

            # Get RL prediction
            raw_actions = self._ensemble.predict(obs)

            # Map actions to tickers
            rl_map = {}
            for i, tic in enumerate(rl_tickers):
                if i < len(raw_actions) and tic in candidates:
                    rl_map[tic] = float(raw_actions[i])

            logger.info("RL signals: %s", {k: f"{v:.3f}" for k, v in rl_map.items()})
            return rl_map

        except Exception as e:
            logger.warning("RL signal generation failed: %s", e)
            return {}

    def _build_reasons(
        self, signals: Dict[str, float], rank: Any,
        market_view: Any, sentiment: float,
    ) -> List[str]:
        """Build human-readable reasons for the decision."""
        reasons = []

        # Quant signal
        if rank.p_outperform > 0.6:
            reasons.append(f"strong quant signal (P(outperform)={rank.p_outperform:.0%})")
        elif rank.p_outperform > 0.5:
            reasons.append(f"positive quant signal (P(outperform)={rank.p_outperform:.0%})")

        # RL ensemble
        rl_val = signals.get("rl", 0.0)
        if rl_val > 0.3:
            reasons.append(f"RL ensemble bullish ({rl_val:+.2f})")
        elif rl_val < -0.3:
            reasons.append(f"RL ensemble bearish ({rl_val:+.2f})")

        # Market/sector
        if signals["market"] > 0.2:
            reasons.append("strong sector momentum")
        elif signals["market"] > 0:
            reasons.append("positive sector momentum")

        # Momentum (v8: z-scored momentum_21d)
        mom_z = rank.rank_features.get("momentum_21d", 0.0)
        if mom_z > 1.0:
            reasons.append(f"strong momentum (z={mom_z:+.1f})")
        elif mom_z < -1.0:
            reasons.append(f"weak momentum (z={mom_z:+.1f})")

        # Sentiment
        if sentiment > 0.3:
            reasons.append(f"positive sentiment ({sentiment:+.2f})")
        elif sentiment < -0.3:
            reasons.append(f"negative sentiment ({sentiment:+.2f})")

        # Regime
        if market_view:
            reasons.append(f"regime: {market_view.regime}")

        return reasons

    def _evaluate_exit(
        self, ticker: str, pos_info: Dict[str, Any],
    ) -> Optional[Decision]:
        """Evaluate whether a held position should be exited.

        Swing trading EXIT rules (from SWING_EXIT_CONFIG):
            - Loss > stop_loss_pct → SELL (stop loss)
            - Profit > take_profit_pct → SELL (take profit)
            - Consecutive scan misses > limit → SELL (dropped from universe)
        """
        entry_price = pos_info.get("entry_price", 0.0)
        current_price = pos_info.get("current_price", 0.0)
        if entry_price <= 0 or current_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price
        miss_count = self._scan_miss_counts.get(ticker, 0)
        reasons: List[str] = []
        should_sell = False

        # Stop loss
        if pnl_pct <= SWING_EXIT_CONFIG["stop_loss_pct"]:
            reasons.append(f"stop loss triggered ({pnl_pct:+.1%})")
            should_sell = True

        # Take profit
        elif pnl_pct >= SWING_EXIT_CONFIG["take_profit_pct"]:
            reasons.append(f"take profit triggered ({pnl_pct:+.1%})")
            should_sell = True

        # Consecutive scan miss
        elif miss_count >= SWING_EXIT_CONFIG["consecutive_miss_limit"]:
            reasons.append(f"dropped from scan {miss_count}x consecutively")
            should_sell = True

        if not should_sell:
            return None

        logger.info("EXIT %s: %s (pnl=%.1f%%, misses=%d)",
                     ticker, reasons, pnl_pct * 100, miss_count)

        market_value = pos_info.get("market_value", current_price)
        return Decision(
            ticker=ticker,
            action="SELL",
            confidence=-0.5,
            size_pct=0.0,  # full exit — broker sells entire position
            size_usd=market_value,
            price=current_price,
            reasons=reasons,
            signals={"exit_pnl": pnl_pct, "scan_misses": float(miss_count)},
        )


# ------------------------------------------------------------------
# Helper: Convert Synthesizer decisions to auto_trader compatible format
# ------------------------------------------------------------------

def decisions_to_legacy(
    decisions: List[Decision],
    market_view: Any,
    portfolio_value: float,
) -> Dict[str, Any]:
    """Convert Synthesizer output to the format expected by /decide endpoint.

    This ensures backwards compatibility with the existing REST API.
    """
    legacy_decisions = []
    for d in decisions:
        legacy_decisions.append({
            "ticker": d.ticker,
            "action": d.action,
            "confidence": d.confidence,
            "position_size_pct": d.size_pct,
            "position_size_usd": d.size_usd,
            "price": d.price,
            "reasons": d.reasons,
            "signals": d.signals,
        })

    total_invested = sum(d.size_pct for d in decisions)
    positions = {d.ticker: d.size_pct for d in decisions}

    summary_parts = []
    if market_view:
        summary_parts.append(
            f"Regime: {market_view.regime} ({market_view.regime_confidence:.0%})"
        )
        top_names = [s.get("sector", "?") for s in (market_view.top_sectors or [])[:3]]
        summary_parts.append(f"Top: {', '.join(top_names)}")

    buy_count = sum(1 for d in decisions if "BUY" in d.action)
    sell_count = sum(1 for d in decisions if "SELL" in d.action)
    summary_parts.append(
        f"BUY {buy_count} / SELL {sell_count} ({total_invested:.0%} alloc)"
    )

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "regime": {
            "state": market_view.regime if market_view else "unknown",
            "confidence": market_view.regime_confidence if market_view else 0,
            "volatility": market_view.volatility if market_view else 0,
            "exposure_scale": market_view.exposure_scale if market_view else 1.0,
        },
        "decisions": legacy_decisions,
        "summary": " | ".join(summary_parts),
        "portfolio_allocation": {
            "positions": positions,
            "total_invested_pct": total_invested,
            "cash_pct": 1.0 - total_invested,
            "num_positions": len(decisions),
        },
        "pipeline": "multi_agent_v3",
    }
