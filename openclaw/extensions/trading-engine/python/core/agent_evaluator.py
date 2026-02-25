"""Agent-based position evaluator — the brain behind hold/exit/add decisions.

This is what makes the system different from hardcoded TP/SL rules.
Each position is re-evaluated with FULL MARKET CONTEXT every 3-5 minutes:

1. Regime change: Did market regime change since entry?
   "Entered in low_vol, now high_vol → reduce exposure"

2. Momentum reversal: Did momentum flip direction?
   "Entered with z=+1.5, now z=-0.8 → momentum died"

3. Cross-asset correlation: Are correlated assets crashing?
   "BTC dropped 5% in 2h, this ETH position has 0.85 correlation → danger"

4. Volume anomaly: Is something unusual happening?
   "Volume 3x above average → other agents/bots are active"

5. Sentiment shift: Did news change?
   "FinBERT sentiment flipped from +0.4 to -0.3"

Each signal votes: HOLD / EXIT / ADD with a confidence.
Weighted combination → final verdict.

This runs inside PositionTracker's agent loop.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading-engine.agent_evaluator")


# Evaluation signal weights (for position re-evaluation)
# Different from entry weights — here we care about CHANGE since entry
EVAL_WEIGHTS = {
    "regime_change": 0.25,     # regime flipped → strongest exit signal
    "momentum_reversal": 0.25, # momentum direction changed
    "correlation_risk": 0.20,  # correlated asset crashing
    "pnl_trajectory": 0.15,    # PnL trend (improving vs deteriorating)
    "time_decay": 0.15,        # longer hold → higher bar for staying
}


def evaluate_position(
    position: Any,  # TrackedPosition
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate whether to HOLD, EXIT, or ADD to a position.

    Args:
        position: TrackedPosition with entry context and current state
        context: Current market context from context_provider:
            - regime: current market regime
            - btc_price: current BTC price (for crypto correlation)
            - btc_change_1h: BTC % change last hour
            - momentum_scores: {ticker: z-score} from latest quant scan
            - volume_ratios: {ticker: current_vol / avg_vol}
            - sentiment_scores: {ticker: finbert_score}

    Returns:
        {"verdict": "HOLD"|"EXIT"|"ADD", "confidence": float, "reasons": list}
    """
    signals: Dict[str, float] = {}
    reasons: List[str] = []

    # 1. Regime change signal
    sig, reason = _eval_regime_change(position, context)
    signals["regime_change"] = sig
    if reason:
        reasons.append(reason)

    # 2. Momentum reversal signal
    sig, reason = _eval_momentum(position, context)
    signals["momentum_reversal"] = sig
    if reason:
        reasons.append(reason)

    # 3. Cross-asset correlation risk
    sig, reason = _eval_correlation_risk(position, context)
    signals["correlation_risk"] = sig
    if reason:
        reasons.append(reason)

    # 4. PnL trajectory
    sig, reason = _eval_pnl_trajectory(position, context)
    signals["pnl_trajectory"] = sig
    if reason:
        reasons.append(reason)

    # 5. Time decay pressure
    sig, reason = _eval_time_decay(position, context)
    signals["time_decay"] = sig
    if reason:
        reasons.append(reason)

    # Weighted combination: positive = HOLD/ADD, negative = EXIT
    weighted_score = sum(
        EVAL_WEIGHTS[k] * signals[k] for k in EVAL_WEIGHTS
    )
    # Clamp to [-1, 1]
    weighted_score = max(-1.0, min(1.0, weighted_score))

    # Determine verdict
    if weighted_score < -0.3:
        verdict = "EXIT"
    elif weighted_score > 0.4:
        verdict = "ADD"
    else:
        verdict = "HOLD"

    confidence = abs(weighted_score)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": reasons,
        "signals": signals,
        "weighted_score": weighted_score,
    }


# ------------------------------------------------------------------
# Signal evaluators
# ------------------------------------------------------------------

def _eval_regime_change(
    position: Any, context: Dict[str, Any]
) -> tuple:
    """Did regime change since entry? This is the strongest exit signal.

    low_vol at entry → high_vol now: -0.8 (strong exit)
    high_vol at entry → low_vol now: +0.3 (opportunity)
    same regime: 0.0 (neutral)
    """
    current_regime = context.get("crypto_regime", context.get("regime", "unknown"))
    entry_regime = getattr(position, "entry_regime", "unknown")

    # entry_regime may be combined format ("low_volatility_goldilocks") —
    # extract crypto part for comparison
    entry_crypto = entry_regime.split("_")[0:2]  # ["low", "volatility"]
    entry_crypto = "_".join(entry_crypto) if len(entry_crypto) >= 2 else entry_regime

    if entry_crypto == "unknown" or current_regime == "unknown":
        return 0.0, None

    if entry_crypto == current_regime:
        return 0.0, None

    # Regime changed
    if entry_crypto == "low_volatility" and current_regime == "high_volatility":
        return -0.8, f"regime flipped: {entry_crypto} -> {current_regime}"

    if entry_crypto == "high_volatility" and current_regime == "low_volatility":
        return 0.3, f"regime improved: {entry_crypto} -> {current_regime}"

    return -0.3, f"regime changed: {entry_crypto} -> {current_regime}"


def _eval_momentum(
    position: Any, context: Dict[str, Any]
) -> tuple:
    """Did momentum reverse since entry?

    Compares entry momentum z-score with current.
    Sign flip = strong reversal signal.
    """
    ticker = position.ticker
    current_scores = context.get("momentum_scores", {})
    current_z = current_scores.get(ticker, 0.0)
    entry_z = getattr(position, "entry_momentum_z", 0.0)

    if entry_z == 0.0:
        return 0.0, None

    # Check for sign flip (momentum reversal)
    if entry_z > 0.5 and current_z < -0.3:
        # Was strongly positive, now negative
        return -0.7, f"momentum reversed: z {entry_z:+.1f} -> {current_z:+.1f}"

    if entry_z > 0.5 and current_z < 0.2:
        # Momentum fading (not reversed, but weakening)
        return -0.3, f"momentum fading: z {entry_z:+.1f} -> {current_z:+.1f}"

    if current_z > entry_z + 0.5:
        # Momentum strengthening
        return 0.4, f"momentum strengthening: z {entry_z:+.1f} -> {current_z:+.1f}"

    return 0.0, None


def _eval_correlation_risk(
    position: Any, context: Dict[str, Any]
) -> tuple:
    """Are correlated assets crashing?

    For crypto: if BTC drops >3% in an hour, everything follows.
    For equities: if SPY drops >2%, risk-off.
    """
    market_type = getattr(position, "market_type", "crypto")

    if market_type == "crypto":
        btc_change = context.get("btc_change_1h", 0.0)
        if btc_change < -0.05:
            return -0.9, f"BTC crashed {btc_change:+.1%} in 1h — correlated risk"
        if btc_change < -0.03:
            return -0.5, f"BTC dropped {btc_change:+.1%} in 1h — moderate risk"
        if btc_change > 0.03:
            return 0.2, f"BTC surging {btc_change:+.1%} in 1h"
    else:
        spy_change = context.get("spy_change_1h", 0.0)
        if spy_change < -0.02:
            return -0.6, f"SPY dropped {spy_change:+.1%} — risk-off"
        if spy_change < -0.01:
            return -0.3, f"SPY weak {spy_change:+.1%}"

    return 0.0, None


def _eval_pnl_trajectory(
    position: Any, context: Dict[str, Any]
) -> tuple:
    """Is PnL improving or deteriorating?

    Not just the current PnL level, but its direction.
    Trailing high vs current → distance from peak.
    """
    pnl = position.pnl_pct
    entry_price = position.entry_price
    current_price = position.current_price
    trailing_high = position.trailing_high

    if entry_price <= 0 or trailing_high <= 0:
        return 0.0, None

    # Distance from trailing high (drawdown from peak)
    if trailing_high > entry_price:
        peak_pnl = (trailing_high - entry_price) / entry_price
        drawdown_from_peak = peak_pnl - pnl
    else:
        drawdown_from_peak = 0.0

    # Heavy drawdown from peak → profit is evaporating
    if drawdown_from_peak > 0.03:
        return -0.6, f"profit evaporating: was +{peak_pnl:.1%}, now +{pnl:.1%} (gave back {drawdown_from_peak:.1%})"

    # Deep loss
    if pnl < -0.03:
        return -0.5, f"deep loss at {pnl:+.1%}"

    # Healthy profit
    if pnl > 0.03:
        return 0.3, f"healthy profit at {pnl:+.1%}"

    return 0.0, None


def _eval_time_decay(
    position: Any, context: Dict[str, Any]
) -> tuple:
    """Longer holds need higher conviction to maintain.

    Swing trading: 2-10 days optimal.
    >10 days with low profit → capital is stuck.
    """
    held_hours = position.held_hours
    pnl = position.pnl_pct
    market_type = getattr(position, "market_type", "crypto")

    # Crypto moves faster: time pressure kicks in earlier
    if market_type == "crypto":
        if held_hours > 168 and pnl < 0.02:  # >7 days, <2% profit
            return -0.5, f"stale crypto position: {held_hours:.0f}h held, only {pnl:+.1%}"
        if held_hours > 72 and pnl < 0.0:  # >3 days, negative
            return -0.4, f"crypto losing for {held_hours:.0f}h: {pnl:+.1%}"
    else:
        if held_hours > 240 and pnl < 0.01:  # >10 days, <1% profit
            return -0.5, f"stale equity position: {held_hours:.0f}h held, only {pnl:+.1%}"
        if held_hours > 120 and pnl < -0.02:  # >5 days, >2% loss
            return -0.4, f"equity losing for {held_hours:.0f}h: {pnl:+.1%}"

    return 0.0, None


# ------------------------------------------------------------------
# Market context builder (used by PositionTracker)
# ------------------------------------------------------------------

def build_market_context(
    regime_detector: Any = None,
    crypto_regime_detector: Any = None,
    binance_broker: Any = None,
    btc_price: float = 0.0,
    btc_change_1h: float = 0.0,
) -> Dict[str, Any]:
    """Build current market context for agent evaluation.

    Called by PositionTracker's context_provider.
    btc_price/btc_change_1h: passed from WS tick cache (avoids REST).
    """
    context: Dict[str, Any] = {}

    # Equity regime
    if regime_detector is not None:
        try:
            regime = regime_detector.detect()
            context["regime"] = regime.get("regime", "unknown")
            context["regime_confidence"] = regime.get("confidence", 0.0)
        except Exception:
            context["regime"] = "unknown"

    # Crypto regime
    if crypto_regime_detector is not None:
        try:
            crypto_regime = crypto_regime_detector.detect()
            context["crypto_regime"] = crypto_regime.get("regime", "unknown")
            context["exposure_scale"] = crypto_regime.get("exposure_scale", 1.0)
        except Exception:
            pass

    # BTC reference price + change (from WS tick cache — zero REST calls)
    if btc_price > 0:
        context["btc_price"] = btc_price
        context["btc_change_1h"] = btc_change_1h

    return context
