"""Scalping signal engine — 5 weighted signals for short-term entries.

Each signal returns a score in [-1, +1]:
    +1 = strong long signal
    -1 = strong short signal
     0 = neutral

Signals are combined via weighted ensemble. The engine reuses TA helpers
from MultiTFAggregator (RSI, EMA calculations).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("trading-engine.scalp_signal")


@dataclass
class SignalResult:
    """Result from a single signal evaluator."""
    name: str
    score: float          # [-1, +1]
    confidence: float     # [0, 1]
    reason: str = ""


@dataclass
class EnsembleResult:
    """Combined result from all signals."""
    ticker: str
    score: float          # weighted [-1, +1]
    direction: str        # "long", "short", "neutral"
    action: str           # "instant", "claude_confirm", "ignore"
    signals: List[SignalResult] = field(default_factory=list)
    agreeing_count: int = 0
    regime_scale: float = 1.0

    def summary(self) -> str:
        parts = [f"{s.name}={s.score:+.2f}" for s in self.signals if abs(s.score) > 0.1]
        return f"{self.ticker} ensemble={self.score:+.3f} ({self.direction}/{self.action}) [{', '.join(parts)}]"


class ScalpSignalEngine:
    """5-signal ensemble for scalping entries."""

    def __init__(self, config: dict, weights: dict, execution: dict) -> None:
        self._cfg = config
        self._weights = weights
        self._exec = execution

    def evaluate(
        self,
        ticker: str,
        bars_1m: List[tuple],
        bars_3m: List[tuple],
        bars_5m: List[tuple],
        derivatives_ctx: Dict[str, Any],
        regime_label: str = "low_volatility",
        regime_scale: float = 1.0,
    ) -> EnsembleResult:
        """Run all 5 signals and produce ensemble decision.

        Args:
            bars_Xm: list of (ts_ms, o, h, l, c, v) tuples
            derivatives_ctx: from DerivativesMonitor.get_context()
            regime_label: current crypto regime
            regime_scale: position size multiplier (0.5 for medium_vol)
        """
        signals: List[SignalResult] = []

        # 1. RSI Divergence (1m primary, 5m confirmation)
        signals.append(self._rsi_divergence(bars_1m, bars_5m))

        # 2. EMA Cross (3m)
        signals.append(self._ema_cross(bars_3m))

        # 3. BB Squeeze Breakout (1m)
        signals.append(self._bb_squeeze(bars_1m))

        # 4. Order Flow / CVD (from derivatives)
        signals.append(self._order_flow(ticker, derivatives_ctx))

        # 5. Funding Rate Bias
        signals.append(self._funding_bias(ticker, derivatives_ctx))

        # Weighted ensemble
        weighted_score = 0.0
        for sig in signals:
            w = self._weights.get(sig.name, 0.0)
            weighted_score += sig.score * w

        # Count agreeing signals
        direction = "long" if weighted_score > 0 else "short" if weighted_score < 0 else "neutral"
        agreeing = sum(
            1 for s in signals
            if (direction == "long" and s.score > 0.1)
            or (direction == "short" and s.score < -0.1)
        )

        # Determine action
        abs_score = abs(weighted_score)
        if abs_score >= self._exec.get("instant_threshold", 0.85):
            action = "instant"
        elif (abs_score >= self._exec.get("claude_threshold", 0.70)
              and agreeing >= self._exec.get("min_agreeing_signals", 2)):
            action = "claude_confirm"
        else:
            action = "ignore"

        result = EnsembleResult(
            ticker=ticker,
            score=round(weighted_score, 4),
            direction=direction,
            action=action,
            signals=signals,
            agreeing_count=agreeing,
            regime_scale=regime_scale,
        )

        if action != "ignore":
            logger.info("[signal] %s", result.summary())

        return result

    # ------------------------------------------------------------------
    # Signal 1: RSI Divergence (weight=0.25)
    # ------------------------------------------------------------------

    def _rsi_divergence(
        self,
        bars_1m: List[tuple],
        bars_5m: List[tuple],
    ) -> SignalResult:
        """Detect price-RSI divergence on 1m, confirm on 5m."""
        if len(bars_1m) < 20:
            return SignalResult("rsi_divergence", 0.0, 0.0, "insufficient data")

        closes_1m = [b[4] for b in bars_1m]
        rsi_1m = _calc_rsi(closes_1m, self._cfg.get("rsi_period", 14))

        lookback = self._cfg.get("rsi_divergence_lookback", 10)
        recent_closes = closes_1m[-lookback:]
        recent_rsis = _calc_rsi_series(closes_1m, self._cfg.get("rsi_period", 14), lookback)

        if len(recent_rsis) < 3:
            return SignalResult("rsi_divergence", 0.0, 0.0, "insufficient RSI data")

        score = 0.0
        reason = ""

        # Bullish divergence: price making lower lows, RSI making higher lows
        price_lower = recent_closes[-1] < min(recent_closes[:-1])
        rsi_higher = recent_rsis[-1] > min(recent_rsis[:-1])
        oversold = rsi_1m < self._cfg.get("rsi_oversold", 30)

        if price_lower and rsi_higher and oversold:
            score = 0.8
            reason = f"bullish divergence (RSI={rsi_1m:.0f})"

        # Bearish divergence: price making higher highs, RSI making lower highs
        price_higher = recent_closes[-1] > max(recent_closes[:-1])
        rsi_lower = recent_rsis[-1] < max(recent_rsis[:-1])
        overbought = rsi_1m > self._cfg.get("rsi_overbought", 70)

        if price_higher and rsi_lower and overbought:
            score = -0.8
            reason = f"bearish divergence (RSI={rsi_1m:.0f})"

        # Simple oversold/overbought (weaker signal)
        if score == 0:
            if rsi_1m < self._cfg.get("rsi_oversold", 30):
                score = 0.4
                reason = f"oversold (RSI={rsi_1m:.0f})"
            elif rsi_1m > self._cfg.get("rsi_overbought", 70):
                score = -0.4
                reason = f"overbought (RSI={rsi_1m:.0f})"

        # 5m confirmation: strengthen if 5m agrees
        if abs(score) > 0 and len(bars_5m) >= 20:
            closes_5m = [b[4] for b in bars_5m]
            rsi_5m = _calc_rsi(closes_5m, self._cfg.get("rsi_period", 14))
            if (score > 0 and rsi_5m < 40) or (score < 0 and rsi_5m > 60):
                score *= 1.2  # strengthen
                reason += f", 5m confirms (RSI={rsi_5m:.0f})"

        return SignalResult(
            "rsi_divergence",
            max(-1.0, min(1.0, score)),
            min(1.0, abs(score)),
            reason or "neutral",
        )

    # ------------------------------------------------------------------
    # Signal 2: EMA Cross (weight=0.20)
    # ------------------------------------------------------------------

    def _ema_cross(self, bars_3m: List[tuple]) -> SignalResult:
        """EMA(9/21) cross on 3m with volume confirmation."""
        fast_p = self._cfg.get("ema_fast", 9)
        slow_p = self._cfg.get("ema_slow", 21)
        min_bars = slow_p + 3

        if len(bars_3m) < min_bars:
            return SignalResult("ema_cross", 0.0, 0.0, "insufficient data")

        closes = [b[4] for b in bars_3m]
        volumes = [b[5] for b in bars_3m]

        ema_fast = _calc_ema(closes, fast_p)
        ema_slow = _calc_ema(closes, slow_p)

        # Previous EMAs (for cross detection)
        ema_fast_prev = _calc_ema(closes[:-1], fast_p)
        ema_slow_prev = _calc_ema(closes[:-1], slow_p)

        score = 0.0
        reason = ""

        # Bullish cross: fast crosses above slow
        if ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev:
            score = 0.7
            reason = f"bullish EMA cross (EMA{fast_p}={ema_fast:.2f} > EMA{slow_p}={ema_slow:.2f})"
        # Bearish cross: fast crosses below slow
        elif ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev:
            score = -0.7
            reason = f"bearish EMA cross (EMA{fast_p}={ema_fast:.2f} < EMA{slow_p}={ema_slow:.2f})"
        # Continuation: already crossed, trending
        elif ema_fast > ema_slow:
            gap_pct = (ema_fast - ema_slow) / ema_slow
            if gap_pct > 0.001:
                score = 0.3
                reason = f"bullish trend (gap={gap_pct:.3%})"
        elif ema_fast < ema_slow:
            gap_pct = (ema_slow - ema_fast) / ema_slow
            if gap_pct > 0.001:
                score = -0.3
                reason = f"bearish trend (gap={gap_pct:.3%})"

        # Volume confirmation
        if abs(score) >= 0.5 and len(volumes) >= 10:
            recent_vol = sum(volumes[-3:]) / 3
            avg_vol = sum(volumes[-10:]) / 10
            vol_ratio = self._cfg.get("ema_volume_confirm_ratio", 1.3)
            if avg_vol > 0 and recent_vol / avg_vol >= vol_ratio:
                score *= 1.3
                reason += f", vol confirmed ({recent_vol / avg_vol:.1f}x)"
            elif abs(score) >= 0.5:
                score *= 0.7
                reason += ", weak volume"

        return SignalResult(
            "ema_cross",
            max(-1.0, min(1.0, score)),
            min(1.0, abs(score)),
            reason or "neutral",
        )

    # ------------------------------------------------------------------
    # Signal 3: Bollinger Band Squeeze Breakout (weight=0.20)
    # ------------------------------------------------------------------

    def _bb_squeeze(self, bars_1m: List[tuple]) -> SignalResult:
        """BB squeeze detection on 1m with volume-confirmed breakout."""
        period = self._cfg.get("bb_period", 20)
        std_mult = self._cfg.get("bb_std", 2.0)

        if len(bars_1m) < period + 5:
            return SignalResult("bb_squeeze", 0.0, 0.0, "insufficient data")

        closes = [b[4] for b in bars_1m]
        volumes = [b[5] for b in bars_1m]

        # Current BB
        sma = sum(closes[-period:]) / period
        variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
        std = math.sqrt(variance) if variance > 0 else 0
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        bandwidth = (upper - lower) / sma if sma > 0 else 0

        # Previous bandwidth (to detect squeeze -> expansion)
        prev_closes = closes[-(period + 1):-1]
        prev_sma = sum(prev_closes[-period:]) / period
        prev_var = sum((c - prev_sma) ** 2 for c in prev_closes[-period:]) / period
        prev_std = math.sqrt(prev_var) if prev_var > 0 else 0
        prev_upper = prev_sma + std_mult * prev_std
        prev_lower = prev_sma - std_mult * prev_std
        prev_bw = (prev_upper - prev_lower) / prev_sma if prev_sma > 0 else 0

        squeeze_threshold = self._cfg.get("bb_squeeze_threshold", 0.001)

        score = 0.0
        reason = ""
        current_price = closes[-1]

        # Squeeze detected: bandwidth was low, now expanding
        was_squeezed = prev_bw < squeeze_threshold
        is_expanding = bandwidth > prev_bw * 1.2

        if was_squeezed and is_expanding:
            # Breakout direction
            if current_price > upper:
                score = 0.8
                reason = f"BB squeeze breakout UP (bw={bandwidth:.4f})"
            elif current_price < lower:
                score = -0.8
                reason = f"BB squeeze breakout DOWN (bw={bandwidth:.4f})"
            else:
                score = 0.3 if current_price > sma else -0.3
                reason = f"BB squeezing (bw={bandwidth:.4f})"

        # Simple band touch signals (weaker)
        elif score == 0:
            if current_price <= lower:
                score = 0.3
                reason = f"touch lower BB ({lower:.2f})"
            elif current_price >= upper:
                score = -0.3
                reason = f"touch upper BB ({upper:.2f})"

        # Volume spike confirmation for breakouts
        if abs(score) >= 0.5 and len(volumes) >= 10:
            recent_vol = volumes[-1]
            avg_vol = sum(volumes[-10:]) / 10
            spike_ratio = self._cfg.get("bb_volume_spike_ratio", 2.0)
            if avg_vol > 0 and recent_vol / avg_vol >= spike_ratio:
                score *= 1.2
                reason += f", vol spike ({recent_vol / avg_vol:.1f}x)"

        return SignalResult(
            "bb_squeeze",
            max(-1.0, min(1.0, score)),
            min(1.0, abs(score)),
            reason or "neutral",
        )

    # ------------------------------------------------------------------
    # Signal 4: Order Flow / CVD (weight=0.20)
    # ------------------------------------------------------------------

    def _order_flow(
        self,
        ticker: str,
        derivatives_ctx: Dict[str, Any],
    ) -> SignalResult:
        """Taker delta / CVD from DerivativesMonitor."""
        taker = derivatives_ctx.get("taker_delta", {}).get(ticker, {})
        if not taker:
            return SignalResult("order_flow", 0.0, 0.0, "no taker data")

        ratio = 0.0
        if isinstance(taker, dict):
            ratio = taker.get("buy_sell_ratio", 1.0)
        elif isinstance(taker, (int, float)):
            ratio = float(taker)

        if ratio <= 0:
            return SignalResult("order_flow", 0.0, 0.0, "invalid ratio")

        extreme = self._cfg.get("cvd_extreme_ratio", 1.15)
        score = 0.0
        reason = ""

        if ratio > extreme:
            # Buyers dominating -> bullish
            score = min(1.0, (ratio - 1.0) / (extreme - 1.0) * 0.8)
            reason = f"buyer dominance (ratio={ratio:.3f})"
        elif ratio < 1.0 / extreme:
            # Sellers dominating -> bearish
            score = max(-1.0, -(1.0 / ratio - 1.0) / (extreme - 1.0) * 0.8)
            reason = f"seller dominance (ratio={ratio:.3f})"
        else:
            reason = f"balanced flow (ratio={ratio:.3f})"

        return SignalResult(
            "order_flow",
            max(-1.0, min(1.0, score)),
            min(1.0, abs(score)),
            reason,
        )

    # ------------------------------------------------------------------
    # Signal 5: Funding Rate Bias (weight=0.15)
    # ------------------------------------------------------------------

    def _funding_bias(
        self,
        ticker: str,
        derivatives_ctx: Dict[str, Any],
    ) -> SignalResult:
        """Negative funding -> long bias, positive -> short bias."""
        funding_rates = derivatives_ctx.get("funding_rates", {})
        rate = funding_rates.get(ticker, 0)

        if not rate or rate == 0:
            return SignalResult("funding_bias", 0.0, 0.0, "no funding data")

        rate = float(rate)
        long_thresh = self._cfg.get("funding_long_threshold", -0.0003)
        short_thresh = self._cfg.get("funding_short_threshold", 0.0003)

        score = 0.0
        reason = ""

        if rate < long_thresh:
            # Negative funding = shorts paying longs = long bias
            score = min(1.0, abs(rate) / abs(long_thresh) * 0.6)
            reason = f"negative funding ({rate:.4%}) -> long bias"
        elif rate > short_thresh:
            # Positive funding = longs paying shorts = short bias
            score = max(-1.0, -(rate / short_thresh) * 0.6)
            reason = f"positive funding ({rate:.4%}) -> short bias"
        else:
            reason = f"neutral funding ({rate:.4%})"

        return SignalResult(
            "funding_bias",
            max(-1.0, min(1.0, score)),
            min(1.0, abs(score)),
            reason,
        )


# ======================================================================
# TA Helpers (pure Python, no external deps)
# ======================================================================

def _calc_rsi(closes: List[float], period: int = 14) -> float:
    """Wilder's RSI — returns current value."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_rsi_series(closes: List[float], period: int, n: int) -> List[float]:
    """Calculate RSI for the last n bars."""
    if len(closes) < period + 1 + n:
        # Calculate as many as we can
        n = max(0, len(closes) - period - 1)
    results = []
    for i in range(n):
        end = len(closes) - (n - 1 - i)
        if end < period + 1:
            continue
        results.append(_calc_rsi(closes[:end], period))
    return results


def _calc_ema(values: List[float], period: int) -> float:
    """EMA — returns last value."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def calc_atr(bars: List[tuple], period: int = 14) -> float:
    """Average True Range from OHLCV tuples: (ts, o, h, l, c, v)."""
    if len(bars) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i][2], bars[i][3], bars[i - 1][4]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0

    # Wilder's smoothed ATR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr
