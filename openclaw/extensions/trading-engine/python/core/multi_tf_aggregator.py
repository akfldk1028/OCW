"""Multi-timeframe candle aggregator for Claude prompt enrichment.

Collects OHLCV bars across multiple intervals (e.g. 15m/1h/4h) and produces
a compact summary for Claude's trading decisions. No external dependencies
beyond stdlib.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class Bar:
    """Single OHLCV bar."""
    o: float
    h: float
    l: float
    c: float
    v: float
    ts: float = 0.0


class MultiTFAggregator:
    """Aggregate candle data across multiple timeframes."""

    def __init__(self, intervals: Tuple[str, ...] = ("15m", "1h", "4h"), maxlen: int = 20):
        self._intervals = intervals
        self._maxlen = maxlen
        # {ticker: {interval: deque[Bar]}}
        self._candles: Dict[str, Dict[str, deque]] = {}

    def update(self, ticker: str, interval: str, bar: Dict[str, float]) -> None:
        """Add or update a candle bar."""
        if interval not in self._intervals:
            return
        if ticker not in self._candles:
            self._candles[ticker] = {iv: deque(maxlen=self._maxlen) for iv in self._intervals}
        q = self._candles[ticker][interval]
        b = Bar(
            o=bar.get("open", 0), h=bar.get("high", 0),
            l=bar.get("low", 0), c=bar.get("close", bar.get("price", 0)),
            v=bar.get("volume", 0), ts=bar.get("timestamp", 0),
        )
        # Replace last bar if same timestamp (update in-progress candle), else append
        if q and q[-1].ts == b.ts:
            q[-1] = b
        else:
            q.append(b)

    def get_summary(self, ticker: str) -> Dict[str, Any]:
        """Return multi-TF summary dict for a ticker."""
        if ticker not in self._candles:
            return {}
        result = {}
        for interval in self._intervals:
            bars = list(self._candles[ticker].get(interval, []))
            if not bars:
                result[interval] = {"status": "no data"}
                continue
            result[interval] = self._summarize_bars(bars)
        return result

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Return summaries for all tickers."""
        return {ticker: self.get_summary(ticker) for ticker in self._candles}

    def format_for_prompt(self, ticker: str) -> str:
        """Format multi-TF summary as human-readable text for Claude."""
        summary = self.get_summary(ticker)
        if not summary:
            return ""
        lines = []
        for interval in self._intervals:
            s = summary.get(interval, {})
            if s.get("status") == "no data":
                lines.append(f"  {interval}: no data yet")
                continue
            direction = s.get("direction", "?")
            green = s.get("green_ratio", "?")
            rsi = s.get("rsi", 0)
            ema_cross = s.get("ema_cross", "neutral")
            vol_trend = s.get("vol_trend", "flat")
            last_close = s.get("last_close", 0)
            lines.append(
                f"  {interval}: {direction} {green} green, "
                f"RSI={rsi:.0f}, EMA={ema_cross}, vol={vol_trend}, "
                f"last=${last_close:,.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal TA calculations (pure Python, no external deps)
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_bars(bars: List[Bar]) -> Dict[str, Any]:
        """Compute lightweight TA summary from bar list."""
        closes = [b.c for b in bars if b.c > 0]
        volumes = [b.v for b in bars]
        n = len(closes)
        if n == 0:
            return {"status": "no data"}

        # Direction: count green bars in last 5
        recent = bars[-min(5, len(bars)):]
        green_count = sum(1 for b in recent if b.c >= b.o)
        total_recent = len(recent)
        direction = "up" if green_count > total_recent / 2 else "down" if green_count < total_recent / 2 else "flat"

        # RSI(14)
        rsi = MultiTFAggregator._calc_rsi(closes, 14) if n >= 15 else 50.0

        # EMA(9) vs EMA(21) cross
        ema_cross = "neutral"
        if n >= 21:
            ema9 = MultiTFAggregator._calc_ema(closes, 9)
            ema21 = MultiTFAggregator._calc_ema(closes, 21)
            if ema9 > ema21:
                ema_cross = "bullish"
            elif ema9 < ema21:
                ema_cross = "bearish"

        # Volume trend (last 5 vs prev 5)
        vol_trend = "flat"
        if len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            prev_vol = sum(volumes[-10:-5]) / 5
            if prev_vol > 0:
                ratio = recent_vol / prev_vol
                if ratio > 1.3:
                    vol_trend = "increasing"
                elif ratio < 0.7:
                    vol_trend = "decreasing"

        return {
            "direction": direction,
            "green_ratio": f"{green_count}/{total_recent}",
            "rsi": rsi,
            "ema_cross": ema_cross,
            "vol_trend": vol_trend,
            "last_close": closes[-1],
            "bar_count": n,
        }

    @staticmethod
    def _calc_rsi(closes: List[float], period: int = 14) -> float:
        """Wilder's RSI."""
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

    @staticmethod
    def _calc_ema(values: List[float], period: int) -> float:
        """Exponential Moving Average — returns last value."""
        if not values:
            return 0.0
        k = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema
