"""Compute 32 H-TS signals from historical OHLCV + derivatives data.

Imports _calc_* functions from core/agent_tools.py directly — no duplication.
Normalization follows runner.py:_extract_ta_signals() logic exactly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

# Import TA calculation functions from core
import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from core.agent_tools import (
    _calc_rsi,
    _calc_ema,
    _calc_ema_series,
    _calc_stoch_rsi,
    _calc_macd,
    _calc_bollinger,
    _calc_atr,
    _calc_vwap,
    _calc_supertrend,
    _calc_mfi,
    _calc_obv_direction,
)

# Signals that require external APIs and cannot be computed historically
UNAVAILABLE_SIGNALS = {
    "liquidation_level",
    "news_sentiment",
    "fear_greed",
    "social_buzz",
    "whale_activity",
    "exchange_flow",
    "dxy_direction",
    "etf_flow",
    "stablecoin_flow",
}


def _bars_from_df(df: pd.DataFrame, end_idx: int, lookback: int) -> List[tuple]:
    """Convert DataFrame slice to bars format: [(ts, open, high, low, close, volume), ...]"""
    start = max(0, end_idx - lookback + 1)
    rows = df.iloc[start:end_idx + 1]
    return [
        (int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else 0,
         r["open"], r["high"], r["low"], r["close"], r["volume"])
        for ts, r in rows.iterrows()
    ]


def compute_ta_at_bar(
    df: pd.DataFrame,
    bar_index: int,
    lookback: int = 200,
) -> Dict[str, float]:
    """Compute all TA-based signals at a specific bar index.

    Uses runner.py _extract_ta_signals() normalization rules exactly.
    Returns dict of {signal_name: normalized_value} in [-1, 1].
    """
    signals: Dict[str, float] = {}

    if bar_index < 50:  # Need minimum history
        return signals

    start = max(0, bar_index - lookback + 1)
    closes = df["close"].iloc[start:bar_index + 1].tolist()
    volumes = df["volume"].iloc[start:bar_index + 1].tolist()
    bars = _bars_from_df(df, bar_index, lookback)
    price = closes[-1]

    if price <= 0 or len(closes) < 30:
        return signals

    # --- RSI → rsi_signal ---
    rsi = _calc_rsi(closes, 14)
    if rsi > 55:
        signals["rsi_signal"] = min(1.0, (rsi - 50) / 50)
    elif rsi < 45:
        signals["rsi_signal"] = max(-1.0, (rsi - 50) / 50)

    # --- StochRSI → stoch_rsi ---
    stoch = _calc_stoch_rsi(closes, 14, 14)
    if stoch > 60:
        signals["stoch_rsi"] = min(1.0, (stoch - 50) / 50)
    elif stoch < 40:
        signals["stoch_rsi"] = max(-1.0, (stoch - 50) / 50)

    # --- EMA cross fast (9 vs 21) → ema_cross_fast ---
    ema_9 = _calc_ema(closes, 9)
    ema_21 = _calc_ema(closes, 21)
    if ema_9 > 0 and ema_21 > 0:
        gap_pct = (ema_9 - ema_21) / ema_21
        if gap_pct > 0:
            signals["ema_cross_fast"] = min(1.0, abs(gap_pct) * 100 + 0.3)
        else:
            signals["ema_cross_fast"] = -min(1.0, abs(gap_pct) * 100 + 0.3)

    # --- EMA cross slow (21 vs 50) → ema_cross_slow ---
    ema_50 = _calc_ema(closes, 50) if len(closes) >= 50 else 0
    if ema_21 > 0 and ema_50 > 0:
        signals["ema_cross_slow"] = 0.5 if ema_21 > ema_50 else -0.5

    # --- MACD histogram ---
    macd = _calc_macd(closes, 12, 26, 9)
    macd_h = macd["histogram"]
    if macd_h and price > 0:
        macd_pct = macd_h / price
        signals["macd_histogram"] = max(-1.0, min(1.0, macd_pct * 500))

    # --- Trend strength (simplified: EMA cross direction) ---
    if ema_9 > 0 and ema_21 > 0:
        signals["trend_strength"] = 0.5 if ema_9 > ema_21 else -0.5

    # --- Supertrend ---
    if len(bars) >= 15:
        st_dir = _calc_supertrend(bars, 10, 3.0)
        if st_dir != 0:
            signals["supertrend"] = 0.6 if st_dir > 0 else -0.6

    # --- Bollinger Bands ---
    bb = _calc_bollinger(closes, 20, 2.0)
    pct_b = bb["pct_b"]
    bandwidth = bb["bandwidth"]

    # bb_deviation
    if pct_b > 0.8:
        signals["bb_deviation"] = min(1.0, (pct_b - 0.5) * 2)
    elif pct_b < 0.2:
        signals["bb_deviation"] = max(-1.0, (pct_b - 0.5) * 2)

    # bb_squeeze
    if bandwidth > 0:
        if bandwidth < 0.02:
            signals["bb_squeeze"] = 0.5
        elif bandwidth > 0.06:
            signals["bb_squeeze"] = -0.3

    # --- VWAP deviation ---
    if len(bars) >= 10:
        vwap_data = _calc_vwap(bars)
        dev_pct = vwap_data.get("deviation_pct", 0)
        if dev_pct and abs(dev_pct) > 0.001:
            signals["vwap_deviation"] = max(-1.0, min(1.0, dev_pct * 10))

        # volume_profile (VWAP proximity)
        upper_1sd = vwap_data.get("upper_1sd", 0)
        lower_1sd = vwap_data.get("lower_1sd", 0)
        if upper_1sd and lower_1sd:
            in_value_area = lower_1sd <= price <= upper_1sd
            signals["volume_profile"] = 0.3 if in_value_area else -0.3

    # --- Support/resistance (simplified: distance to recent high/low) ---
    if len(closes) >= 20:
        recent_high = max(df["high"].iloc[max(0, bar_index - 20):bar_index + 1])
        recent_low = min(df["low"].iloc[max(0, bar_index - 20):bar_index + 1])
        dist_high = (price - recent_high) / price if price > 0 else 0
        dist_low = (price - recent_low) / price if price > 0 else 0
        # Use closest level
        dist = dist_high if abs(dist_high) < abs(dist_low) else dist_low
        if abs(dist) > 0.001:
            signals["support_resistance"] = max(-1.0, min(1.0, -dist * 50))

    # --- Volume spike ---
    if len(volumes) >= 20:
        avg_vol = sum(volumes[-20:]) / 20
        if avg_vol > 0:
            vol_ratio = volumes[-1] / avg_vol
            if vol_ratio > 1.5:
                signals["volume_spike"] = min(1.0, (vol_ratio - 1) / 2)
            elif vol_ratio < 0.5:
                signals["volume_spike"] = max(-1.0, (vol_ratio - 1))

    # --- CVD signal (buy/sell volume proxy using close position in bar range) ---
    if len(bars) >= 5:
        recent_bars = bars[-5:]
        buy_vol = 0.0
        sell_vol = 0.0
        for b in recent_bars:
            bar_range = b[2] - b[3]  # high - low
            if bar_range > 0:
                buy_pct = (b[4] - b[3]) / bar_range  # (close - low) / range
                buy_vol += b[5] * buy_pct
                sell_vol += b[5] * (1 - buy_pct)
        if sell_vol > 0:
            bsr = buy_vol / sell_vol
            if abs(bsr - 1.0) > 0.02:
                signals["cvd_signal"] = max(-1.0, min(1.0, (bsr - 1.0) * 5))

    # --- OBV divergence ---
    if len(closes) >= 12:
        obv_dir = _calc_obv_direction(closes, volumes, 10)
        if obv_dir != 0:
            signals["obv_divergence"] = max(-1.0, min(1.0, obv_dir))

    # --- MFI signal ---
    if len(bars) >= 15:
        mfi = _calc_mfi(bars, 14)
        signals["mfi_signal"] = max(-1.0, min(1.0, (mfi - 50) / 50))

    # --- Volatility regime (ATR-based) ---
    if len(bars) >= 15:
        atr = _calc_atr(bars, 14)
        if price > 0 and atr > 0:
            atr_pct = (atr / price) * 100
            if atr_pct > 2.0:
                signals["volatility_regime"] = 0.7
            elif atr_pct < 0.5:
                signals["volatility_regime"] = -0.5

    # --- Market regime (simplified: trend + volatility) ---
    if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
        if ema_9 > ema_21 > ema_50:
            signals["market_regime"] = 0.5
        elif ema_9 < ema_21 < ema_50:
            signals["market_regime"] = -0.5

    return signals


def add_derivatives_signals(
    signals: Dict[str, float],
    timestamp: pd.Timestamp,
    price: float,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ls_ratio_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Add derivatives-based signals to existing TA signals dict.

    Matches runner.py normalization exactly.
    """
    # Funding rate
    if funding_df is not None and not funding_df.empty:
        # Find closest funding rate <= timestamp
        mask = funding_df.index <= timestamp
        if mask.any():
            fr = funding_df.loc[mask].iloc[-1]["fundingRate"]
            if abs(fr) > 0.0001:
                signals["funding_rate"] = max(-1.0, min(1.0, -fr * 2000))

    # OI change
    if oi_df is not None and not oi_df.empty:
        mask = oi_df.index <= timestamp
        if mask.any():
            recent = oi_df.loc[mask]
            if len(recent) >= 2:
                oi_now = recent.iloc[-1]["sumOpenInterestValue"]
                oi_prev = recent.iloc[-2]["sumOpenInterestValue"]
                if oi_prev > 0:
                    oi_chg = (oi_now - oi_prev) / oi_prev
                    if abs(oi_chg) > 0.002:
                        signals["oi_change"] = max(-1.0, min(1.0, oi_chg * 10))

    # Long/short ratio
    if ls_ratio_df is not None and not ls_ratio_df.empty:
        mask = ls_ratio_df.index <= timestamp
        if mask.any():
            ratio = ls_ratio_df.loc[mask].iloc[-1]["longShortRatio"]
            if abs(ratio - 1.0) > 0.05:
                signals["long_short_ratio"] = max(-1.0, min(1.0, -(ratio - 1.0) * 2))

    # Basis spread (not available in historical API — approximate from funding)
    if funding_df is not None and not funding_df.empty:
        mask = funding_df.index <= timestamp
        if mask.any():
            fr = funding_df.loc[mask].iloc[-1]["fundingRate"]
            # Annualized basis ≈ funding × 3 × 365
            basis_approx = fr * 3 * 365 * 100  # as percentage
            if abs(basis_approx) > 0.05:
                signals["basis_spread"] = max(-1.0, min(1.0, basis_approx / 50))

    return signals


def compute_all_signals_timeseries(
    ohlcv_df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ls_ratio_df: Optional[pd.DataFrame] = None,
    lookback: int = 200,
) -> pd.DataFrame:
    """Compute all 32 signals for every bar in the OHLCV DataFrame.

    Returns DataFrame with index=timestamp, columns=signal names.
    Unavailable signals are omitted (they'll remain at Beta(1,1) prior).
    """
    records = []
    n = len(ohlcv_df)

    for i in range(lookback, n):
        ts = ohlcv_df.index[i]
        price = ohlcv_df.iloc[i]["close"]

        # TA signals
        sigs = compute_ta_at_bar(ohlcv_df, i, lookback)

        # Derivatives signals
        add_derivatives_signals(sigs, ts, price, funding_df, oi_df, ls_ratio_df)

        sigs["_timestamp"] = ts
        sigs["_price"] = price
        records.append(sigs)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.set_index("_timestamp")
    return result
