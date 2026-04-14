"""Simulate the FULL trading loop on historical data.

Replicates the exact learning flow from runner.py:
1. Every bar: compute signals → strategy decides BUY/SHORT/HOLD
2. HOLD → save hold snapshot for counterfactual learning
3. Trade exit → record_trade() with full params (ta_snapshot, confidence, position_pct, mfe/mae)
4. Post-exit → exit regret check (premature vs validated)
5. Periodically → counterfactual check on hold snapshots (3 horizons)
6. Periodically → correct_hold check (avoided loss)

This gives H-TS the same 4-signal learning as the live system.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import ROUND_TRIP_FEE, STRATEGY_WEIGHTS
from .compute_signals import compute_ta_at_bar, add_derivatives_signals

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
    _calc_supertrend,
)
from core.ts.hierarchical import HierarchicalOnlineLearner

logger = logging.getLogger("pretrain.simulate")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """A currently open position."""
    ticker: str
    strategy: str
    side: str  # "long" | "short"
    entry_idx: int
    entry_price: float
    agent_signals: Dict[str, float]
    ta_snapshot: Dict[str, float]
    regime: str
    confidence: float
    position_pct: float


@dataclass
class HoldSnapshot:
    """Snapshot of a HOLD decision for counterfactual learning."""
    bar_idx: int
    ticker: str
    price: float
    ta_signals: Dict[str, float]
    regime: str


@dataclass
class ExitSnapshot:
    """Snapshot of a closed position for exit-regret learning."""
    bar_idx: int
    ticker: str
    exit_price: float
    entry_price: float
    pnl_pct: float
    held_bars: int
    position_side: str
    agent_signals: Dict[str, float]
    regime: str


# ---------------------------------------------------------------------------
# Regime computation
# ---------------------------------------------------------------------------

class HistoricalCryptoRegimeDetector:
    """Offline version of CryptoRegimeDetector using same HMM logic.

    Identical to analysis/regime_detector_crypto.py but operates on
    pre-loaded OHLCV DataFrames instead of yfinance downloads.
    Uses BIC-based state selection (2-4 states), maps to same labels:
    low_volatility, medium_volatility, high_volatility, extreme_volatility.
    """

    REGIME_LABELS = {0: "low_volatility", 1: "medium_volatility", 2: "high_volatility", 3: "extreme_volatility"}

    def __init__(self, btc_df: pd.DataFrame, vol_window: int = 14, lookback_days: int = 365):
        self.vol_window = vol_window
        self.lookback_days = lookback_days

        # Resample 1h → daily close (same as yfinance BTC-USD daily)
        daily = btc_df["close"].resample("1D").last().dropna()
        self._daily = daily

        # Compute features (same as CryptoRegimeDetector._prepare_features)
        import numpy as np
        log_returns = np.log(daily / daily.shift(1)).dropna()
        realised_vol = log_returns.rolling(window=vol_window).std() * np.sqrt(365)
        vol_of_vol = realised_vol.rolling(window=vol_window).std()

        self._features = pd.DataFrame({
            "returns": log_returns,
            "realised_vol": realised_vol,
            "vol_of_vol": vol_of_vol,
        }).dropna()

        # Fit HMM with BIC state selection (same as CryptoRegimeDetector._fit_hmm)
        self._model = None
        self._n_states = 2
        self._state_labels = {}  # daily date → regime label
        self._fit_and_predict()

    def _fit_and_predict(self):
        """Fit HMM on full history and predict regime for every day."""
        import numpy as np
        from hmmlearn.hmm import GaussianHMM

        X = self._features.values
        if len(X) < 30:
            logger.warning("Insufficient data for HMM (%d days), using fallback", len(X))
            for date in self._features.index:
                self._state_labels[date.date()] = "medium_volatility"
            return

        # BIC search (same as CryptoRegimeDetector)
        best_model = None
        best_bic = float("inf")
        best_n = 2
        for n in [2, 3, 4]:
            try:
                m = GaussianHMM(n_components=n, covariance_type="full",
                                n_iter=100, random_state=42, tol=0.01)
                m.fit(X)
                ll = m.score(X) * len(X)
                k = n * n + n * X.shape[1] * 2 + n - 1
                bic = -2 * ll + k * np.log(len(X))
                if bic < best_bic:
                    best_bic = bic
                    best_model = m
                    best_n = n
            except Exception:
                continue

        if best_model is None:
            for date in self._features.index:
                self._state_labels[date.date()] = "medium_volatility"
            return

        self._model = best_model
        self._n_states = best_n
        states = best_model.predict(X)

        # Map states: 0=lowest vol → highest vol (same as CryptoRegimeDetector._map_states)
        vol_idx = list(self._features.columns).index("realised_vol")
        state_vols = {}
        for s in range(best_n):
            mask = states == s
            state_vols[s] = float(self._features.iloc[mask, vol_idx].mean()) if mask.sum() > 0 else 0.0
        sorted_states = sorted(state_vols, key=lambda s: state_vols[s])
        state_map = {old: new for new, old in enumerate(sorted_states)}

        for i, date in enumerate(self._features.index):
            mapped = state_map.get(int(states[i]), 0)
            self._state_labels[date.date()] = self.REGIME_LABELS.get(mapped, "medium_volatility")

        logger.info("HMM regime detector: %d states (BIC=%.1f), %d days, "
                     "distribution=%s",
                     best_n, best_bic, len(states),
                     {v: sum(1 for x in self._state_labels.values() if x == v)
                      for v in set(self._state_labels.values())})

    def get_regime(self, timestamp: pd.Timestamp) -> str:
        """Get regime label for a given timestamp."""
        if hasattr(timestamp, 'date'):
            d = timestamp.date()
        else:
            d = timestamp
        # Find closest date <= timestamp
        label = self._state_labels.get(d)
        if label:
            return label
        # Search backwards up to 5 days
        from datetime import timedelta
        for offset in range(1, 6):
            from datetime import date as date_cls
            try:
                prev = d - timedelta(days=offset)
                label = self._state_labels.get(prev)
                if label:
                    return label
            except Exception:
                pass
        return "medium_volatility"


# Macro regime approximation 2021-01 ~ 2026-03
# Based on actual Fed policy, CPI, GDP data:
#   Growth↑ Inflation↓ = goldilocks
#   Growth↑ Inflation↑ = reflation
#   Growth↓ Inflation↑ = stagflation
#   Growth↓ Inflation↓ = deflation
_MACRO_REGIME_CALENDAR = {
    # 2021: Post-COVID stimulus boom → reflation (growth↑, inflation starting to rise)
    (2021, 1): "reflation", (2021, 2): "reflation", (2021, 3): "reflation",
    (2021, 4): "reflation", (2021, 5): "reflation", (2021, 6): "reflation",
    # 2021 H2: Inflation accelerating, growth strong → reflation
    (2021, 7): "reflation", (2021, 8): "reflation", (2021, 9): "reflation",
    (2021, 10): "reflation", (2021, 11): "reflation", (2021, 12): "reflation",
    # 2022 Q1: Inflation surging, growth slowing → stagflation
    (2022, 1): "stagflation", (2022, 2): "stagflation", (2022, 3): "stagflation",
    # 2022 Q2-Q3: Fed hiking, recession fears, inflation peak → stagflation
    (2022, 4): "stagflation", (2022, 5): "stagflation", (2022, 6): "stagflation",
    (2022, 7): "stagflation", (2022, 8): "stagflation", (2022, 9): "stagflation",
    # 2022 Q4: Inflation rolling over, growth still weak → deflation
    (2022, 10): "deflation", (2022, 11): "deflation", (2022, 12): "deflation",
    # 2023 Q1-Q2: Disinflation, resilient growth → goldilocks
    (2023, 1): "goldilocks", (2023, 2): "goldilocks", (2023, 3): "goldilocks",
    (2023, 4): "goldilocks", (2023, 5): "goldilocks", (2023, 6): "goldilocks",
    # 2023 Q3-Q4: Soft landing narrative, growth OK, inflation falling → goldilocks
    (2023, 7): "goldilocks", (2023, 8): "goldilocks", (2023, 9): "goldilocks",
    (2023, 10): "goldilocks", (2023, 11): "goldilocks", (2023, 12): "goldilocks",
    # 2024 Q1-Q2: Growth steady, inflation sticky above 3% → reflation
    (2024, 1): "reflation", (2024, 2): "reflation", (2024, 3): "reflation",
    (2024, 4): "reflation", (2024, 5): "reflation", (2024, 6): "reflation",
    # 2024 Q3: Inflation easing, growth OK → goldilocks
    (2024, 7): "goldilocks", (2024, 8): "goldilocks", (2024, 9): "goldilocks",
    # 2024 Q4: Rate cuts begin, growth + inflation both moderate → goldilocks
    (2024, 10): "goldilocks", (2024, 11): "goldilocks", (2024, 12): "goldilocks",
    # 2025 Q1: Post-election optimism, growth↑, inflation↑ → reflation
    (2025, 1): "reflation", (2025, 2): "reflation", (2025, 3): "reflation",
    # 2025 Q2: Tariff uncertainty, growth slowing → stagflation
    (2025, 4): "stagflation", (2025, 5): "stagflation", (2025, 6): "stagflation",
    # 2025 Q3: Rate cuts resume, growth↑, inflation↑ → reflation
    (2025, 7): "reflation", (2025, 8): "reflation", (2025, 9): "reflation",
    # 2025 Q4: Soft landing hopes → goldilocks
    (2025, 10): "reflation", (2025, 11): "goldilocks", (2025, 12): "goldilocks",
    # 2026 Q1: Tariff shock round 2, growth↓, inflation↑ → stagflation
    (2026, 1): "stagflation", (2026, 2): "stagflation", (2026, 3): "stagflation",
}


def compute_regime_at_bar(
    df: pd.DataFrame, bar_idx: int, all_closes: list, ema9_series: list, ema21_series: list,
    hmm_detector: Optional['HistoricalCryptoRegimeDetector'] = None,
) -> str:
    """Compute compound regime matching LIVE system format EXACTLY.

    Live format: {crypto_vol}_{trend}_{macro}
    Example: high_volatility_ranging_stagflation

    crypto_vol: CryptoRegimeDetector (HMM) → low/medium/high/extreme_volatility
    trend: EMA9 vs EMA21 + Supertrend → uptrend/downtrend/ranging
    macro: MacroRegimeDetector → goldilocks/reflation/stagflation/deflation
    """
    if bar_idx < 200:
        return "unknown"

    closes = all_closes[max(0, bar_idx - 200): bar_idx + 1]

    # 1. Crypto volatility regime from HMM (identical to live CryptoRegimeDetector)
    ts = df.index[bar_idx]
    if hmm_detector:
        crypto_vol = hmm_detector.get_regime(ts)
    else:
        crypto_vol = "medium_volatility"

    # 2. Trend direction (matches runner.py: EMA9 vs EMA21 + Supertrend)
    ema9 = ema9_series[bar_idx]
    ema21 = ema21_series[bar_idx]

    bars = [
        (0, r["open"], r["high"], r["low"], r["close"], r["volume"])
        for _, r in df.iloc[max(0, bar_idx - 200): bar_idx + 1].iterrows()
    ]
    st = _calc_supertrend(bars, 10, 3.0) if len(bars) >= 15 else 0

    if ema9 > ema21 and st > 0:
        trend = "uptrend"
    elif ema9 < ema21 and st < 0:
        trend = "downtrend"
    else:
        trend = "ranging"

    # 3. Macro regime (from calendar — FRED not available offline)
    ts = df.index[bar_idx]
    if hasattr(ts, 'year'):
        macro = _MACRO_REGIME_CALENDAR.get((ts.year, ts.month), "unknown")
    else:
        macro = "unknown"

    return f"{crypto_vol}_{trend}_{macro}"


# ---------------------------------------------------------------------------
# Strategy entry/exit logic
# ---------------------------------------------------------------------------

def _check_entry(
    df: pd.DataFrame,
    bar_idx: int,
    all_closes: list,
    ema9_series: list,
    ema21_series: list,
    bb_cache: dict,
    open_positions: Dict[str, OpenPosition],
    ticker: str,
) -> Optional[Tuple[str, str, float]]:
    """Check if any strategy triggers entry at this bar.

    Returns (strategy_name, side, confidence) or None.
    """
    # Allow up to 2 concurrent strategies (not same strategy twice)
    if len(open_positions) >= 2:
        return None

    closes = all_closes[max(0, bar_idx - 200): bar_idx + 1]
    if len(closes) < 50:
        return None

    price = closes[-1]
    rsi = _calc_rsi(closes, 14)
    stoch = _calc_stoch_rsi(closes, 14, 14)
    bb = _calc_bollinger(closes, 20, 2.0)
    macd = _calc_macd(closes, 12, 26, 9)
    ema9 = ema9_series[bar_idx]
    ema21 = ema21_series[bar_idx]
    prev_ema9 = ema9_series[bar_idx - 1] if bar_idx > 0 else ema9
    prev_ema21 = ema21_series[bar_idx - 1] if bar_idx > 0 else ema21

    bars = [
        (0, r["open"], r["high"], r["low"], r["close"], r["volume"])
        for _, r in df.iloc[max(0, bar_idx - 200): bar_idx + 1].iterrows()
    ]

    # --- Mean Reversion ---
    if rsi < 30 and (stoch < 20 or bb["pct_b"] < 0.1):
        return ("mean_reversion", "long", 0.65)
    if rsi > 70 and (stoch > 80 or bb["pct_b"] > 0.9):
        return ("mean_reversion", "short", 0.65)

    # --- Momentum (EMA cross) ---
    if prev_ema9 <= prev_ema21 and ema9 > ema21 and macd["histogram"] > 0:
        return ("momentum", "long", 0.60)
    if prev_ema9 >= prev_ema21 and ema9 < ema21 and macd["histogram"] < 0:
        return ("momentum", "short", 0.60)

    # --- Breakout ---
    bw_20th = bb_cache.get("bw_20th", 0.02)
    volumes = df["volume"].iloc[max(0, bar_idx - 20): bar_idx + 1].tolist()
    avg_vol = sum(volumes[:-1]) / max(1, len(volumes) - 1) if len(volumes) > 1 else 1
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

    if bb["bandwidth"] < bw_20th and vol_ratio > 1.5:
        if price > bb["upper"]:
            return ("breakout", "long", 0.55)
        elif price < bb["lower"]:
            return ("breakout", "short", 0.55)

    # --- Trend Follow ---
    if len(bars) >= 15:
        st_dir = _calc_supertrend(bars, 10, 3.0)
        trend_str = abs(ema9 - ema21) / ema21 if ema21 > 0 else 0
        if st_dir > 0 and macd["macd"] > macd["signal"] and trend_str > 0.003:
            return ("trend_follow", "long", 0.70)
        if st_dir < 0 and macd["macd"] < macd["signal"] and trend_str > 0.003:
            return ("trend_follow", "short", 0.70)

    return None


def _check_exit(
    df: pd.DataFrame,
    bar_idx: int,
    pos: OpenPosition,
    all_closes: list,
    ema9_series: list,
    ema21_series: list,
) -> bool:
    """Check if position should be exited at this bar."""
    closes = all_closes[max(0, bar_idx - 200): bar_idx + 1]
    price = closes[-1]
    entry = pos.entry_price
    held = bar_idx - pos.entry_idx

    if pos.side == "long":
        pnl = (price - entry) / entry
    else:
        pnl = (entry - price) / entry

    # Universal stops
    if pnl <= -0.03:  # -3% stop loss
        return True
    if held >= 72:  # max 72h hold
        return True

    strat = pos.strategy

    if strat == "mean_reversion":
        rsi = _calc_rsi(closes, 14)
        if pos.side == "long" and rsi > 70:
            return True
        if pos.side == "short" and rsi < 30:
            return True
        if pnl >= 0.04:  # +4% TP
            return True
        if held >= 24:  # 24h timeout
            return True

    elif strat == "momentum":
        ema9 = ema9_series[bar_idx]
        ema21 = ema21_series[bar_idx]
        if pos.side == "long" and ema9 < ema21:
            return True
        if pos.side == "short" and ema9 > ema21:
            return True
        if pnl >= 0.05:  # +5% TP
            return True
        # Trailing stop 2%
        if held >= 3:  # min 3 bar hold
            highs = df["high"].iloc[pos.entry_idx: bar_idx + 1]
            lows = df["low"].iloc[pos.entry_idx: bar_idx + 1]
            if pos.side == "long":
                best = highs.max()
                trail = (price - best) / best
            else:
                best = lows.min()
                trail = (best - price) / best
            if trail <= -0.02:
                return True

    elif strat == "breakout":
        bb = _calc_bollinger(closes, 20, 2.0)
        if pos.side == "long" and price <= bb["middle"]:
            return True
        if pos.side == "short" and price >= bb["middle"]:
            return True

    elif strat == "trend_follow":
        bars = [
            (0, r["open"], r["high"], r["low"], r["close"], r["volume"])
            for _, r in df.iloc[max(0, bar_idx - 200): bar_idx + 1].iterrows()
        ]
        if len(bars) >= 15:
            st = _calc_supertrend(bars, 10, 3.0)
            if pos.side == "long" and st < 0:
                return True
            if pos.side == "short" and st > 0:
                return True
        # Trailing stop 3%
        if held >= 3:
            highs = df["high"].iloc[pos.entry_idx: bar_idx + 1]
            lows = df["low"].iloc[pos.entry_idx: bar_idx + 1]
            if pos.side == "long":
                best = highs.max()
                trail = (price - best) / best
            else:
                best = lows.min()
                trail = (best - price) / best
            if trail <= -0.03:
                return True

    return False


def _compute_mfe_mae(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    side: str,
) -> Tuple[float, float, float]:
    """Compute MFE, MAE, capture_ratio."""
    if entry_idx >= exit_idx or entry_price <= 0:
        return 0.0, 0.0, 0.0

    highs = df["high"].iloc[entry_idx: exit_idx + 1]
    lows = df["low"].iloc[entry_idx: exit_idx + 1]
    exit_price = df.iloc[exit_idx]["close"]

    if side == "long":
        mfe = (highs.max() - entry_price) / entry_price
        mae = (lows.min() - entry_price) / entry_price
        actual_pnl = (exit_price - entry_price) / entry_price
    else:
        mfe = (entry_price - lows.min()) / entry_price
        mae = (entry_price - highs.max()) / entry_price
        actual_pnl = (entry_price - exit_price) / entry_price

    cap = actual_pnl / mfe if mfe > 0.001 else 0.0
    return mfe, mae, max(0.0, min(1.0, cap))


# ---------------------------------------------------------------------------
# Full simulation loop (mirrors runner.py event loop)
# ---------------------------------------------------------------------------

def simulate_full_loop(
    ticker: str,
    df: pd.DataFrame,
    learner: HierarchicalOnlineLearner,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ls_ratio_df: Optional[pd.DataFrame] = None,
    hmm_detector: Optional[HistoricalCryptoRegimeDetector] = None,
) -> Dict:
    """Simulate the FULL trading + learning loop for one ticker.

    This mirrors runner.py's event loop:
    - Every bar: decide entry/hold/exit
    - HOLD → save snapshot for CF learning
    - Trade exit → record_trade + exit regret snapshot
    - Every N bars → check counterfactuals (3 horizons)
    - Every N bars → check exit regrets
    """
    n = len(df)
    if n < 250:
        return {"trades": 0, "cf": 0, "exit_regrets": 0}

    all_closes = df["close"].tolist()
    ema9_series = _calc_ema_series(all_closes, 9)
    ema21_series = _calc_ema_series(all_closes, 21)

    # Pre-compute BB bandwidth percentile for breakout
    bandwidths = []
    for i in range(200, n):
        bb = _calc_bollinger(all_closes[max(0, i - 200): i + 1], 20, 2.0)
        bandwidths.append(bb["bandwidth"])
    bw_20th = sorted(bandwidths)[int(len(bandwidths) * 0.20)] if bandwidths else 0.02
    bb_cache = {"bw_20th": bw_20th}

    # State — keyed by strategy to allow multiple concurrent positions
    open_positions: Dict[str, OpenPosition] = {}  # key: strategy name
    hold_snapshots: deque = deque(maxlen=300)
    swing_snapshots: deque = deque(maxlen=200)
    trend_snapshots: deque = deque(maxlen=100)
    exit_snapshots: deque = deque(maxlen=50)

    stats = {"trades": 0, "cf": 0, "correct_hold": 0, "exit_regrets": 0, "wins": 0, "pnl": 0.0}
    strategy_counts: Dict[str, int] = {}
    cf_count = 0
    cf_cap_per_period = 80  # cap per 100-bar window

    for i in range(200, n):
        price = all_closes[i]
        regime = compute_regime_at_bar(df, i, all_closes, ema9_series, ema21_series, hmm_detector)

        # --- Check exits first ---
        for strat_key in list(open_positions.keys()):
            pos = open_positions[strat_key]
            if _check_exit(df, i, pos, all_closes, ema9_series, ema21_series):
                exit_price = price
                held_bars = i - pos.entry_idx
                if pos.side == "long":
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - ROUND_TRIP_FEE
                else:
                    pnl_pct = (pos.entry_price - exit_price) / pos.entry_price - ROUND_TRIP_FEE

                mfe, mae, cap = _compute_mfe_mae(df, pos.entry_idx, i, pos.entry_price, pos.side)

                merged_signals = dict(pos.ta_snapshot)
                merged_signals.update(pos.agent_signals)

                learner.record_trade(
                    ticker=ticker,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    held_hours=float(held_bars),
                    agent_signals=merged_signals,
                    regime=regime,
                    position_pct_used=pos.position_pct,
                    confidence_at_entry=pos.confidence,
                    position_side=pos.side,
                    ta_snapshot=None,
                    mfe=mfe,
                    mae=mae,
                    capture_ratio=cap,
                )

                stats["trades"] += 1
                stats["pnl"] += pnl_pct
                if pnl_pct > 0:
                    stats["wins"] += 1
                strategy_counts[pos.strategy] = strategy_counts.get(pos.strategy, 0) + 1

                exit_snapshots.append(ExitSnapshot(
                    bar_idx=i, ticker=ticker, exit_price=exit_price,
                    entry_price=pos.entry_price, pnl_pct=pnl_pct,
                    held_bars=held_bars, position_side=pos.side,
                    agent_signals=merged_signals, regime=regime,
                ))

                del open_positions[strat_key]

        # --- Check ALL strategy entries (multiple concurrent allowed) ---
        for strat_name in ["mean_reversion", "momentum", "breakout", "trend_follow"]:
            if strat_name in open_positions:
                continue  # this strategy already has a position
            entry = _check_entry(df, i, all_closes, ema9_series, ema21_series, bb_cache, open_positions, ticker)
            if entry:
                strategy, side, confidence = entry
                if strategy != strat_name:
                    continue  # not this strategy's turn

                ta_snapshot = compute_ta_at_bar(df, i, 200)
                add_derivatives_signals(ta_snapshot, df.index[i], price, funding_df, oi_df, ls_ratio_df)
                agent_signals = _strategy_signal_subset(ta_snapshot, strategy, side)
                position_pct = random.uniform(0.05, 0.15)

                open_positions[strategy] = OpenPosition(
                    ticker=ticker, strategy=strategy, side=side,
                    entry_idx=i, entry_price=price,
                    agent_signals=agent_signals, ta_snapshot=ta_snapshot,
                    regime=regime, confidence=confidence, position_pct=position_pct,
                )

        # HOLD snapshots (when no positions open)
        if not open_positions and i % 2 == 0:  # every 2 bars
                ta_signals = compute_ta_at_bar(df, i, 200)
                add_derivatives_signals(ta_signals, df.index[i], price, funding_df, oi_df, ls_ratio_df)
                if ta_signals:
                    snap = HoldSnapshot(
                        bar_idx=i, ticker=ticker, price=price,
                        ta_signals=ta_signals, regime=regime,
                    )
                    hold_snapshots.append(snap)
                    swing_snapshots.append(HoldSnapshot(
                        bar_idx=i, ticker=ticker, price=price,
                        ta_signals=dict(ta_signals), regime=regime,
                    ))
                    trend_snapshots.append(HoldSnapshot(
                        bar_idx=i, ticker=ticker, price=price,
                        ta_signals=dict(ta_signals), regime=regime,
                    ))

        # --- Counterfactual checks (every 5 bars, matches heartbeat frequency) ---
        if i % 5 == 0 and cf_count < cf_cap_per_period:
            # 3 horizons matching runner.py line 787-792
            cf_horizons = [
                # (deque, min_bars, max_bars, discount, label)
                (hold_snapshots, 10, 30, 0.30, "scalp"),    # 10-30h → bars
                (swing_snapshots, 6, 24, 0.25, "swing"),    # 6-24h
                (trend_snapshots, 24, 72, 0.20, "trend"),   # 24-72h
            ]

            for snaps, min_age, max_age, discount, label in cf_horizons:
                to_remove = []
                for snap in snaps:
                    age = i - snap.bar_idx
                    if age < min_age:
                        continue
                    if age > max_age:
                        to_remove.append(snap)
                        continue

                    current_price = all_closes[i]
                    raw_pnl = (current_price - snap.price) / snap.price

                    if cf_count >= cf_cap_per_period:
                        to_remove.append(snap)
                        continue

                    if raw_pnl > 0:
                        # Missed buy opportunity (runner line 816-830)
                        cf_count += 1
                        learner.record_counterfactual(
                            ticker=ticker,
                            price_at_hold=snap.price,
                            price_now=current_price,
                            ta_signals=snap.ta_signals,
                            regime=snap.regime,
                            discount_factor=discount,
                        )
                        stats["cf"] += 1
                    else:
                        # Price dropped → correct hold (runner line 844-871)
                        cf_count += 1
                        learner.record_correct_hold(
                            ticker=ticker,
                            price_at_hold=snap.price,
                            price_now=current_price,
                            ta_signals=snap.ta_signals,
                            regime=snap.regime,
                            discount_factor=discount,
                        )
                        stats["correct_hold"] += 1

                        # Missed SHORT (runner line 876-893)
                        short_pnl = -raw_pnl
                        if short_pnl > 0.005 and cf_count < cf_cap_per_period:
                            cf_count += 1
                            learner.record_counterfactual(
                                ticker=ticker,
                                price_at_hold=snap.price,
                                price_now=current_price,
                                ta_signals=snap.ta_signals,
                                regime=snap.regime,
                                discount_factor=discount * 0.5,
                            )
                            stats["cf"] += 1

                    to_remove.append(snap)

                for snap in to_remove:
                    try:
                        snaps.remove(snap)
                    except ValueError:
                        pass

            # Reset CF cap every 100 bars
            if i % 100 == 0:
                cf_count = 0

        # --- Exit regret checks (every 3 bars) ---
        if i % 3 == 0:
            exit_regret_horizons = [
                # (min_bars, max_bars, discount, label)
                (3, 10, 0.15, "quick"),      # 3-10h
                (10, 30, 0.12, "medium"),     # 10-30h
                (30, 72, 0.10, "extended"),   # 30-72h
            ]

            to_remove_exits = []
            for snap in exit_snapshots:
                age = i - snap.bar_idx
                current_price = all_closes[i]

                for min_age, max_age, discount, label in exit_regret_horizons:
                    if age < min_age or age > max_age:
                        continue

                    # Hypothetical PnL if still held (runner line 107-112)
                    if snap.position_side == "short":
                        hypo_pnl = (snap.entry_price - current_price) / snap.entry_price
                    else:
                        hypo_pnl = (current_price - snap.entry_price) / snap.entry_price

                    additional_pnl = hypo_pnl - snap.pnl_pct

                    if additional_pnl > 0.003:
                        # Premature exit (runner line 924-938)
                        learner.record_exit_regret(
                            ticker=snap.ticker,
                            exit_price=snap.exit_price,
                            price_now=current_price,
                            pnl_at_exit=snap.pnl_pct,
                            held_hours=float(snap.held_bars),
                            agent_signals=snap.agent_signals,
                            regime=snap.regime,
                            position_side=snap.position_side,
                            discount_factor=discount,
                            was_premature=True,
                        )
                        stats["exit_regrets"] += 1
                        to_remove_exits.append(snap)
                        break

                    elif additional_pnl < -0.003:
                        # Validated exit (runner line 954-968)
                        learner.record_exit_regret(
                            ticker=snap.ticker,
                            exit_price=snap.exit_price,
                            price_now=current_price,
                            pnl_at_exit=snap.pnl_pct,
                            held_hours=float(snap.held_bars),
                            agent_signals=snap.agent_signals,
                            regime=snap.regime,
                            position_side=snap.position_side,
                            discount_factor=discount,
                            was_premature=False,
                        )
                        stats["exit_regrets"] += 1
                        to_remove_exits.append(snap)
                        break

                # Expire old snapshots
                if age > 72:
                    to_remove_exits.append(snap)

            for snap in to_remove_exits:
                try:
                    exit_snapshots.remove(snap)
                except ValueError:
                    pass

    # Close any remaining positions at last bar
    for strat_key, pos in list(open_positions.items()):
        exit_price = all_closes[-1]
        held_bars = n - 1 - pos.entry_idx
        if pos.side == "long":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - ROUND_TRIP_FEE
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price - ROUND_TRIP_FEE
        mfe, mae, cap = _compute_mfe_mae(df, pos.entry_idx, n - 1, pos.entry_price, pos.side)
        merged = dict(pos.ta_snapshot)
        merged.update(pos.agent_signals)
        learner.record_trade(
            ticker=ticker, entry_price=pos.entry_price, exit_price=exit_price,
            pnl_pct=pnl_pct, held_hours=float(held_bars),
            agent_signals=merged, regime=compute_regime_at_bar(df, n - 1, all_closes, ema9_series, ema21_series),
            position_pct_used=pos.position_pct, confidence_at_entry=pos.confidence,
            position_side=pos.side, mfe=mfe, mae=mae, capture_ratio=cap,
        )
        stats["trades"] += 1
        stats["pnl"] += pnl_pct
        if pnl_pct > 0:
            stats["wins"] += 1

    stats["strategy_counts"] = strategy_counts
    return stats


def _strategy_signal_subset(
    ta_snapshot: Dict[str, float],
    strategy: str,
    side: str,
) -> Dict[str, float]:
    """Select signal subset that this strategy 'focuses on'.

    Simulates Claude's signal_weights — each strategy emphasizes
    different signals, just like Claude would.
    """
    # Each strategy has primary and secondary signal sets
    strategy_focus = {
        "mean_reversion": {
            "primary": ["rsi_signal", "stoch_rsi", "bb_deviation", "vwap_deviation", "support_resistance", "mfi_signal"],
            "secondary": ["volume_spike", "funding_rate", "bb_squeeze"],
        },
        "momentum": {
            "primary": ["ema_cross_fast", "ema_cross_slow", "macd_histogram", "trend_strength"],
            "secondary": ["supertrend", "volume_spike", "oi_change"],
        },
        "breakout": {
            "primary": ["bb_squeeze", "volume_spike", "bb_deviation", "supertrend"],
            "secondary": ["ema_cross_fast", "macd_histogram", "cvd_signal"],
        },
        "trend_follow": {
            "primary": ["supertrend", "macd_histogram", "trend_strength", "ema_cross_slow"],
            "secondary": ["ema_cross_fast", "market_regime", "funding_rate"],
        },
    }

    focus = strategy_focus.get(strategy, {"primary": [], "secondary": []})
    result = {}

    # Include all primary signals
    for sig in focus["primary"]:
        if sig in ta_snapshot:
            result[sig] = ta_snapshot[sig]

    # Include secondary signals with 50% chance (simulates Claude's variable attention)
    for sig in focus["secondary"]:
        if sig in ta_snapshot and random.random() > 0.5:
            result[sig] = ta_snapshot[sig]

    # Ensure at least 3 signals
    if len(result) < 3:
        for sig, val in ta_snapshot.items():
            if sig not in result and abs(val) >= 0.1:
                result[sig] = val
            if len(result) >= 5:
                break

    return result


def simulate_all_tickers(
    ticker_data: Dict[str, Dict[str, pd.DataFrame]],
    learner: HierarchicalOnlineLearner,
) -> Dict:
    """Run full simulation loop for all tickers sequentially."""
    total_stats = {"trades": 0, "cf": 0, "correct_hold": 0, "exit_regrets": 0, "wins": 0, "pnl": 0.0}

    # Build HMM regime detector from BTC data (same as live CryptoRegimeDetector)
    btc_df = ticker_data.get("BTC/USDT", {}).get("klines")
    hmm_detector = None
    if btc_df is not None and not btc_df.empty:
        logger.info("Fitting HMM CryptoRegimeDetector on BTC data (%d bars)...", len(btc_df))
        hmm_detector = HistoricalCryptoRegimeDetector(btc_df)
    else:
        logger.warning("No BTC data for HMM regime detector, using fallback")

    for ticker, data in ticker_data.items():
        df = data.get("klines")
        if df is None or df.empty:
            logger.warning("No klines for %s, skipping", ticker)
            continue

        logger.info("Simulating full loop for %s (%d bars)...", ticker, len(df))

        stats = simulate_full_loop(
            ticker=ticker,
            df=df,
            learner=learner,
            funding_df=data.get("funding"),
            oi_df=data.get("oi"),
            ls_ratio_df=data.get("ls_ratio"),
            hmm_detector=hmm_detector,
        )

        for k in ["trades", "cf", "correct_hold", "exit_regrets", "wins", "pnl"]:
            total_stats[k] += stats.get(k, 0)

        logger.info("  %s: trades=%d (WR=%.1f%%), CF=%d, correct_hold=%d, exit_regrets=%d",
                     ticker, stats["trades"],
                     stats["wins"] / max(1, stats["trades"]) * 100,
                     stats["cf"], stats["correct_hold"], stats["exit_regrets"])
        logger.info("  Strategy breakdown: %s", stats.get("strategy_counts", {}))

    logger.info("TOTAL: trades=%d (WR=%.1f%%), CF=%d, correct_hold=%d, exit_regrets=%d, PnL=%+.2f%%",
                 total_stats["trades"],
                 total_stats["wins"] / max(1, total_stats["trades"]) * 100,
                 total_stats["cf"], total_stats["correct_hold"],
                 total_stats["exit_regrets"], total_stats["pnl"] * 100)

    return total_stats
