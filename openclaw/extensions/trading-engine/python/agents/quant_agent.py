"""Cross-sectional factor analyst agent — XGBoost top-quartile prediction (v8).

Ports the strategy validated in backtest_v8.py to a live agent interface.
The agent computes 13 technical factors, Z-score normalizes them cross-
sectionally across the universe, and uses XGBoost to predict P(top-quartile
absolute return).

Key v8 changes from v6:
    - Z-score normalization (NOT percentile ranking) — preserves signal magnitude
    - Top-quartile labeling (NOT SPY-relative) — better learning signal
    - 13 features (added RSI, BB width, 52w high, 3-month momentum)
    - 18-month training window
    - scale_pos_weight for class imbalance (top 25% = ~25% positive)

13 Factors:
    momentum_5d    — short-term momentum (1 week)
    momentum_21d   — medium-term momentum (1 month)
    momentum_63d   — quarterly momentum (strongest factor in literature)
    rsi_14         — Relative Strength Index (mean-reversion signal)
    mfi            — Money Flow Index (volume-weighted RSI)
    obv_slope      — On-Balance Volume slope (accumulation)
    gk_vol         — Garman-Klass volatility (OHLC-based)
    amihud         — Amihud illiquidity (liquidity premium)
    adx            — Average Directional Index (trend strength)
    volume_ratio   — Volume surge ratio (vs 20d SMA)
    bb_width       — Bollinger Band width (volatility regime)
    high_52w_pct   — Proximity to 52-week high (strength signal)
    sector_momentum — Relative sector ETF strength
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

from core.event_bus import EventBus

logger = logging.getLogger("trading-engine.agents.quant")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quant-agent")

# --- Constants (from backtest_v8.py) ---
FEATURE_NAMES = [
    "momentum_5d",
    "momentum_21d",
    "momentum_63d",
    "rsi_14",
    "mfi",
    "obv_slope",
    "gk_vol",
    "amihud",
    "adx",
    "volume_ratio",
    "bb_width",
    "high_52w_pct",
    "sector_momentum",
]

FORWARD_HORIZON = 21
PURGE_DAYS = 21  # Must match FORWARD_HORIZON to prevent look-ahead bias
TOP_QUARTILE = 0.25
DEFAULT_TRAIN_MONTHS = 24  # Increased from 18 to compensate for larger purge window
MODEL_DIR = Path(__file__).parent.parent / "models"

SECTOR_MAP = {
    "Technology":     {"etf": "XLK",  "stocks": ["NVDA","AAPL","MSFT","AVGO","AMD","CRM","ORCL"]},
    "Semis":          {"etf": "SMH",  "stocks": ["TSM","MU","ASML","QCOM"]},
    "Financials":     {"etf": "XLF",  "stocks": ["JPM","V","MA","BAC","GS","WFC"]},
    "Healthcare":     {"etf": "XLV",  "stocks": ["LLY","UNH","JNJ","ABBV","TMO","MRK"]},
    "ConsDisc":       {"etf": "XLY",  "stocks": ["AMZN","TSLA","HD","MCD","NKE"]},
    "Communication":  {"etf": "XLC",  "stocks": ["META","GOOGL","NFLX","DIS","TMUS"]},
    "Industrials":    {"etf": "XLI",  "stocks": ["CAT","GE","RTX","BA","UNP","HON"]},
    "Energy":         {"etf": "XLE",  "stocks": ["XOM","CVX","COP","SLB"]},
    "Staples":        {"etf": "XLP",  "stocks": ["WMT","COST","PG","KO","PEP"]},
    "Biotech":        {"etf": "IBB",  "stocks": ["VRTX","GILD","AMGN","REGN"]},
    "Materials":      {"etf": "XLB",  "stocks": ["LIN","APD","SHW","ECL"]},
    "Utilities":      {"etf": "XLU",  "stocks": ["NEE","DUK","SO","D"]},
    "RealEstate":     {"etf": "XLRE", "stocks": ["PLD","AMT","EQIX","PSA"]},
}

TICKER_SECTOR = {}
for _sec_name, _info in SECTOR_MAP.items():
    for _tic in _info["stocks"]:
        TICKER_SECTOR[_tic] = _sec_name

ALL_STOCKS = sorted(set(tic for s in SECTOR_MAP.values() for tic in s["stocks"]))
ETF_TICKERS = list(set([s["etf"] for s in SECTOR_MAP.values()] + ["SPY"]))

MOMENTUM_WINDOWS = {"2w": 10, "4w": 20, "12w": 60}
MOMENTUM_WEIGHTS = {"2w": 0.3, "4w": 0.4, "12w": 0.3}


# ============================================================
# Technical Indicators
# ============================================================

def _calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(period).sum()
    neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(period).sum()
    pos_val = pos_mf.iloc[-1]
    neg_val = neg_mf.iloc[-1]
    if pd.isna(neg_val) or neg_val == 0:
        return 100.0 if pos_val > 0 else 50.0
    return float(100 - 100 / (1 + pos_val / neg_val))


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.where(loss > 0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _calc_obv_slope(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    if len(close) < period + 1:
        return 0.0
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_recent = obv.iloc[-period:]
    if len(obv_recent) < period:
        return 0.0
    x = np.arange(period, dtype=float)
    y = obv_recent.values.astype(float)
    slope = np.polyfit(x, y, 1)[0]
    avg_vol = volume.iloc[-period:].mean()
    return float(slope / avg_vol) if avg_vol > 0 else 0.0


def _calc_garman_klass(high: pd.Series, low: pd.Series, close: pd.Series,
                       open_: pd.Series, period: int = 20) -> float:
    if len(close) < period:
        return 0.02
    h, l, c, o = high.iloc[-period:], low.iloc[-period:], close.iloc[-period:], open_.iloc[-period:]
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    gk_mean = gk.mean()
    if pd.isna(gk_mean) or gk_mean < 0:
        return 0.02
    return float(np.sqrt(gk_mean * 252))


def _calc_amihud(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    if len(close) < period + 1:
        return 0.0
    returns = close.pct_change().iloc[-period:]
    dollar_vol = (close * volume).iloc[-period:]
    illiq = returns.abs() / dollar_vol
    illiq = illiq.replace([np.inf, -np.inf], np.nan).dropna()
    return float(illiq.mean() * 1e6) if len(illiq) > 0 else 0.0


def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < period * 2 + 1:
        return 25.0
    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / dx_sum.where(dx_sum > 0, 1.0)
    adx = dx.ewm(span=period, adjust=False).mean()
    val = adx.iloc[-1]
    return float(val) if not pd.isna(val) else 25.0


def _calc_bb_width(close: pd.Series, period: int = 20) -> float:
    if len(close) < period:
        return 0.1
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    mid = sma.iloc[-1]
    if pd.isna(mid) or mid <= 0:
        return 0.1
    width = (upper.iloc[-1] - lower.iloc[-1]) / mid
    return float(width) if not pd.isna(width) else 0.1


def _calc_52w_high_pct(close: pd.Series) -> float:
    lookback = min(len(close), 252)
    if lookback < 20:
        return 0.5
    high_52w = close.iloc[-lookback:].max()
    current = close.iloc[-1]
    if pd.isna(high_52w) or high_52w <= 0:
        return 0.5
    return float(current / high_52w)


# ============================================================
# Feature computation — RAW values (no ranking!)
# ============================================================

def calc_features(tic: str, eval_date,
                  stock_close: pd.DataFrame, stock_high: pd.DataFrame,
                  stock_low: pd.DataFrame, stock_volume: pd.DataFrame,
                  stock_open: pd.DataFrame,
                  sector_scores: Optional[Dict[str, float]] = None) -> Optional[Dict]:
    """Compute 13 technical indicators for one ticker (raw values)."""
    if tic not in stock_close.columns:
        return None
    c = stock_close[tic].loc[:eval_date].dropna()
    if len(c) < 63:
        return None
    h = stock_high[tic].loc[:eval_date].dropna() if tic in stock_high.columns else c
    l = stock_low[tic].loc[:eval_date].dropna() if tic in stock_low.columns else c
    v = stock_volume[tic].loc[:eval_date].dropna() if tic in stock_volume.columns else pd.Series(1, index=c.index)
    o = stock_open[tic].loc[:eval_date].dropna() if tic in stock_open.columns else c
    price = float(c.iloc[-1])
    if price <= 0:
        return None
    vol_sma = v.rolling(20).mean()
    if len(vol_sma) > 20 and vol_sma.iloc[-1] < 500_000:
        return None
    min_len = min(len(c), len(h), len(l), len(v), len(o))
    c, h, l, v, o = c.iloc[-min_len:], h.iloc[-min_len:], l.iloc[-min_len:], v.iloc[-min_len:], o.iloc[-min_len:]

    sec_name = TICKER_SECTOR.get(tic, "")
    sec_mom = (sector_scores or {}).get(sec_name, 0.0)

    return {
        "ticker": tic,
        "price": price,
        "momentum_5d": float(c.iloc[-1] / c.iloc[-5] - 1) if len(c) >= 5 else 0,
        "momentum_21d": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0,
        "momentum_63d": float(c.iloc[-1] / c.iloc[-63] - 1) if len(c) >= 63 else 0,
        "rsi_14": _calc_rsi(c),
        "mfi": _calc_mfi(h, l, c, v),
        "obv_slope": _calc_obv_slope(c, v),
        "gk_vol": _calc_garman_klass(h, l, c, o),
        "amihud": _calc_amihud(c, v),
        "adx": _calc_adx(h, l, c),
        "volume_ratio": float(v.iloc[-1] / vol_sma.iloc[-1]) if len(vol_sma) > 20 and vol_sma.iloc[-1] > 0 else 1.0,
        "bb_width": _calc_bb_width(c),
        "high_52w_pct": _calc_52w_high_pct(c),
        "sector_momentum": sec_mom,
    }


def compute_zscore_features(all_tickers: List[str], eval_date,
                            stock_close: pd.DataFrame, stock_high: pd.DataFrame,
                            stock_low: pd.DataFrame, stock_volume: pd.DataFrame,
                            stock_open: pd.DataFrame,
                            sector_scores: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
    """Compute features and Z-score normalize cross-sectionally.

    Unlike percentile ranking (v6/v7) which creates uniform [0, 1]:
    - Z-scores preserve HOW FAR a stock deviates from peers
    - A stock 2σ above mean is very different from 0.5σ
    - XGBoost can split on meaningful magnitude differences
    - Winsorized at ±3σ to limit outlier influence
    """
    raw = {}
    for tic in all_tickers:
        feats = calc_features(tic, eval_date, stock_close, stock_high,
                              stock_low, stock_volume, stock_open, sector_scores)
        if feats is not None:
            raw[tic] = feats
    if len(raw) < 10:
        return raw

    for fname in FEATURE_NAMES:
        vals = np.array([raw[tic][fname] for tic in raw])
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-10:
            for tic in raw:
                raw[tic][fname] = 0.0
        else:
            for tic in raw:
                z = (raw[tic][fname] - mean) / std
                raw[tic][fname] = float(np.clip(z, -3.0, 3.0))
    return raw


# ============================================================
# Labels — top quartile absolute returns
# ============================================================

def _compute_labels(all_tickers: List[str], feature_date_idx: int,
                    horizon: int, stock_close: pd.DataFrame) -> Dict[str, int]:
    """Label stocks: 1 if in top quartile of forward returns, 0 otherwise."""
    target_idx = feature_date_idx + horizon
    if target_idx >= len(stock_close):
        return {}

    returns = {}
    for tic in all_tickers:
        if tic not in stock_close.columns:
            continue
        entry = stock_close[tic].iloc[feature_date_idx]
        exit_ = stock_close[tic].iloc[target_idx]
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
            continue
        returns[tic] = (exit_ / entry) - 1

    if len(returns) < 10:
        return {}

    threshold = np.percentile(list(returns.values()), (1 - TOP_QUARTILE) * 100)
    return {tic: (1 if ret >= threshold else 0) for tic, ret in returns.items()}


def _rank_sectors(etf_close: pd.DataFrame, eval_date, benchmark: str = "SPY") -> Dict[str, float]:
    """Compute sector momentum scores relative to SPY."""
    hist = etf_close.loc[:eval_date]
    if len(hist) < 130:
        return {}
    spy = hist[benchmark].dropna() if benchmark in hist.columns else None
    if spy is None or len(spy) < 130:
        return {}
    spy_rets = {}
    for label, days in MOMENTUM_WINDOWS.items():
        spy_rets[label] = float(spy.iloc[-1] / spy.iloc[-days] - 1) if len(spy) > days else 0
    sector_scores = {}
    for name, info in SECTOR_MAP.items():
        etf = info["etf"]
        if etf not in hist.columns:
            continue
        s = hist[etf].dropna()
        if len(s) < 20:
            continue
        composite = 0.0
        for label, days in MOMENTUM_WINDOWS.items():
            ret = float(s.iloc[-1] / s.iloc[-days] - 1) if len(s) > days else 0.0
            composite += MOMENTUM_WEIGHTS[label] * (ret - spy_rets.get(label, 0))
        sector_scores[name] = composite
    return sector_scores


# ============================================================
# StockRank dataclass
# ============================================================

@dataclass
class StockRank:
    ticker: str
    p_outperform: float   # P(top-quartile return) from XGBoost
    price: float
    rank_features: Dict[str, float]  # z-scored features
    confidence: float     # model confidence (calibrated)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "p_outperform": self.p_outperform,
            "price": self.price,
            "rank_features": self.rank_features,
            "confidence": self.confidence,
        }


# ============================================================
# QuantAgent
# ============================================================

class QuantAgent:
    """Cross-sectional factor analyst — XGBoost top-quartile prediction (v8).

    Ports the strategy validated in backtest_v8.py to live:

    1. Compute 13 factors for all ~64 stocks
    2. Z-score normalize cross-sectionally
    3. XGBoost P(top-quartile) prediction
    4. Return ranked stock list

    Usage::

        agent = QuantAgent(event_bus)
        rankings = await agent.rank_stocks(["NVDA", "AAPL", "MSFT"])
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._model: Optional[XGBClassifier] = None
        self._feature_importances: Optional[np.ndarray] = None
        self._last_train_date: Optional[str] = None
        self._last_rankings: List[StockRank] = []

        self._load_model()

    async def rank_stocks(
        self,
        candidates: List[str],
        sector_scores: Optional[Dict[str, float]] = None,
    ) -> List[StockRank]:
        """Rank stocks by P(top-quartile return).

        Downloads latest OHLCV data, computes z-scored features for the
        full universe (~64 stocks), then returns rankings for *candidates*.
        """
        loop = asyncio.get_running_loop()
        rankings = await loop.run_in_executor(
            _executor, self._rank_stocks_sync, candidates, sector_scores
        )
        self._last_rankings = rankings

        await self._bus.publish("quant.rankings", {
            "rankings": [r.to_dict() for r in rankings],
            "model_trained": self._model is not None,
            "feature_importances": (
                {f: float(w) for f, w in zip(FEATURE_NAMES, self._feature_importances)}
                if self._feature_importances is not None else None
            ),
        })

        logger.info(
            "QuantAgent: ranked %d stocks, top=%s",
            len(rankings),
            [(r.ticker, f"{r.p_outperform:.2f}") for r in rankings[:5]],
        )

        return rankings

    async def retrain(self, months: int = DEFAULT_TRAIN_MONTHS) -> Dict[str, Any]:
        """Retrain XGBoost classifier on a rolling window.

        Should be called weekly via cron job.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, self._retrain_sync, months)
        return result

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def last_rankings(self) -> List[StockRank]:
        return self._last_rankings

    # ------------------------------------------------------------------
    # Sync implementations (run in thread pool)
    # ------------------------------------------------------------------

    def _rank_stocks_sync(
        self,
        candidates: List[str],
        sector_scores: Optional[Dict[str, float]] = None,
    ) -> List[StockRank]:
        """Synchronous ranking — called via executor."""
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=600)  # enough for 252d 52w-high + training

        all_tickers = list(set(ALL_STOCKS + candidates))
        download_tickers = list(set(all_tickers + ETF_TICKERS))

        try:
            data = yf.download(
                download_tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error("QuantAgent data download failed: %s", e)
            return []

        def extract(field):
            if isinstance(data.columns, pd.MultiIndex):
                return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
            return data

        stock_close = extract("Close")
        stock_high = extract("High")
        stock_low = extract("Low")
        stock_volume = extract("Volume")
        stock_open = extract("Open")

        if stock_close.empty:
            return []

        eval_date = stock_close.index[-1]

        # Compute sector scores if not provided
        if sector_scores is None:
            etf_close = stock_close[[c for c in ETF_TICKERS if c in stock_close.columns]]
            sector_scores = _rank_sectors(etf_close, eval_date)

        # Retrain if no model or stale (>7 days)
        if self._model is None or self._is_model_stale():
            self._train_model(
                ALL_STOCKS, eval_date, DEFAULT_TRAIN_MONTHS,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
                sector_scores,
            )

        # Compute z-scored features for full universe
        zscored = compute_zscore_features(
            all_tickers, eval_date,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            sector_scores,
        )

        if not zscored:
            return []

        # Score candidates using XGBoost
        rankings: List[StockRank] = []
        candidate_feats = {t: zscored[t] for t in candidates if t in zscored}

        if self._model is not None and candidate_feats:
            tickers_ordered = list(candidate_feats.keys())
            X = np.array([
                [candidate_feats[t][f] for f in FEATURE_NAMES]
                for t in tickers_ordered
            ])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            probas = self._model.predict_proba(X)
            classes = list(self._model.classes_)

            for tic, proba in zip(tickers_ordered, probas):
                p_dict = {cls: float(p) for cls, p in zip(classes, proba)}
                p_win = p_dict.get(1, 0.25)
                rankings.append(StockRank(
                    ticker=tic,
                    p_outperform=p_win,
                    price=candidate_feats[tic].get("price", 0.0),
                    rank_features={f: candidate_feats[tic][f] for f in FEATURE_NAMES},
                    confidence=abs(p_win - 0.5) * 2,
                ))
        else:
            # Fallback: use momentum z-score as score
            for tic, feats in candidate_feats.items():
                mom_z = feats.get("momentum_21d", 0.0)
                # Convert z-score to pseudo-probability via sigmoid
                p_approx = 1.0 / (1.0 + np.exp(-mom_z))
                rankings.append(StockRank(
                    ticker=tic,
                    p_outperform=float(p_approx),
                    price=feats.get("price", 0.0),
                    rank_features={f: feats[f] for f in FEATURE_NAMES},
                    confidence=0.3,
                ))

        rankings.sort(key=lambda r: r.p_outperform, reverse=True)
        return rankings

    def _retrain_sync(self, months: int = DEFAULT_TRAIN_MONTHS) -> Dict[str, Any]:
        """Synchronous retraining."""
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=(months + 20) * 30 + 300)

        download_tickers = list(set(ALL_STOCKS + ETF_TICKERS))

        try:
            data = yf.download(
                download_tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            return {"error": str(e)}

        def extract(field):
            if isinstance(data.columns, pd.MultiIndex):
                return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
            return data

        stock_close = extract("Close")
        stock_high = extract("High")
        stock_low = extract("Low")
        stock_volume = extract("Volume")
        stock_open = extract("Open")

        if stock_close.empty:
            return {"error": "No data available"}

        eval_date = stock_close.index[-1]
        etf_close = stock_close[[c for c in ETF_TICKERS if c in stock_close.columns]]
        sector_scores = _rank_sectors(etf_close, eval_date)

        result = self._train_model(
            ALL_STOCKS, eval_date, months,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            sector_scores,
        )
        return result

    # ------------------------------------------------------------------
    # Internal training — v8: z-scores + top-quartile labels
    # ------------------------------------------------------------------

    def _train_model(
        self, all_tickers: List[str], eval_date, train_months: int,
        stock_close, stock_high, stock_low, stock_volume, stock_open,
        sector_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Train XGBoost on z-scored features with top-quartile labeling (v8)."""
        X_train, y_train = [], []
        trading_days = stock_close.index
        eval_idx = trading_days.get_indexer([eval_date], method='pad')[0]

        # Pre-compute sector scores for training dates
        etf_cols = [c for c in ETF_TICKERS if c in stock_close.columns]
        etf_close = stock_close[etf_cols] if etf_cols else pd.DataFrame()

        for month_offset in range(train_months, 0, -1):
            feature_date_idx = eval_idx - (month_offset + 1) * 21
            target_end_idx = feature_date_idx + FORWARD_HORIZON

            if feature_date_idx < 63:
                continue
            if target_end_idx >= eval_idx - PURGE_DAYS:
                continue

            feature_date = trading_days[feature_date_idx]

            # Sector scores for this training date
            train_sec_scores = _rank_sectors(etf_close, feature_date) if not etf_close.empty else {}

            zscored = compute_zscore_features(
                all_tickers, feature_date,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
                train_sec_scores,
            )

            # Top-quartile labels
            labels = _compute_labels(all_tickers, feature_date_idx,
                                     FORWARD_HORIZON, stock_close)

            for tic, feats in zscored.items():
                if tic not in labels:
                    continue
                X_train.append([feats[f] for f in FEATURE_NAMES])
                y_train.append(labels[tic])

        if len(X_train) < 80:
            logger.warning("QuantAgent: not enough training data (%d samples)", len(X_train))
            return {"error": "Not enough training data", "samples": len(X_train)}

        X = np.nan_to_num(np.array(X_train), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y_train)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return {"error": "Single class in labels", "distribution": dict(zip(unique.tolist(), counts.tolist()))}

        # Handle class imbalance (top quartile ≈ 25% positive)
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        scale_pos_weight = n_neg / max(n_pos, 1)

        # 80/20 temporal split for OOS validation
        split_idx = int(len(X) * 0.8)
        X_tr, X_oos = X[:split_idx], X[split_idx:]
        y_tr, y_oos = y[:split_idx], y[split_idx:]

        use_early_stop = len(X_oos) > 10 and len(np.unique(y_oos)) > 1
        model = XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=1.5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
            early_stopping_rounds=20 if use_early_stop else None,
        )
        fit_kwargs = {"verbose": False}
        if use_early_stop:
            fit_kwargs["eval_set"] = [(X_oos, y_oos)]
        model.fit(X_tr, y_tr, **fit_kwargs)

        # OOS AUC check — reject model if barely better than random
        oos_auc = None
        if len(X_oos) > 10 and len(np.unique(y_oos)) > 1:
            from sklearn.metrics import roc_auc_score
            oos_probas = model.predict_proba(X_oos)
            classes = list(model.classes_)
            if 1 in classes:
                oos_auc = roc_auc_score(y_oos, oos_probas[:, classes.index(1)])
                if oos_auc < 0.52:
                    logger.warning(
                        "QuantAgent: OOS AUC=%.3f < 0.52 — model rejected (near random)",
                        oos_auc,
                    )
                    return {
                        "error": "Model rejected: OOS AUC below threshold",
                        "oos_auc": float(oos_auc),
                        "samples": len(X_train),
                    }
                logger.info("QuantAgent: OOS AUC=%.3f (pass)", oos_auc)

        # Retrain on full data now that OOS passed
        bi = getattr(model, 'best_iteration', None)
        n_est = (bi + 1) if bi is not None and bi > 0 else 300
        model_full = XGBClassifier(
            max_depth=4,
            n_estimators=n_est,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=1.5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
        )
        model_full.fit(X, y)
        model = model_full

        importances = model.feature_importances_
        total = importances.sum()
        if total > 0:
            importances = importances / total
        else:
            importances = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

        # Overfitting warning: single feature > 40% importance
        max_imp_idx = np.argmax(importances)
        max_imp_val = importances[max_imp_idx]
        overfit_warning = None
        if max_imp_val > 0.40:
            overfit_warning = (
                f"Feature '{FEATURE_NAMES[max_imp_idx]}' has {max_imp_val:.0%} importance "
                f"— possible overfitting on single factor"
            )
            logger.warning("QuantAgent: %s", overfit_warning)

        self._model = model
        self._feature_importances = importances
        self._last_train_date = str(eval_date)

        self._save_model()

        label_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(
            "QuantAgent v8: trained on %d samples, labels=%s, scale_pos_weight=%.2f, OOS_AUC=%s",
            len(X_train), label_dist, scale_pos_weight,
            f"{oos_auc:.3f}" if oos_auc is not None else "n/a",
        )

        return {
            "status": "trained",
            "version": "v8",
            "samples": len(X_train),
            "label_distribution": label_dist,
            "feature_importances": {f: float(w) for f, w in zip(FEATURE_NAMES, importances)},
            "scale_pos_weight": float(scale_pos_weight),
            "oos_auc": float(oos_auc) if oos_auc is not None else None,
            "overfit_warning": overfit_warning,
        }

    def _is_model_stale(self) -> bool:
        if self._last_train_date is None:
            return True
        from datetime import datetime
        try:
            last = pd.Timestamp(self._last_train_date)
            return (pd.Timestamp(datetime.now()) - last).days > 7
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / "quant_xgboost_v8.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "importances": self._feature_importances,
                    "train_date": self._last_train_date,
                    "version": "v8",
                    "feature_names": FEATURE_NAMES,
                }, f)
            logger.info("QuantAgent: model saved to %s", path)
        except Exception as e:
            logger.error("QuantAgent: failed to save model: %s", e)

    def _load_model(self) -> None:
        path = MODEL_DIR / "quant_xgboost_v8.pkl"
        if not path.exists():
            # Try loading v6 model as fallback (won't work with v8 features, but signals stale)
            old_path = MODEL_DIR / "quant_xgboost.pkl"
            if old_path.exists():
                logger.info("QuantAgent: v8 model not found, v6 model exists but incompatible — will retrain")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            saved_features = data.get("feature_names", [])
            if saved_features != FEATURE_NAMES:
                logger.warning("QuantAgent: saved model has different features, will retrain")
                return
            self._model = data.get("model")
            self._feature_importances = data.get("importances")
            self._last_train_date = data.get("train_date")
            logger.info("QuantAgent v8: loaded model from %s (trained %s)", path, self._last_train_date)
        except Exception as e:
            logger.warning("QuantAgent: failed to load model: %s", e)
