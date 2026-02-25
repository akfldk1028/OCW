"""Cross-sectional crypto factor analyst — XGBoost top-quartile prediction.

Adapts the equity QuantAgent (v8) for crypto markets.

Key differences from equity QuantAgent:
    - Forward horizon: 7 days (vs 21 for equities — crypto moves faster)
    - PURGE_DAYS: 3 (vs 5 for equities, proportional to horizon)
    - Training window: 6 months (vs 18 for equities — crypto regimes shift fast)
    - Universe: 12 major crypto pairs via yfinance (BTC-USD, ETH-USD, ...)
    - No sector_momentum → replaced with btc_correlation (30d rolling)
    - high_30d_pct instead of 52w high (crypto lacks meaningful 52w patterns)
    - Momentum windows: 1d/7d/21d (faster than equity 5d/21d/63d)
    - Volume filter: much lower (crypto is 24/7)
    - Data source: yfinance with "-USD" suffix tickers

13 Features:
    momentum_1d, momentum_7d, momentum_21d, rsi_14, mfi, obv_slope,
    gk_vol, amihud, adx, volume_ratio, bb_width, high_30d_pct,
    btc_correlation
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

from agents.quant_agent import (
    StockRank,
    _calc_adx,
    _calc_amihud,
    _calc_bb_width,
    _calc_garman_klass,
    _calc_mfi,
    _calc_obv_slope,
    _calc_rsi,
)
from core.event_bus import EventBus

logger = logging.getLogger("trading-engine.agents.quant_crypto")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quant-crypto")

# --- Constants ---
FEATURE_NAMES = [
    "momentum_1d",
    "momentum_7d",
    "momentum_21d",
    "rsi_14",
    "mfi",
    "obv_slope",
    "gk_vol",
    "amihud",
    "adx",
    "volume_ratio",
    "bb_width",
    "high_30d_pct",
    "btc_correlation",
]

FORWARD_HORIZON = 7
PURGE_DAYS = 3
TOP_QUARTILE = 0.25
DEFAULT_TRAIN_MONTHS = 6
MODEL_DIR = Path(__file__).parent.parent / "models"

# 12 major crypto pairs (exchange notation → yfinance ticker via _normalize_yf_ticker)
CRYPTO_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
    "DOT/USDT", "MATIC/USDT", "LINK/USDT", "NEAR/USDT",
]

BTC_YF_TICKER = "BTC-USD"


def _normalize_yf_ticker(pair: str) -> str:
    """Convert exchange pair notation to yfinance ticker.

    'BTC/USDT' -> 'BTC-USD'
    'ETH/USDT' -> 'ETH-USD'
    Already in yfinance format -> pass through.
    """
    if "/" in pair:
        base = pair.split("/")[0]
        return f"{base}-USD"
    if pair.endswith("-USD"):
        return pair
    return f"{pair}-USD"


ALL_YF_TICKERS = sorted(set(_normalize_yf_ticker(p) for p in CRYPTO_PAIRS))


# ============================================================
# Feature computation
# ============================================================

def _calc_high_30d_pct(close: pd.Series) -> float:
    """Proximity to 30-day high (crypto equivalent of 52w high)."""
    lookback = min(len(close), 30)
    if lookback < 5:
        return 0.5
    high_30d = close.iloc[-lookback:].max()
    current = close.iloc[-1]
    if pd.isna(high_30d) or high_30d <= 0:
        return 0.5
    return float(current / high_30d)


def _calc_btc_correlation(close: pd.Series, btc_close: pd.Series, window: int = 30) -> float:
    """30-day rolling correlation with BTC returns."""
    if len(close) < window + 1 or len(btc_close) < window + 1:
        return 0.0
    # Align indices
    aligned = pd.DataFrame({"asset": close, "btc": btc_close}).dropna()
    if len(aligned) < window + 1:
        return 0.0
    asset_ret = aligned["asset"].pct_change().dropna()
    btc_ret = aligned["btc"].pct_change().dropna()
    if len(asset_ret) < window or len(btc_ret) < window:
        return 0.0
    corr = asset_ret.iloc[-window:].corr(btc_ret.iloc[-window:])
    return float(corr) if not pd.isna(corr) else 0.0


def calc_crypto_features(
    tic: str,
    eval_date,
    stock_close: pd.DataFrame,
    stock_high: pd.DataFrame,
    stock_low: pd.DataFrame,
    stock_volume: pd.DataFrame,
    stock_open: pd.DataFrame,
    btc_close: Optional[pd.Series] = None,
) -> Optional[Dict]:
    """Compute 13 technical indicators for one crypto ticker (raw values)."""
    if tic not in stock_close.columns:
        return None
    c = stock_close[tic].loc[:eval_date].dropna()
    if len(c) < 30:
        return None
    h = stock_high[tic].loc[:eval_date].dropna() if tic in stock_high.columns else c
    l = stock_low[tic].loc[:eval_date].dropna() if tic in stock_low.columns else c
    v = stock_volume[tic].loc[:eval_date].dropna() if tic in stock_volume.columns else pd.Series(1, index=c.index)
    o = stock_open[tic].loc[:eval_date].dropna() if tic in stock_open.columns else c

    price = float(c.iloc[-1])
    if price <= 0:
        return None

    min_len = min(len(c), len(h), len(l), len(v), len(o))
    c, h, l, v, o = c.iloc[-min_len:], h.iloc[-min_len:], l.iloc[-min_len:], v.iloc[-min_len:], o.iloc[-min_len:]

    # BTC correlation
    btc_corr = 0.0
    if btc_close is not None and tic != BTC_YF_TICKER:
        btc_corr = _calc_btc_correlation(c, btc_close.loc[:eval_date].dropna())
    # BTC correlates perfectly with itself — use 1.0
    if tic == BTC_YF_TICKER:
        btc_corr = 1.0

    vol_sma = v.rolling(20).mean()
    vol_ratio = 1.0
    if len(vol_sma) > 20 and not pd.isna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
        vol_ratio = float(v.iloc[-1] / vol_sma.iloc[-1])

    return {
        "ticker": tic,
        "price": price,
        "momentum_1d": float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) >= 2 else 0,
        "momentum_7d": float(c.iloc[-1] / c.iloc[-7] - 1) if len(c) >= 7 else 0,
        "momentum_21d": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0,
        "rsi_14": _calc_rsi(c),
        "mfi": _calc_mfi(h, l, c, v),
        "obv_slope": _calc_obv_slope(c, v),
        "gk_vol": _calc_garman_klass(h, l, c, o),
        "amihud": _calc_amihud(c, v),
        "adx": _calc_adx(h, l, c),
        "volume_ratio": vol_ratio,
        "bb_width": _calc_bb_width(c),
        "high_30d_pct": _calc_high_30d_pct(c),
        "btc_correlation": btc_corr,
    }


def compute_crypto_zscore_features(
    all_tickers: List[str],
    eval_date,
    stock_close: pd.DataFrame,
    stock_high: pd.DataFrame,
    stock_low: pd.DataFrame,
    stock_volume: pd.DataFrame,
    stock_open: pd.DataFrame,
    btc_close: Optional[pd.Series] = None,
) -> Dict[str, Dict]:
    """Compute features and Z-score normalize cross-sectionally."""
    raw = {}
    for tic in all_tickers:
        feats = calc_crypto_features(
            tic, eval_date, stock_close, stock_high, stock_low,
            stock_volume, stock_open, btc_close,
        )
        if feats is not None:
            raw[tic] = feats

    if len(raw) < 4:
        return raw

    for fname in FEATURE_NAMES:
        vals = np.array([raw[tic][fname] for tic in raw])
        mean, std = np.mean(vals), np.std(vals)
        if std < 1e-10:
            for tic in raw:
                raw[tic][fname] = 0.0
        else:
            for tic in raw:
                z = (raw[tic][fname] - mean) / std
                raw[tic][fname] = float(np.clip(z, -3.0, 3.0))
    return raw


# ============================================================
# Labels
# ============================================================

def _compute_crypto_labels(
    all_tickers: List[str],
    feature_date_idx: int,
    horizon: int,
    stock_close: pd.DataFrame,
) -> Dict[str, int]:
    """Label cryptos: 1 if in top quartile of forward returns, 0 otherwise."""
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

    if len(returns) < 4:
        return {}

    threshold = np.percentile(list(returns.values()), (1 - TOP_QUARTILE) * 100)
    return {tic: (1 if ret >= threshold else 0) for tic, ret in returns.items()}


# ============================================================
# CryptoQuantAgent
# ============================================================

class CryptoQuantAgent:
    """Cross-sectional crypto factor analyst — XGBoost top-quartile prediction.

    Mirrors QuantAgent interface but adapted for crypto:
    - 7-day forward horizon
    - 6-month training window
    - 12 major crypto pairs
    - BTC correlation feature instead of sector momentum

    Usage::

        agent = CryptoQuantAgent(event_bus)
        rankings = await agent.rank_stocks(["BTC/USDT", "ETH/USDT"])
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
        **kwargs,
    ) -> List[StockRank]:
        """Rank crypto assets by P(top-quartile return).

        Accepts candidates in exchange format ('BTC/USDT') or
        yfinance format ('BTC-USD').
        """
        loop = asyncio.get_running_loop()
        rankings = await loop.run_in_executor(
            _executor, self._rank_stocks_sync, candidates,
        )
        self._last_rankings = rankings

        await self._bus.publish("quant_crypto.rankings", {
            "rankings": [r.to_dict() for r in rankings],
            "model_trained": self._model is not None,
            "feature_importances": (
                {f: float(w) for f, w in zip(FEATURE_NAMES, self._feature_importances)}
                if self._feature_importances is not None else None
            ),
        })

        logger.info(
            "CryptoQuantAgent: ranked %d assets, top=%s",
            len(rankings),
            [(r.ticker, f"{r.p_outperform:.2f}") for r in rankings[:5]],
        )
        return rankings

    async def retrain(self, months: int = DEFAULT_TRAIN_MONTHS) -> Dict[str, Any]:
        """Retrain XGBoost on rolling window. Call weekly via cron."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._retrain_sync, months)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def last_rankings(self) -> List[StockRank]:
        return self._last_rankings

    # ------------------------------------------------------------------
    # Sync implementations
    # ------------------------------------------------------------------

    def _rank_stocks_sync(self, candidates: List[str]) -> List[StockRank]:
        yf_candidates = [_normalize_yf_ticker(c) for c in candidates]
        all_tickers = sorted(set(ALL_YF_TICKERS + yf_candidates))

        end = datetime.now()
        start = end - timedelta(days=400)

        try:
            data = yf.download(
                all_tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error("CryptoQuantAgent data download failed: %s", e)
            return []

        stock_close, stock_high, stock_low, stock_volume, stock_open = self._extract_ohlcv(data)
        if stock_close.empty:
            return []

        eval_date = stock_close.index[-1]
        btc_close = stock_close[BTC_YF_TICKER] if BTC_YF_TICKER in stock_close.columns else None

        # Retrain if needed
        if self._model is None or self._is_model_stale():
            self._train_model(
                all_tickers, eval_date, DEFAULT_TRAIN_MONTHS,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
                btc_close,
            )

        zscored = compute_crypto_zscore_features(
            all_tickers, eval_date,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            btc_close,
        )
        if not zscored:
            return []

        # Build candidate map (exchange name -> yf ticker)
        candidate_map = {_normalize_yf_ticker(c): c for c in candidates}
        candidate_feats = {t: zscored[t] for t in yf_candidates if t in zscored}

        rankings: List[StockRank] = []

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
                original_name = candidate_map.get(tic, tic)
                rankings.append(StockRank(
                    ticker=original_name,
                    p_outperform=p_win,
                    price=candidate_feats[tic].get("price", 0.0),
                    rank_features={f: candidate_feats[tic][f] for f in FEATURE_NAMES},
                    confidence=abs(p_win - 0.5) * 2,
                ))
        else:
            for tic, feats in candidate_feats.items():
                mom_z = feats.get("momentum_7d", 0.0)
                p_approx = 1.0 / (1.0 + np.exp(-mom_z))
                original_name = candidate_map.get(tic, tic)
                rankings.append(StockRank(
                    ticker=original_name,
                    p_outperform=float(p_approx),
                    price=feats.get("price", 0.0),
                    rank_features={f: feats[f] for f in FEATURE_NAMES},
                    confidence=0.3,
                ))

        rankings.sort(key=lambda r: r.p_outperform, reverse=True)
        return rankings

    def _retrain_sync(self, months: int = DEFAULT_TRAIN_MONTHS) -> Dict[str, Any]:
        end = datetime.now()
        start = end - timedelta(days=(months + 6) * 30 + 100)

        try:
            data = yf.download(
                ALL_YF_TICKERS,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            return {"error": str(e)}

        stock_close, stock_high, stock_low, stock_volume, stock_open = self._extract_ohlcv(data)
        if stock_close.empty:
            return {"error": "No data available"}

        eval_date = stock_close.index[-1]
        btc_close = stock_close[BTC_YF_TICKER] if BTC_YF_TICKER in stock_close.columns else None

        return self._train_model(
            ALL_YF_TICKERS, eval_date, months,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            btc_close,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_model(
        self, all_tickers, eval_date, train_months,
        stock_close, stock_high, stock_low, stock_volume, stock_open,
        btc_close,
    ) -> Dict[str, Any]:
        """Train XGBoost on z-scored crypto features with top-quartile labeling."""
        X_train, y_train = [], []
        trading_days = stock_close.index
        eval_idx = trading_days.get_indexer([eval_date], method="pad")[0]

        # Crypto trades daily, so ~30 days per month
        for month_offset in range(train_months, 0, -1):
            feature_date_idx = eval_idx - (month_offset + 1) * 30
            target_end_idx = feature_date_idx + FORWARD_HORIZON

            if feature_date_idx < 30:
                continue
            if target_end_idx >= eval_idx - PURGE_DAYS:
                continue

            feature_date = trading_days[feature_date_idx]

            zscored = compute_crypto_zscore_features(
                list(all_tickers), feature_date,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
                btc_close,
            )

            labels = _compute_crypto_labels(
                list(all_tickers), feature_date_idx, FORWARD_HORIZON, stock_close,
            )

            for tic, feats in zscored.items():
                if tic not in labels:
                    continue
                X_train.append([feats[f] for f in FEATURE_NAMES])
                y_train.append(labels[tic])

        if len(X_train) < 30:
            logger.warning("CryptoQuantAgent: not enough training data (%d samples)", len(X_train))
            return {"error": "Not enough training data", "samples": len(X_train)}

        X = np.nan_to_num(np.array(X_train), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y_train)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return {"error": "Single class in labels", "distribution": dict(zip(unique.tolist(), counts.tolist()))}

        n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            max_depth=3,
            n_estimators=200,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=1.5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            scale_pos_weight=scale_pos_weight,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X, y)

        importances = model.feature_importances_
        total = importances.sum()
        importances = importances / total if total > 0 else np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

        self._model = model
        self._feature_importances = importances
        self._last_train_date = str(eval_date)
        self._save_model()

        label_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(
            "CryptoQuantAgent: trained on %d samples, labels=%s",
            len(X_train), label_dist,
        )

        return {
            "status": "trained",
            "version": "crypto_v1",
            "samples": len(X_train),
            "label_distribution": label_dist,
            "feature_importances": {f: float(w) for f, w in zip(FEATURE_NAMES, importances)},
            "scale_pos_weight": float(scale_pos_weight),
        }

    def _is_model_stale(self) -> bool:
        if self._last_train_date is None:
            return True
        try:
            last = pd.Timestamp(self._last_train_date)
            return (pd.Timestamp(datetime.now()) - last).days > 7
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ohlcv(data: pd.DataFrame):
        """Extract OHLCV dataframes from yfinance download result."""
        def extract(field):
            if isinstance(data.columns, pd.MultiIndex):
                return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
            return data

        return (
            extract("Close"),
            extract("High"),
            extract("Low"),
            extract("Volume"),
            extract("Open"),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / "quant_crypto_xgboost.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "importances": self._feature_importances,
                    "train_date": self._last_train_date,
                    "version": "crypto_v1",
                    "feature_names": FEATURE_NAMES,
                }, f)
            logger.info("CryptoQuantAgent: model saved to %s", path)
        except Exception as e:
            logger.error("CryptoQuantAgent: failed to save model: %s", e)

    def _load_model(self) -> None:
        path = MODEL_DIR / "quant_crypto_xgboost.pkl"
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("feature_names", []) != FEATURE_NAMES:
                logger.warning("CryptoQuantAgent: saved model has different features, will retrain")
                return
            self._model = data.get("model")
            self._feature_importances = data.get("importances")
            self._last_train_date = data.get("train_date")
            logger.info("CryptoQuantAgent: loaded model from %s (trained %s)", path, self._last_train_date)
        except Exception as e:
            logger.warning("CryptoQuantAgent: failed to load model: %s", e)
