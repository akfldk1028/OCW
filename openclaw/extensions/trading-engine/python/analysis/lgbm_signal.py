"""LightGBM-based short-timeframe signal generator for crypto trading.

Based on:
  - arXiv 2511.00665: TA + ML (LightGBM vs LSTM for Bitcoin)
  - arXiv 2503.18096: Informer for HF Bitcoin (5m/15m/30m)
  - arXiv 2309.00626: Ensemble DRL for intraday crypto

Features: 50+ technical indicators from OHLCV data
Target:   Next N-bar return direction (binary: up/down)
Model:    LightGBM binary classifier with rolling retrain
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LGBM_SIGNAL_CONFIG

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ------------------------------------------------------------------ #
# Feature Engineering
# ------------------------------------------------------------------ #

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 50+ TA features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
            (lowercase). Index = datetime.

    Returns:
        DataFrame with feature columns, NaN rows dropped.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feat = pd.DataFrame(index=df.index)

    # --- Price returns ---
    for p in [1, 4, 16, 64]:  # 15m, 1h, 4h, 16h
        feat[f"ret_{p}"] = c.pct_change(p)

    # --- RSI ---
    for period in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # --- MACD ---
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # --- EMAs ---
    for span in [9, 21, 50]:
        ema = c.ewm(span=span).mean()
        feat[f"ema_{span}_dist"] = (c - ema) / ema  # distance from EMA

    # --- SMAs ---
    for win in [20, 50]:
        sma = c.rolling(win).mean()
        feat[f"sma_{win}_dist"] = (c - sma) / sma

    # --- Bollinger Bands ---
    for win in [20]:
        sma = c.rolling(win).mean()
        std = c.rolling(win).std()
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        bb_width = (bb_upper - bb_lower) / sma
        feat[f"bb_pctb_{win}"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        feat[f"bb_width_{win}"] = bb_width

    # --- ATR ---
    for period in [14]:
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        feat[f"atr_{period}"] = atr / c  # normalized

    # --- ADX ---
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    # Zero out when other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    ], axis=1).max(axis=1).rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    feat["adx_14"] = dx.rolling(14).mean()

    # --- Stochastic ---
    for period in [14]:
        lowest = l.rolling(period).min()
        highest = h.rolling(period).max()
        feat[f"stoch_k_{period}"] = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
        feat[f"stoch_d_{period}"] = feat[f"stoch_k_{period}"].rolling(3).mean()

    # --- CCI ---
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    feat["cci_20"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

    # --- Williams %R ---
    highest14 = h.rolling(14).max()
    lowest14 = l.rolling(14).min()
    feat["williams_r"] = -100 * (highest14 - c) / (highest14 - lowest14).replace(0, np.nan)

    # --- ROC ---
    for period in [10, 20]:
        feat[f"roc_{period}"] = c.pct_change(period) * 100

    # --- MFI ---
    tp_mfi = (h + l + c) / 3
    raw_mf = tp_mfi * v
    pos_mf = raw_mf.where(tp_mfi > tp_mfi.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp_mfi <= tp_mfi.shift(1), 0).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    feat["mfi_14"] = 100 - 100 / (1 + mfr)

    # --- OBV ---
    obv = (v * np.sign(c.diff())).fillna(0).cumsum()
    feat["obv_slope"] = obv.pct_change(10)

    # --- Volume features ---
    vol_sma = v.rolling(20).mean()
    feat["vol_ratio"] = v / vol_sma.replace(0, np.nan)
    feat["vol_std"] = v.rolling(20).std() / vol_sma.replace(0, np.nan)

    # --- Volatility features ---
    for win in [16, 64, 192]:  # 4h, 16h, 48h
        feat[f"volatility_{win}"] = c.pct_change().rolling(win).std()

    # --- Candle patterns ---
    body = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)
    feat["body_ratio"] = body / full_range
    feat["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / full_range
    feat["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / full_range

    # --- Cross-ticker momentum (if multi-asset) ---
    # Will be added externally for multi-asset case

    # Drop rows with NaN
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

class LGBMSignalModel:
    """LightGBM binary classifier for crypto direction prediction."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or LGBM_SIGNAL_CONFIG
        self.models = {}  # {ticker: lgb.LGBMClassifier}
        self.feature_names = None
        self.last_train_idx = {}  # {ticker: last_train_bar_index}

    def _make_target(self, close: pd.Series, horizon: int) -> pd.Series:
        """Binary target: 1 if price goes up in next `horizon` bars."""
        future_ret = close.shift(-horizon) / close - 1
        return (future_ret > 0).astype(int)

    def train(self, ohlcv: pd.DataFrame, ticker: str) -> dict:
        """Train LightGBM on OHLCV data for a single ticker.

        Args:
            ohlcv: DataFrame with [open, high, low, close, volume]
            ticker: ticker name for model storage

        Returns:
            dict with train metrics {auc, n_samples, n_features}
        """
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed. pip install lightgbm")

        features = compute_features(ohlcv)
        target = self._make_target(ohlcv["close"], self.cfg["predict_horizon_bars"])

        # Align
        valid = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid]
        y = target.loc[valid]

        if len(X) < self.cfg["min_train_samples"]:
            return {"error": f"insufficient data: {len(X)} < {self.cfg['min_train_samples']}"}

        # Train/val split (last 20% for validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        params = dict(self.cfg["lgbm_params"])
        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        self.models[ticker] = model
        self.feature_names = list(X.columns)
        self.last_train_idx[ticker] = len(ohlcv)

        # Validation AUC
        val_pred = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_pred)

        return {
            "auc": round(auc, 4),
            "n_samples": len(X_train),
            "n_val": len(X_val),
            "n_features": len(self.feature_names),
            "val_accuracy": round((model.predict(X_val) == y_val).mean(), 4),
        }

    def predict(self, ohlcv: pd.DataFrame, ticker: str) -> dict:
        """Predict direction for the latest bar.

        Returns:
            {signal: "BUY"/"SELL"/"HOLD", proba: float, reason: str}
        """
        if ticker not in self.models:
            return {"signal": "HOLD", "proba": 0.5, "reason": "no_model"}

        features = compute_features(ohlcv)
        if features.empty:
            return {"signal": "HOLD", "proba": 0.5, "reason": "no_features"}

        # Last valid row
        last_feat = features.dropna().iloc[-1:]
        if last_feat.empty:
            return {"signal": "HOLD", "proba": 0.5, "reason": "nan_features"}

        model = self.models[ticker]
        proba = model.predict_proba(last_feat)[:, 1][0]

        if proba > self.cfg["buy_threshold"]:
            signal = "BUY"
            reason = f"LGBM: p={proba:.3f}"
        elif proba < self.cfg["sell_threshold"]:
            signal = "SELL"
            reason = f"LGBM: p={proba:.3f}"
        else:
            signal = "HOLD"
            reason = f"LGBM: p={proba:.3f} (neutral)"

        return {"signal": signal, "proba": round(proba, 4), "reason": reason}

    def needs_retrain(self, ticker: str, current_len: int) -> bool:
        """Check if model needs retraining based on new data bars."""
        if ticker not in self.last_train_idx:
            return True
        bars_per_day = 96  # 15min bars per day
        retrain_bars = self.cfg["retrain_interval_days"] * bars_per_day
        return (current_len - self.last_train_idx[ticker]) >= retrain_bars

    def feature_importance(self, ticker: str, top_n: int = 15) -> list:
        """Return top N important features."""
        if ticker not in self.models or self.feature_names is None:
            return []
        imp = self.models[ticker].feature_importances_
        pairs = sorted(zip(self.feature_names, imp), key=lambda x: -x[1])
        return [(name, int(score)) for name, score in pairs[:top_n]]
