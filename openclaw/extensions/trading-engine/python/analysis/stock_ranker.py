"""XGBoost-based stock ranker with learned factor weights.

Replaces the equal-weight scoring in SectorScanner with a model
that learns which factors actually predict future returns.

Paper backing:
- arXiv 2508.18592: XGBoost + dynamic weighting, ROC-AUC 0.953
- arXiv 2507.07107: Adaptive factor weights >> equal weighting

Approach:
- Train on historical data: features = [momentum, volume_ratio, rsi,
  sentiment, sector_momentum] -> target = next_20d_return > median
- Feature importances become the factor weights
- Retrain monthly or on-demand with latest data

Usage::

    from stock_ranker import StockRanker
    ranker = StockRanker()
    ranker.train()           # fit on historical data
    weights = ranker.get_factor_weights()
    ranked = ranker.rank(candidates_df)
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import SECTOR_MAP, SECTOR_SCAN_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)

RANKER_MODEL_PATH = MODELS_DIR / "stock_ranker.pkl"

# Features used by the ranker
FACTOR_NAMES = ["momentum", "volume_ratio", "rsi_score", "sentiment"]


class StockRanker:
    """XGBoost-based stock scoring and ranking.

    Learns factor weights from historical data instead of
    using equal (25%) weighting.
    """

    def __init__(self) -> None:
        self._model = None
        self._feature_importances: Dict[str, float] = {}
        self._last_train_date: Optional[str] = None
        self._train_metrics: Dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        lookback_months: int = 12,
        forward_days: int = 21,
    ) -> Dict[str, Any]:
        """Train the ranker on historical stock data.

        For each month in the past lookback_months:
        - Compute stock features (momentum, volume_ratio, rsi, etc.)
        - Label: next forward_days return > median = 1, else = 0
        - Train XGBoost classifier

        Returns training metrics.
        """
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, accuracy_score

        logger.info("StockRanker: preparing training data for %d months", lookback_months)

        X, y, dates = self._prepare_training_data(lookback_months, forward_days)

        if X is None or len(X) < 50:
            return {"error": "Insufficient training data", "samples": len(X) if X is not None else 0}

        logger.info("StockRanker: training on %d samples with %d features", len(X), X.shape[1])

        # Time-series aware cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            acc = accuracy_score(y_val, model.predict(X_val))
            cv_scores.append({"auc": auc, "accuracy": acc})

        # Final model on all data
        final_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        final_model.fit(X, y, verbose=False)

        self._model = final_model
        self._last_train_date = datetime.now().strftime("%Y-%m-%d")

        # Extract feature importances as factor weights
        importances = final_model.feature_importances_
        feature_names = FACTOR_NAMES[:X.shape[1]]
        self._feature_importances = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }

        # Normalise to sum to 1
        total = sum(self._feature_importances.values())
        if total > 0:
            self._feature_importances = {
                k: round(v / total, 4)
                for k, v in self._feature_importances.items()
            }

        self._train_metrics = {
            "samples": len(X),
            "features": X.shape[1],
            "cv_scores": cv_scores,
            "avg_auc": round(np.mean([s["auc"] for s in cv_scores]), 4),
            "avg_accuracy": round(np.mean([s["accuracy"] for s in cv_scores]), 4),
            "feature_importances": self._feature_importances,
            "train_date": self._last_train_date,
        }

        # Save model
        self._save_model()

        logger.info(
            "StockRanker trained: AUC=%.3f, Acc=%.1f%%, weights=%s",
            self._train_metrics["avg_auc"],
            self._train_metrics["avg_accuracy"] * 100,
            self._feature_importances,
        )

        return self._train_metrics

    def get_factor_weights(self) -> Dict[str, float]:
        """Return learned factor weights.

        If not trained, returns equal weights as fallback.
        """
        if self._feature_importances:
            return dict(self._feature_importances)

        # Try loading saved model
        if self._load_model():
            return dict(self._feature_importances)

        # Fallback: equal weights
        n = len(FACTOR_NAMES)
        return {name: round(1.0 / n, 4) for name in FACTOR_NAMES}

    def rank_stocks(
        self,
        stocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank stocks using learned weights instead of equal weights.

        Each stock dict must have keys matching FACTOR_NAMES.
        Returns the list with added 'learned_score' and updated 'final_score'.
        """
        weights = self.get_factor_weights()

        if not stocks:
            return stocks

        # Normalise factors to [0, 1] within the candidate set
        for factor in FACTOR_NAMES:
            vals = [s.get(factor, 0) for s in stocks]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin
            for s in stocks:
                if rng > 0:
                    s[f"{factor}_norm"] = (s.get(factor, 0) - vmin) / rng
                else:
                    s[f"{factor}_norm"] = 0.5

        # Compute weighted score
        for s in stocks:
            learned_score = sum(
                weights.get(f, 0.25) * s.get(f"{f}_norm", 0.5)
                for f in FACTOR_NAMES
            )
            s["learned_score"] = round(learned_score, 4)
            s["final_score"] = s["learned_score"]
            s["factor_weights"] = weights

        stocks.sort(key=lambda x: x["final_score"], reverse=True)
        return stocks

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_training_data(
        self,
        lookback_months: int,
        forward_days: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """Build feature matrix and labels from historical data."""
        mom_days = SECTOR_SCAN_CONFIG.get("stock_momentum_days", 60)
        end_date = datetime.now()
        total_days = lookback_months * 30 + mom_days + forward_days + 60
        start_date = end_date - timedelta(days=total_days)

        # Collect all stock tickers
        all_stocks = list(set(
            tic for info in SECTOR_MAP.values() for tic in info["stocks"]
        ))

        logger.info("StockRanker: downloading data for %d stocks", len(all_stocks))

        data = yf.download(
            all_stocks,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return None, None, None

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            volume = data["Volume"]
        else:
            close = data[["Close"]].rename(columns={"Close": all_stocks[0]})
            volume = data[["Volume"]].rename(columns={"Volume": all_stocks[0]})

        # Generate monthly evaluation points
        all_features = []
        all_labels = []
        all_dates = []

        for m in range(lookback_months, 0, -1):
            eval_dt = end_date - timedelta(days=m * 30)
            ts = pd.Timestamp(eval_dt)
            if hasattr(close.index, 'tz') and close.index.tz is not None:
                ts = ts.tz_localize(close.index.tz)
            if hasattr(close.index, 'dtype'):
                try:
                    ts = ts.as_unit(close.index.dtype.str.split('[')[-1].rstrip(']'))
                except Exception:
                    pass

            idx = close.index.searchsorted(ts)
            if idx >= len(close.index) or idx < mom_days:
                continue

            eval_date = close.index[idx]
            historical = close.iloc[:idx + 1]
            historical_vol = volume.iloc[:idx + 1]

            # Future data for labelling
            future_end_idx = min(idx + forward_days + 1, len(close))
            if future_end_idx - idx < forward_days:
                continue

            for tic in all_stocks:
                if tic not in close.columns:
                    continue

                prices = historical[tic].dropna()
                vols = historical_vol[tic].dropna() if tic in historical_vol.columns else None

                if len(prices) < mom_days:
                    continue

                # Features
                momentum = float(prices.iloc[-1] / prices.iloc[-mom_days] - 1)

                if vols is not None and len(vols) >= 20:
                    current_vol = float(vols.iloc[-1])
                    vol_sma = float(vols.iloc[-20:].mean())
                    volume_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0
                else:
                    volume_ratio = 1.0

                rsi = self._compute_rsi(prices)
                rsi_score = 1.0 - abs(rsi - 50.0) / 50.0

                # No sentiment in training (too slow, and paper says
                # keyword sentiment is unreliable anyway)
                sentiment = 0.0

                # Label: forward return > median
                future_prices = close[tic].iloc[idx:future_end_idx].dropna()
                if len(future_prices) < forward_days:
                    continue

                forward_return = float(future_prices.iloc[-1] / future_prices.iloc[0] - 1)

                all_features.append([momentum, volume_ratio, rsi_score, sentiment])
                all_labels.append(forward_return)
                all_dates.append(eval_date.strftime("%Y-%m-%d"))

        if not all_features:
            return None, None, None

        X = np.array(all_features, dtype=np.float32)
        returns = np.array(all_labels, dtype=np.float32)

        # Binary label: above median return = 1
        median_return = np.median(returns)
        y = (returns > median_return).astype(np.int32)

        logger.info(
            "StockRanker: prepared %d samples, median return=%.2f%%, positive ratio=%.1f%%",
            len(X), median_return * 100, y.mean() * 100,
        )

        return X, y, all_dates

    @staticmethod
    def _compute_rsi(prices, period: int = 14) -> float:
        """Compute RSI for the last value."""
        if len(prices) < period + 1:
            return 50.0
        deltas = prices.diff().iloc[1:]
        gains = deltas.clip(lower=0)
        losses = (-deltas.clip(upper=0))
        avg_gain = float(gains.iloc[-period:].mean())
        avg_loss = float(losses.iloc[-period:].mean())
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        """Save model and metadata to disk."""
        try:
            data = {
                "model": self._model,
                "feature_importances": self._feature_importances,
                "train_date": self._last_train_date,
                "train_metrics": self._train_metrics,
            }
            with open(RANKER_MODEL_PATH, "wb") as f:
                pickle.dump(data, f)
            logger.info("StockRanker model saved to %s", RANKER_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to save ranker model: %s", exc)

    def _load_model(self) -> bool:
        """Load saved model from disk."""
        if not RANKER_MODEL_PATH.exists():
            return False
        try:
            with open(RANKER_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._feature_importances = data["feature_importances"]
            self._last_train_date = data["train_date"]
            self._train_metrics = data.get("train_metrics", {})
            logger.info("StockRanker model loaded (trained %s)", self._last_train_date)
            return True
        except Exception as exc:
            logger.error("Failed to load ranker model: %s", exc)
            return False


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ranker = StockRanker()
    metrics = ranker.train(lookback_months=12)

    if "error" in metrics:
        print(f"Training failed: {metrics['error']}")
    else:
        print(f"\n{'='*60}")
        print(f"STOCK RANKER TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Samples:          {metrics['samples']}")
        print(f"Avg AUC:          {metrics['avg_auc']:.3f}")
        print(f"Avg Accuracy:     {metrics['avg_accuracy']:.1%}")
        print(f"")
        print(f"Learned Factor Weights (vs equal 25%):")
        for name, weight in metrics["feature_importances"].items():
            diff = weight - 0.25
            arrow = "^" if diff > 0.02 else ("v" if diff < -0.02 else "=")
            print(f"  {name:20s}  {weight:.1%}  ({arrow} vs 25%)")

        print(f"\nCV Scores:")
        for i, cv in enumerate(metrics["cv_scores"]):
            print(f"  Fold {i+1}: AUC={cv['auc']:.3f}  Acc={cv['accuracy']:.1%}")
