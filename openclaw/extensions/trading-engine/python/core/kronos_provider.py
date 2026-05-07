"""KronosProvider — Kronos Foundation Model integration for H-TS prediction signals.

Lazy-loads Kronos-mini (4.1M params) from HuggingFace on first prediction request.
Converts OHLCV predictions into 4 normalized signals for the "prediction" group.

Design:
    - Lazy loading: no startup penalty
    - 15-min TTL cache: avoids redundant inference
    - Graceful degradation: failures return empty dict → existing 32 signals work fine
    - No hardcoding: signals are [-1, 1] normalized, H-TS learns the weights
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading-engine.kronos")

# Model identifiers
_MODEL_REPO = "NeoQuasar/Kronos-mini"
_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-2k"
_PRED_LEN = 12       # 12 × 1h = 12h ahead
_SAMPLE_COUNT = 5    # samples for uncertainty estimation (5 × full inference)
_MAX_CONTEXT = 512   # Kronos-mini context window
_CACHE_TTL = 900.0   # 15 minutes


class KronosProvider:
    """Provides Kronos AI prediction signals for H-TS integration."""

    def __init__(self) -> None:
        self._predictor = None  # lazy loaded
        self._permanent_error: Optional[str] = None  # ImportError 등 복구 불가
        self._last_load_attempt: float = 0.0
        self._load_retry_interval: float = 300.0  # 5분 후 재시도 (네트워크 등)
        self._cache: Dict[str, Dict] = {}  # {ticker: {ts, signals, raw}}

    def _load_model(self) -> bool:
        """Lazy-load Kronos-mini from HuggingFace. Returns True on success."""
        if self._predictor is not None:
            return True
        if self._permanent_error is not None:
            return False  # ImportError 등 → 영구 비활성

        # 네트워크 에러 등은 5분 후 재시도
        now = time.time()
        if self._last_load_attempt > 0 and (now - self._last_load_attempt) < self._load_retry_interval:
            return False
        self._last_load_attempt = now

        try:
            import torch
            from core.kronos import KronosTokenizer, Kronos, KronosPredictor

            logger.info("[kronos] Loading Kronos-mini from HuggingFace...")
            tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_REPO)
            model = Kronos.from_pretrained(_MODEL_REPO)

            # Auto-detect device
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._predictor = KronosPredictor(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_context=_MAX_CONTEXT,
            )
            logger.info("[kronos] Model loaded on %s", device)
            return True

        except ImportError as e:
            self._permanent_error = f"Missing dependency: {e}"
            logger.warning("[kronos] %s — prediction signals permanently disabled", self._permanent_error)
            return False
        except Exception as e:
            logger.warning("[kronos] Load failed (will retry in 5min): %s", e)
            return False

    def get_signals(
        self,
        ticker: str,
        ohlcv_bars: Any,
        current_trend: str = "",
    ) -> Dict[str, float]:
        """Get 4 Kronos prediction signals for a ticker.

        Args:
            ticker: e.g. "BTC/USDT"
            ohlcv_bars: Either a DataFrame with [open, high, low, close, volume]
                        and DatetimeIndex, or a list of (ts_ms, o, h, l, c, v) tuples
                        from OHLCVStore.get_bars(). At least 50 bars of 1h data.
            current_trend: TA-based trend direction ("bullish"/"bearish"/"neutral")

        Returns:
            Dict with keys: kronos_direction, kronos_magnitude,
            kronos_confidence, kronos_trend_alignment.
            Empty dict on failure.
        """
        # Cache check
        cached = self._cache.get(ticker)
        if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
            return cached["signals"]

        # Validate input
        if ohlcv_bars is None or len(ohlcv_bars) < 50:
            return {}

        # Convert raw tuples to DataFrame if needed
        if isinstance(ohlcv_bars, list):
            ohlcv_bars = self._tuples_to_df(ohlcv_bars)
            if ohlcv_bars is None or len(ohlcv_bars) < 50:
                return {}

        # Lazy load
        if not self._load_model():
            return {}

        try:
            predictions = self._predict(ohlcv_bars)
            if predictions is None:
                return {}

            signals = self._to_signals(predictions, ohlcv_bars, current_trend)

            # Cache
            self._cache[ticker] = {
                "ts": time.time(),
                "signals": signals,
                "raw": predictions,
            }
            return signals

        except Exception as e:
            logger.warning("[kronos] Prediction failed for %s: %s", ticker, e)
            return {}

    def get_raw_predictions(self, ticker: str) -> Optional[Dict]:
        """Get cached raw predictions for prompt display."""
        cached = self._cache.get(ticker)
        if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
            return cached.get("raw")
        return None

    def invalidate_cache(self, ticker: str) -> None:
        """Invalidate cache for a ticker (e.g. on significant move)."""
        self._cache.pop(ticker, None)

    @staticmethod
    def _tuples_to_df(bars: List) -> Optional[pd.DataFrame]:
        """Convert OHLCVStore tuples [(ts_ms, o, h, l, c, v), ...] to DataFrame."""
        if not bars:
            return None
        try:
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            return None

    def _predict(self, ohlcv_bars: pd.DataFrame) -> Optional[Dict]:
        """Run Kronos multi-sample prediction, return summary stats."""
        try:
            df = ohlcv_bars.copy()

            # Ensure required columns
            required = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                return None

            if "volume" not in df.columns:
                df["volume"] = 0.0

            # Timestamps
            if isinstance(df.index, pd.DatetimeIndex):
                x_timestamp = df.index
            else:
                return None

            # Future timestamps: 12 × 1h
            last_ts = x_timestamp[-1]
            y_timestamp = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1),
                periods=_PRED_LEN,
                freq="1h",
            )

            # Run multiple samples for uncertainty estimation
            current_close = float(df["close"].iloc[-1])
            all_pred_closes: List[np.ndarray] = []

            for _ in range(_SAMPLE_COUNT):
                pred_df = self._predictor.predict(
                    df=df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=_PRED_LEN,
                    T=1.0,
                    top_k=0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False,
                )
                all_pred_closes.append(pred_df["close"].values)

            pred_matrix = np.stack(all_pred_closes, axis=0)  # (samples, pred_len)

            # Compute returns relative to current close
            returns_matrix = (pred_matrix - current_close) / current_close  # (samples, pred_len)

            # Time-weighted average return (closer predictions weighted more)
            weights = np.exp(-0.1 * np.arange(_PRED_LEN))  # exponential decay
            weights /= weights.sum()

            # Per-sample weighted return
            sample_returns = (returns_matrix * weights[None, :]).sum(axis=1)  # (samples,)
            mean_return = float(np.mean(sample_returns))
            std_return = float(np.std(sample_returns))

            return {
                "mean_return": mean_return,
                "std_return": std_return,
                "sample_returns": sample_returns.tolist(),
                "current_close": current_close,
                "pred_closes_mean": float(np.mean(pred_matrix[:, -1])),
                "pred_horizon_h": _PRED_LEN,
            }

        except Exception as e:
            logger.warning("[kronos] Inference error: %s", e)
            return None

    def _to_signals(
        self,
        predictions: Dict,
        ohlcv_bars: pd.DataFrame,
        current_trend: str,
    ) -> Dict[str, float]:
        """Convert raw predictions to 4 normalized signals."""
        mean_ret = predictions["mean_return"]
        std_ret = predictions["std_return"]

        # 1. kronos_direction: weighted-average predicted return / 0.02
        #    +1 = strongly bullish, -1 = strongly bearish
        direction = max(-1.0, min(1.0, mean_ret / 0.02))

        # 2. kronos_magnitude: absolute predicted move / 0.03
        #    0 = flat, 1 = big move expected
        magnitude = min(1.0, abs(mean_ret) / 0.03)

        # 3. kronos_confidence: sample agreement
        #    Uses normalized std: std/0.01 (1% move = high disagreement)
        #    CV is unstable near zero mean, so use absolute std instead
        confidence = max(0.0, min(1.0, 1.0 - std_ret / 0.01))

        # 4. kronos_trend_alignment: direction × TA trend sign
        #    +1 = prediction aligns with TA, -1 = divergent
        if "bull" in current_trend.lower():
            ta_sign = 1.0
        elif "bear" in current_trend.lower():
            ta_sign = -1.0
        else:
            ta_sign = 0.0

        if ta_sign != 0.0:
            alignment = max(-1.0, min(1.0, direction * ta_sign))
        else:
            alignment = 0.0  # no TA trend → neutral alignment

        return {
            "kronos_direction": round(direction, 4),
            "kronos_magnitude": round(magnitude, 4),
            "kronos_confidence": round(confidence, 4),
            "kronos_trend_alignment": round(alignment, 4),
        }
