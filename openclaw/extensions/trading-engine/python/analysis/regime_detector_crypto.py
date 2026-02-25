"""HMM-based crypto market regime detector.

Detects crypto market regimes (low-vol / high-vol) using a Gaussian HMM
fitted on BTC daily returns and realised volatility.  BTC serves as the
market proxy for crypto the same way SPY does for equities.

Key differences from equity RegimeDetector:
    - Benchmark: BTC-USD (via yfinance) instead of SPY
    - Lookback: 365 calendar days (~365 trading days, crypto is 24/7)
    - Volatility thresholds are much higher (crypto vol ~40-100%+ annualised)
    - High-vol exposure scale: 0.6 (vs ~0.7-0.8 for equities)
    - Vol window: 14 days (faster-moving crypto markets)

Usage::

    from regime_detector_crypto import CryptoRegimeDetector
    rd = CryptoRegimeDetector()
    result = rd.detect()
    adj = rd.get_adjustments(result)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "low_volatility",
    1: "medium_volatility",
    2: "high_volatility",
}
# BIC search range for optimal state count (arXiv:2602.03874 recommends 3+)
_BIC_SEARCH_RANGE = [2, 3, 4]


class CryptoRegimeDetector:
    """Gaussian HMM crypto regime detector using BTC as market proxy."""

    def __init__(
        self,
        benchmark: str = "BTC-USD",
        n_states: int = 2,
        lookback_days: int = 365,
        vol_window: int = 14,
    ) -> None:
        self.benchmark = benchmark
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.vol_window = vol_window

        self._model = None
        self._last_fit_date: Optional[str] = None
        self._regime_history: List[int] = []
        self._cached_features: Optional[pd.DataFrame] = None

    def detect(self, force_refit: bool = False) -> Dict[str, Any]:
        """Detect crypto market regime via BTC volatility HMM.

        Returns dict with: regime, regime_label, confidence, volatility,
        exposure_scale, features, regime_history_60d, regime_stability,
        transition_probability.
        """
        features_df = self._prepare_features()
        if features_df is None or len(features_df) < 30:
            logger.warning("Insufficient BTC data for regime detection, defaulting to low_vol")
            return self._default_result()

        model = self._fit_hmm(features_df, force_refit=force_refit)
        if model is None:
            return self._default_result()

        X = features_df.values
        states = model.predict(X)
        posteriors = model.predict_proba(X)

        current_state = int(states[-1])
        current_confidence = float(posteriors[-1, current_state])

        # Map states so 0=low_vol, 1=high_vol
        current_state = self._map_states(features_df, states, current_state)
        self._regime_history = [
            self._map_states(features_df, states, int(s)) for s in states[-60:]
        ]

        last_row = features_df.iloc[-1]
        vol = float(last_row.get("realised_vol", 0))

        # Exposure scale: low_vol=1.0, medium_vol=0.8, high_vol=0.6
        _EXPOSURE_MAP = {0: 1.0, 1: 0.8, 2: 0.6}
        exposure_scale = _EXPOSURE_MAP.get(current_state, 0.6)

        return {
            "regime": current_state,
            "regime_label": REGIME_LABELS.get(current_state, f"state_{current_state}"),
            "confidence": round(current_confidence, 4),
            "volatility": round(vol, 4),
            "exposure_scale": round(exposure_scale, 4),
            "features": {
                "returns": round(float(last_row.get("returns", 0)), 6),
                "realised_vol": round(vol, 4),
                "vol_of_vol": round(float(last_row.get("vol_of_vol", 0)), 4),
            },
            "regime_history_60d": self._regime_history,
            "regime_stability": self._compute_stability(),
            "transition_probability": self._get_transition_prob(model, current_state),
        }

    def get_adjustments(self, regime_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get risk adjustments for crypto regime."""
        if regime_result is None:
            regime_result = self.detect()

        regime = regime_result["regime"]
        confidence = regime_result["confidence"]

        if regime == 0:
            return {
                "regime": "low_volatility",
                "exposure_scale": 1.0,
                "momentum_weight_multiplier": 1.0 + 0.2 * confidence,
                "prefer_alts": True,
                "description": "Low-vol crypto regime: full exposure, favour momentum alts",
            }
        elif regime == 1:
            return {
                "regime": "medium_volatility",
                "exposure_scale": 0.8,
                "momentum_weight_multiplier": max(0.6, 1.0 - 0.2 * confidence),
                "prefer_alts": True,
                "description": "Medium-vol regime: 80% exposure, balanced approach",
            }
        else:
            exposure_scale = max(0.4, 0.6 * (1.0 + 0.1 * (1.0 - confidence)))
            return {
                "regime": "high_volatility",
                "exposure_scale": round(exposure_scale, 4),
                "momentum_weight_multiplier": max(0.4, 1.0 - 0.4 * confidence),
                "prefer_alts": False,
                "description": "High-vol crypto regime: reduce to 60% exposure, favour BTC/ETH",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(self) -> Optional[pd.DataFrame]:
        """Download BTC data and compute HMM input features."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))

        logger.info("CryptoRegimeDetector: downloading %s data", self.benchmark)
        try:
            data = yf.download(
                self.benchmark,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error("Failed to download BTC data: %s", exc)
            return self._cached_features

        if data.empty:
            return self._cached_features

        close = data.get("Close", data.get("close"))
        if close is None:
            return self._cached_features
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < self.lookback_days // 2:
            return self._cached_features

        returns = np.log(close / close.shift(1)).dropna()
        # Annualise with 365 (crypto trades every day)
        realised_vol = returns.rolling(window=self.vol_window).std() * np.sqrt(365)
        vol_of_vol = realised_vol.rolling(window=self.vol_window).std()

        features = pd.DataFrame({
            "returns": returns,
            "realised_vol": realised_vol,
            "vol_of_vol": vol_of_vol,
        }).dropna()

        if len(features) > self.lookback_days:
            features = features.iloc[-self.lookback_days:]

        self._cached_features = features
        return features

    def _fit_hmm(self, features_df: pd.DataFrame, force_refit: bool = False):
        from hmmlearn.hmm import GaussianHMM

        today = datetime.now().strftime("%Y-%m-%d")
        if self._model is not None and self._last_fit_date == today and not force_refit:
            return self._model

        X = features_df.values

        # BIC-based automatic state selection (arXiv:2602.03874)
        best_model = None
        best_bic = float("inf")
        best_n = self.n_states
        for n in _BIC_SEARCH_RANGE:
            try:
                m = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                    tol=0.01,
                )
                m.fit(X)
                # BIC = -2 * log_likelihood + k * log(n_samples)
                ll = m.score(X) * len(X)
                k = n * n + n * X.shape[1] * 2 + n - 1  # params: transmat + means + covars
                bic = -2 * ll + k * np.log(len(X))
                logger.debug("HMM BIC search: n_states=%d, BIC=%.1f, LL=%.1f", n, bic, ll)
                if bic < best_bic:
                    best_bic = bic
                    best_model = m
                    best_n = n
            except Exception:
                continue

        if best_model is None:
            logger.error("All HMM fits failed")
            return None

        self.n_states = best_n
        self._model = best_model
        self._last_fit_date = today
        logger.info("Crypto HMM fitted: %d states (BIC=%.1f) on %d observations",
                     best_n, best_bic, len(features_df))
        return best_model

    def _map_states(self, features_df: pd.DataFrame, states: np.ndarray, current_state: int) -> int:
        """Ensure 0=low_vol, 1=high_vol by comparing mean realised_vol per state."""
        vol_idx = list(features_df.columns).index("realised_vol")
        state_vols = {}
        for s in range(self.n_states):
            mask = states == s
            state_vols[s] = float(features_df.iloc[mask, vol_idx].mean()) if mask.sum() > 0 else 0.0
        sorted_states = sorted(state_vols, key=lambda s: state_vols[s])
        state_map = {old: new for new, old in enumerate(sorted_states)}
        return state_map.get(current_state, current_state)

    def _compute_stability(self) -> float:
        if len(self._regime_history) < 10:
            return 1.0
        recent = self._regime_history[-20:]
        transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        max_t = len(recent) - 1
        return round(1.0 - transitions / max_t, 4) if max_t > 0 else 1.0

    def _get_transition_prob(self, model, current_state: int) -> Dict[str, float]:
        if model is None or not hasattr(model, "transmat_"):
            return {}
        return {
            f"to_{REGIME_LABELS.get(t, f'state_{t}')}": round(float(model.transmat_[current_state, t]), 4)
            for t in range(self.n_states)
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "regime": 0,
            "regime_label": "low_volatility",
            "confidence": 0.5,
            "volatility": 0.0,
            "exposure_scale": 1.0,
            "features": {},
            "regime_history_60d": [],
            "regime_stability": 1.0,
            "transition_probability": {},
        }
