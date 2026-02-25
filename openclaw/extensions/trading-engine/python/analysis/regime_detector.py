"""HMM-based market regime detector.

Detects market regimes (bull/bear/high-vol) using a Gaussian HMM
fitted on recent market returns and volatility.  The detected regime
is used to adjust sector scanner weights and stock selection.

Paper backing:
- arXiv 2601.19504: Regime-Adaptive Trading, HMM Sharpe 1.05,
  regime-switch alpha +1-4% annually
- arXiv 2402.05272: Jump Model Regime, +1-4% alpha on regime switch
- arXiv 2505.07078: LLM ignoring regime -> failure

States:
  0 = low-vol (normal/bull)  -> favour momentum sectors
  1 = high-vol (risk-off)    -> favour defensive sectors, reduce exposure
  2 = trending (optional)    -> if 3-state model is used

Usage::

    from regime_detector import RegimeDetector
    rd = RegimeDetector()
    regime = rd.detect()        # 0 or 1
    adj = rd.get_adjustments()  # dict of strategy adjustments
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Regime labels
REGIME_LABELS = {
    0: "low_volatility",
    1: "high_volatility",
}


class RegimeDetector:
    """Gaussian HMM market regime detector.

    Fits a 2-state HMM on SPY daily returns and realised volatility
    to classify the current market regime.
    """

    def __init__(
        self,
        benchmark: str = "SPY",
        n_states: int = 2,
        lookback_days: int = 504,  # ~2 years of trading days
        vol_window: int = 21,      # 1-month realised vol
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
        """Detect the current market regime.

        Returns a dict with:
        - regime: int (0=low_vol, 1=high_vol)
        - regime_label: str
        - confidence: float (posterior probability)
        - volatility: float (current realised vol)
        - features: dict of the feature values used
        - regime_history: list of recent regime states
        """
        features_df = self._prepare_features()
        if features_df is None or len(features_df) < 60:
            logger.warning("Insufficient data for regime detection, defaulting to low_vol")
            return self._default_result()

        model = self._fit_hmm(features_df, force_refit=force_refit)
        if model is None:
            return self._default_result()

        # Predict regime for the most recent observation
        X = features_df.values
        states = model.predict(X)
        posteriors = model.predict_proba(X)

        current_state = int(states[-1])
        current_confidence = float(posteriors[-1, current_state])

        # Ensure state 0 = low vol, state 1 = high vol
        # HMM states are arbitrary, so we map by comparing mean volatility
        current_state = self._map_states(model, features_df, states, current_state)

        # Store recent history (last 60 days)
        self._regime_history = [int(self._map_states(model, features_df, states, s)) for s in states[-60:]]

        # Current features
        last_row = features_df.iloc[-1]

        return {
            "regime": current_state,
            "regime_label": REGIME_LABELS.get(current_state, f"state_{current_state}"),
            "confidence": round(current_confidence, 4),
            "volatility": round(float(last_row.get("realised_vol", 0)), 4),
            "mean_return": round(float(last_row.get("returns", 0)), 6),
            "features": {
                "returns": round(float(last_row.get("returns", 0)), 6),
                "realised_vol": round(float(last_row.get("realised_vol", 0)), 4),
                "vol_of_vol": round(float(last_row.get("vol_of_vol", 0)), 4),
            },
            "regime_history_60d": self._regime_history,
            "regime_stability": self._compute_stability(),
            "transition_probability": self._get_transition_prob(model, current_state),
        }

    def get_adjustments(self, regime_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get strategy adjustments based on the current regime.

        Returns adjustments for:
        - momentum_weight_multiplier: scale momentum factor weight
        - defensive_bias: how much to bias toward defensive sectors
        - exposure_scale: scale overall exposure (1.0 = full, 0.5 = half)
        - preferred_sectors: list of sector types to favour
        - avoid_sectors: list of sector types to avoid
        """
        if regime_result is None:
            regime_result = self.detect()

        regime = regime_result["regime"]
        confidence = regime_result["confidence"]
        stability = regime_result.get("regime_stability", 1.0)

        if regime == 0:  # Low volatility / bull
            return {
                "regime": "low_volatility",
                "momentum_weight_multiplier": 1.0 + 0.2 * confidence,
                "defensive_bias": 0.0,
                "exposure_scale": 1.0,
                "sector_preference": ["growth", "momentum"],
                "sector_avoid": [],
                "description": "Normal/bull regime: favour momentum, full exposure",
            }
        else:  # High volatility / risk-off
            # Scale adjustments by confidence
            defensive_bias = 0.3 * confidence
            exposure_reduction = 0.3 * confidence * stability

            return {
                "regime": "high_volatility",
                "momentum_weight_multiplier": max(0.5, 1.0 - 0.3 * confidence),
                "defensive_bias": round(defensive_bias, 4),
                "exposure_scale": round(1.0 - exposure_reduction, 4),
                "sector_preference": ["defensive", "quality"],
                "sector_avoid": ["speculative", "high_beta"],
                "description": "High-vol regime: reduce exposure, favour defensives",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(self) -> Optional[pd.DataFrame]:
        """Download benchmark data and compute HMM input features."""
        end_date = datetime.now()
        # Extra buffer for vol computation
        start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))

        logger.info("Regime detector: downloading %s data", self.benchmark)

        try:
            data = yf.download(
                self.benchmark,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error("Failed to download benchmark data: %s", exc)
            if self._cached_features is not None:
                logger.warning("Using cached features (last download failed)")
                return self._cached_features
            return None

        logger.info("Regime data download: shape=%s, columns=%s", data.shape, list(data.columns))

        if data.empty:
            if self._cached_features is not None:
                logger.warning("Using cached features (empty download)")
                return self._cached_features
            return None

        # Defend against column name variants (yfinance version differences)
        if "Close" in data.columns:
            close = data["Close"]
        elif "close" in data.columns:
            close = data["close"]
        else:
            logger.warning("Unexpected column structure: %s", list(data.columns)[:10])
            if self._cached_features is not None:
                logger.warning("Using cached features (unexpected columns)")
                return self._cached_features
            return None

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        if len(close) < self.lookback_days // 2:
            if self._cached_features is not None:
                logger.warning("Using cached features (insufficient close data: %d)", len(close))
                return self._cached_features
            return None

        # Feature 1: Daily log returns
        returns = np.log(close / close.shift(1)).dropna()

        # Feature 2: Realised volatility (rolling std of returns)
        realised_vol = returns.rolling(window=self.vol_window).std() * np.sqrt(252)

        # Feature 3: Volatility of volatility (regime instability indicator)
        vol_of_vol = realised_vol.rolling(window=self.vol_window).std()

        features = pd.DataFrame({
            "returns": returns,
            "realised_vol": realised_vol,
            "vol_of_vol": vol_of_vol,
        }).dropna()

        # Trim to lookback
        if len(features) > self.lookback_days:
            features = features.iloc[-self.lookback_days:]

        self._cached_features = features
        return features

    def _fit_hmm(
        self,
        features_df: pd.DataFrame,
        force_refit: bool = False,
    ):
        """Fit or reuse the HMM model."""
        from hmmlearn.hmm import GaussianHMM

        today = datetime.now().strftime("%Y-%m-%d")

        if self._model is not None and self._last_fit_date == today and not force_refit:
            return self._model

        X = features_df.values

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                tol=0.01,
            )
            model.fit(X)
            self._model = model
            self._last_fit_date = today
            logger.info("HMM fitted with %d states on %d observations", self.n_states, len(X))
            return model

        except Exception as exc:
            logger.error("HMM fitting failed: %s", exc)
            return None

    def _map_states(
        self,
        model,
        features_df: pd.DataFrame,
        states: np.ndarray,
        current_state: int,
    ) -> int:
        """Ensure state 0 = low vol, state 1 = high vol.

        HMM assigns arbitrary state labels. We map them by comparing
        the mean realised volatility in each state.
        """
        vol_col_idx = list(features_df.columns).index("realised_vol")

        state_vols = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_vols[s] = float(features_df.iloc[mask, vol_col_idx].mean())
            else:
                state_vols[s] = 0.0

        # Sort states by volatility
        sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols[s])

        # Create mapping: lowest vol -> 0, highest vol -> 1
        state_map = {old: new for new, old in enumerate(sorted_states)}

        return state_map.get(current_state, current_state)

    def _compute_stability(self) -> float:
        """Compute regime stability (0 = unstable, 1 = stable).

        Measures how often the regime has stayed the same over
        the recent history window.
        """
        if len(self._regime_history) < 10:
            return 1.0

        recent = self._regime_history[-20:]
        if not recent:
            return 1.0

        # Count transitions
        transitions = sum(
            1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
        )
        max_transitions = len(recent) - 1
        stability = 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0
        return round(stability, 4)

    def _get_transition_prob(self, model, current_state: int) -> Dict[str, float]:
        """Get transition probabilities from the current state."""
        if model is None or not hasattr(model, "transmat_"):
            return {}

        transmat = model.transmat_
        # Map to our labels
        result = {}
        for target in range(self.n_states):
            label = REGIME_LABELS.get(target, f"state_{target}")
            prob = float(transmat[current_state, target])
            result[f"to_{label}"] = round(prob, 4)

        return result

    def _default_result(self) -> Dict[str, Any]:
        """Return a safe default when detection fails."""
        return {
            "regime": 0,
            "regime_label": "low_volatility",
            "confidence": 0.5,
            "volatility": 0.0,
            "mean_return": 0.0,
            "features": {},
            "regime_history_60d": [],
            "regime_stability": 1.0,
            "transition_probability": {},
        }


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rd = RegimeDetector()
    result = rd.detect()

    print(f"\n{'='*50}")
    print(f"MARKET REGIME DETECTION")
    print(f"{'='*50}")
    print(f"Regime:               {result['regime']} ({result['regime_label']})")
    print(f"Confidence:           {result['confidence']:.1%}")
    print(f"Realised Volatility:  {result['volatility']:.2%}")
    print(f"Mean Return:          {result['mean_return']:.4%}")
    print(f"Stability:            {result['regime_stability']:.1%}")
    print(f"Transition probs:     {result['transition_probability']}")

    adj = rd.get_adjustments(result)
    print(f"\nStrategy Adjustments:")
    print(f"  Momentum multiplier: {adj['momentum_weight_multiplier']:.2f}")
    print(f"  Defensive bias:      {adj['defensive_bias']:.2f}")
    print(f"  Exposure scale:      {adj['exposure_scale']:.2f}")
    print(f"  Prefer:              {adj['sector_preference']}")
    print(f"  Avoid:               {adj['sector_avoid']}")
    print(f"  Description:         {adj['description']}")

    # Show regime history
    history = result["regime_history_60d"]
    if history:
        low_pct = history.count(0) / len(history)
        high_pct = history.count(1) / len(history)
        print(f"\nRegime history (60d): {low_pct:.0%} low-vol, {high_pct:.0%} high-vol")
