"""Out-of-Distribution (OOD) detector using Mahalanobis distance.

Simplified from arXiv:2512.23773 (FineFT, KDD 2026) — VAE replaced with
Mahalanobis distance for zero-training-required OOD detection. Works from
day 1 with rolling statistics.

Detects market states that are statistically unusual compared to recent
history. When OOD is detected, the information is passed to Claude as
context (not as a hard trading constraint — RL decides how to use it).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("trading-engine.ood")


class OODDetector:
    """Rolling Mahalanobis distance OOD detector.

    Maintains a rolling window of feature vectors and computes the
    Mahalanobis distance of new observations from the empirical
    distribution. Normal market: distance 2-4. OOD: distance > 6.

    Args:
        window: Number of recent observations to keep.
        min_samples: Minimum observations before scoring is possible.
        recompute_interval: Recompute mean/covariance every N updates.
        threshold: Mahalanobis distance above which a sample is OOD.
    """

    def __init__(
        self,
        window: int = 200,
        min_samples: int = 30,
        recompute_interval: int = 10,
        threshold: float = 6.0,
    ) -> None:
        self._window = window
        self._min_samples = min_samples
        self._recompute_interval = recompute_interval
        self._threshold = threshold

        self._observations: deque = deque(maxlen=window)
        self._feature_names: list = []
        self._mean: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._inv_cov_diag: Optional[np.ndarray] = None
        self._update_count: int = 0

    def update(self, features: Dict[str, float]) -> None:
        """Add a new observation to the rolling window.

        Args:
            features: {signal_name: value} dict from _extract_ta_signals().
        """
        if not features:
            return

        # On first call, lock the feature set from the first observation.
        # After that, ignore new keys — just fill missing with 0.0.
        # This prevents constant window resets from conditional signals.
        if not self._feature_names:
            self._feature_names = sorted(features.keys())

        # Build vector in consistent order, filling missing with 0.0
        vec = [features.get(name, 0.0) for name in self._feature_names]
        self._observations.append(vec)
        self._update_count += 1

        # Recompute statistics periodically
        if (self._update_count % self._recompute_interval == 0
                and len(self._observations) >= self._min_samples):
            self._recompute()

    def score(self, features: Dict[str, float]) -> Optional[float]:
        """Compute Mahalanobis distance for a feature vector.

        Returns:
            Mahalanobis distance (float), or None if insufficient data.
            Normal market: 2-4. Unusual: 4-6. OOD: >6.
        """
        if self._mean is None or self._inv_cov is None:
            return None

        vec = np.array([features.get(name, 0.0) for name in self._feature_names])
        diff = vec - self._mean
        # Mahalanobis: sqrt(diff^T @ inv_cov @ diff)
        mahal_sq = diff @ self._inv_cov @ diff
        if mahal_sq < 0:
            return 0.0  # numerical artifact
        # Cap at 50 — anything above is equally "extreme OOD"
        return min(50.0, float(np.sqrt(mahal_sq)))

    def score_decomposed(self, features: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
        """Kinlaw-Turkington turbulence decomposition.

        Decomposes Mahalanobis distance into magnitude surprise (large moves)
        and correlation surprise (structural breaks). Magnitude-only = opportunity,
        correlation break = danger.

        Returns: (total, magnitude_surprise, correlation_surprise) or None.
        """
        if self._mean is None or self._inv_cov is None or self._inv_cov_diag is None:
            return None
        vec = np.array([features.get(name, 0.0) for name in self._feature_names])
        diff = vec - self._mean
        total = min(50.0, float(np.sqrt(max(0.0, diff @ self._inv_cov @ diff))))
        magnitude = min(50.0, float(np.sqrt(max(0.0, diff @ self._inv_cov_diag @ diff))))
        correlation = total / magnitude if magnitude > 0.01 else 1.0
        return (total, magnitude, correlation)

    def is_ood(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Check if features are OOD.

        Uses Kinlaw-Turkington decomposition when available:
        OOD = correlation break (structural danger), NOT magnitude-only (big move).

        Returns:
            (is_ood: bool, distance: float). Distance is 0.0 if
            insufficient data for scoring.
        """
        decomposed = self.score_decomposed(features)
        if decomposed is None:
            dist = self.score(features)
            if dist is None:
                return False, 0.0
            return dist > self._threshold, dist
        total, magnitude, correlation = decomposed
        # OOD = correlation break (structural risk). Magnitude-only (big move) is NOT OOD
        is_dangerous = correlation > 1.5 and total > self._threshold
        return is_dangerous, total

    def _recompute(self) -> None:
        """Recompute mean and inverse covariance from rolling window."""
        data = np.array(list(self._observations))  # (N, D)
        n, d = data.shape
        if n < self._min_samples or d == 0:
            return

        self._mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)

        # Handle 1D case (single feature)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Regularize for numerical stability — scale with median variance
        # to handle zero-variance features (fixed signals like ±0.5)
        med_var = max(np.median(np.diag(cov)), 1e-4)
        cov += np.eye(d) * med_var * 0.01

        # Kinlaw & Turkington (2013) — diagonal-only cov for magnitude surprise
        diag_cov = np.diag(np.diag(cov))
        try:
            self._inv_cov_diag = np.linalg.inv(diag_cov)
        except np.linalg.LinAlgError:
            self._inv_cov_diag = np.linalg.pinv(diag_cov)

        try:
            self._inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular matrix — use pseudo-inverse
            self._inv_cov = np.linalg.pinv(cov)
            self._inv_cov_diag = np.linalg.pinv(diag_cov)
            logger.debug("[ood] Covariance singular, using pseudo-inverse")

    def get_status(self) -> Dict:
        """Monitoring info."""
        return {
            "observations": len(self._observations),
            "min_samples": self._min_samples,
            "feature_count": len(self._feature_names),
            "ready": self._mean is not None,
        }
