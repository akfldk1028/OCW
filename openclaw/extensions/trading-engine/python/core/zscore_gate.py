"""Adaptive gate — lets Claude control its own monitoring schedule.

4-layer gatekeeper between raw market ticks and Claude API calls:
    Gate 1: Candle close (always pass)
    Gate 2: Claude's timer expired (next_check_seconds)
    Gate 3: Z-score outlier (|z| >= threshold)
    Gate 4: Claude's wake conditions (metric/operator/threshold)

All gates respect min_check_seconds cooldown to prevent spam.
No external dependencies (stdlib only).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("trading-engine.gate")


class ZScoreTracker:
    """Rolling z-score using Welford's online algorithm.

    Welford avoids catastrophic floating-point cancellation that occurs
    with sum-of-squares on large values (e.g. BTC $90K+).
    """

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._values: deque = deque(maxlen=window)
        self._mean: float = 0.0
        self._M2: float = 0.0  # sum of squared deviations from mean
        self._n: int = 0

    def update(self, value: float) -> Optional[float]:
        """Add value, return z-score (None if < 3 samples)."""
        # Remove oldest if at capacity (reverse Welford step)
        if len(self._values) == self._window:
            old = self._values[0]
            self._n -= 1
            if self._n > 0:
                old_mean = self._mean
                self._mean = (old_mean * (self._n + 1) - old) / self._n
                self._M2 -= (old - old_mean) * (old - self._mean)
            else:
                self._mean = 0.0
                self._M2 = 0.0

        self._values.append(value)
        # Welford update
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2

        if self._n < 3:
            return None

        var = self._M2 / self._n if self._n > 1 else 0.0
        if var <= 0:
            return 0.0
        return (value - self._mean) / sqrt(var)


@dataclass
class WakeCondition:
    """A condition Claude sets to be woken up early.

    Once triggered, the condition is marked and will not fire again.
    Claude must send new wake_conditions to re-arm.
    """

    metric: str         # e.g. "btc_price", "funding_rate"
    operator: str       # gt, lt, crosses_above, crosses_below, abs_change_pct_gt
    threshold: float
    reason: str = ""
    triggered: bool = False  # marked True after first fire — prevents spam

    def evaluate(self, current: Optional[float], previous: Optional[float]) -> bool:
        if self.triggered or current is None:
            return False

        if self.operator == "gt":
            return current > self.threshold
        elif self.operator == "lt":
            return current < self.threshold
        elif self.operator == "crosses_above":
            if previous is None:
                return False
            return previous <= self.threshold < current
        elif self.operator == "crosses_below":
            if previous is None:
                return False
            return previous >= self.threshold > current
        elif self.operator == "abs_change_pct_gt":
            if previous is None or previous == 0:
                return False
            return abs(current - previous) / abs(previous) > self.threshold
        return False


class AdaptiveGate:
    """4-layer gatekeeper — decides whether to wake Claude."""

    def __init__(
        self,
        zscore_threshold: float = 2.5,
        zscore_window: int = 50,
        max_check_seconds: float = 7200,
        min_check_seconds: float = 120.0,
    ) -> None:
        self._zscore_threshold = zscore_threshold
        self._max_check_seconds = max_check_seconds
        self._min_check_seconds = min_check_seconds

        # Z-score trackers per feature
        self._trackers: Dict[str, ZScoreTracker] = {}
        self._zscore_window = zscore_window

        # Claude-set state (first check after max_check_seconds)
        self._next_check_at: float = time.time() + max_check_seconds
        self._wake_conditions: List[WakeCondition] = []

        # Cooldown (only rate-limiter — z-score extreme moves always get through)
        self._last_wake_time: float = 0.0

        # Previous feature values (for crosses_above/below)
        self._prev_features: Dict[str, float] = {}

    def evaluate(
        self,
        features: Dict[str, float],
        is_candle_close: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Check all gates. Returns (should_wake, reasons).

        Z-score trackers are ALWAYS updated (to keep statistics current),
        even when cooldown blocks the wake.  All gates are peers — candle
        close no longer short-circuits past z-score updates.
        """
        now = time.time()
        reasons: List[str] = []
        in_cooldown = (now - self._last_wake_time) < self._min_check_seconds

        # Always update z-score trackers so statistics stay current
        zscore_alerts: List[str] = []
        for name, value in features.items():
            tracker = self._trackers.get(name)
            if tracker is None:
                tracker = ZScoreTracker(window=self._zscore_window)
                self._trackers[name] = tracker
            z = tracker.update(value)
            if z is not None and abs(z) >= self._zscore_threshold:
                zscore_alerts.append(f"zscore:{name}={z:+.2f}")

        # Cooldown blocks all wake events
        if in_cooldown:
            self._update_prev(features)
            return False, []

        # Gate 1: Candle close
        if is_candle_close:
            reasons.append("candle_close")

        # Gate 2: Timer expired (routine check)
        if self._next_check_at > 0 and now >= self._next_check_at:
            reasons.append("timer_expired")

        # Gate 3: Z-score outlier alerts
        reasons.extend(zscore_alerts)

        # Gate 4: Claude's wake conditions (single-fire)
        for cond in self._wake_conditions:
            current = features.get(cond.metric)
            previous = self._prev_features.get(cond.metric)
            if cond.evaluate(current, previous):
                cond.triggered = True
                label = cond.reason or f"{cond.metric} {cond.operator} {cond.threshold}"
                reasons.append(f"wake_cond:{label}")

        self._update_prev(features)

        if reasons:
            self._last_wake_time = now
            logger.info("[gate] WAKE: %s", ", ".join(reasons))
            return True, reasons

        return False, []

    def update_from_claude(
        self,
        next_check_seconds: Optional[float] = None,
        wake_conditions: Optional[List[Dict]] = None,
    ) -> None:
        """Apply Claude's scheduling decisions."""
        now = time.time()

        if next_check_seconds is not None:
            clamped = max(
                self._min_check_seconds,
                min(next_check_seconds, self._max_check_seconds),
            )
            self._next_check_at = now + clamped
            self._last_wake_time = now
            # z-score is NEVER blocked — extreme moves always trigger
            # Only the 60s cooldown rate-limits calls
            logger.info("[gate] Next routine check in %.0fs (z-score always active, cooldown=%.0fs)",
                        clamped, self._min_check_seconds)

        if wake_conditions is not None:
            self._wake_conditions = []
            for wc in wake_conditions:
                try:
                    self._wake_conditions.append(WakeCondition(
                        metric=wc["metric"],
                        operator=wc["operator"],
                        threshold=float(wc["threshold"]),
                        reason=wc.get("reason", ""),
                    ))
                except (KeyError, ValueError) as exc:
                    logger.warning("[gate] Invalid wake condition %s: %s", wc, exc)
            if self._wake_conditions:
                logger.info("[gate] %d wake conditions set", len(self._wake_conditions))

    def get_status(self) -> Dict:
        """Monitoring/debugging info."""
        now = time.time()
        return {
            "next_check_in": max(0, self._next_check_at - now) if self._next_check_at > 0 else None,
            "zscore_always_active": True,
            "wake_conditions": len(self._wake_conditions),
            "tracked_features": list(self._trackers.keys()),
            "cooldown_remaining": max(0, self._min_check_seconds - (now - self._last_wake_time)),
        }

    def _update_prev(self, features: Dict[str, float]) -> None:
        self._prev_features = dict(features)
