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
    """Rolling z-score for a single feature."""

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._values: deque = deque(maxlen=window)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    def update(self, value: float) -> Optional[float]:
        """Add value, return z-score (None if < 3 samples)."""
        if len(self._values) == self._window:
            old = self._values[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

        n = len(self._values)
        if n < 3:
            return None

        mean = self._sum / n
        var = self._sum_sq / n - mean * mean
        if var <= 0:
            return 0.0
        return (value - mean) / sqrt(var)


@dataclass
class WakeCondition:
    """A condition Claude sets to be woken up early."""

    metric: str         # e.g. "btc_price", "funding_rate"
    operator: str       # gt, lt, crosses_above, crosses_below, abs_change_pct_gt
    threshold: float
    reason: str = ""

    def evaluate(self, current: Optional[float], previous: Optional[float]) -> bool:
        if current is None:
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
        max_check_seconds: float = 14400,
        min_check_seconds: float = 60.0,
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

        # Cooldown
        self._last_wake_time: float = 0.0
        # Post-decision cooldown: after Claude decides, block z-score triggers for a period
        self._post_decision_until: float = 0.0

        # Previous feature values (for crosses_above/below)
        self._prev_features: Dict[str, float] = {}

    def evaluate(
        self,
        features: Dict[str, float],
        is_candle_close: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Check all gates. Returns (should_wake, reasons)."""
        now = time.time()
        reasons: List[str] = []

        # Cooldown check — overrides everything except candle close
        in_cooldown = (now - self._last_wake_time) < self._min_check_seconds

        # Gate 1: Candle close — must also respect Claude's timer
        # Claude sets next_check_seconds; candle_close should NOT override it.
        # Only pass if Claude's timer has expired or no timer is set.
        if is_candle_close:
            timer_active = self._next_check_at > 0 and now < self._next_check_at
            if in_cooldown or timer_active:
                self._update_prev(features)
                return False, []
            reasons.append("candle_close")
            self._last_wake_time = now
            self._update_prev(features)
            return True, reasons

        if in_cooldown:
            self._update_prev(features)
            return False, []

        # Post-decision cooldown: only wake_conditions and timer can override,
        # z-score outliers are blocked until Claude's requested interval elapses
        in_post_decision = now < self._post_decision_until

        # Gate 2: Timer expired
        if self._next_check_at > 0 and now >= self._next_check_at:
            reasons.append("timer_expired")

        # Gate 3: Z-score outliers — blocked during post-decision cooldown
        for name, value in features.items():
            tracker = self._trackers.get(name)
            if tracker is None:
                tracker = ZScoreTracker(window=self._zscore_window)
                self._trackers[name] = tracker
            z = tracker.update(value)
            if not in_post_decision and z is not None and abs(z) >= self._zscore_threshold:
                reasons.append(f"zscore:{name}={z:+.2f}")

        # Gate 4: Claude's wake conditions (always active — these are Claude's own triggers)
        for cond in self._wake_conditions:
            current = features.get(cond.metric)
            previous = self._prev_features.get(cond.metric)
            if cond.evaluate(current, previous):
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
            # Block z-score triggers for half the requested interval (min 120s, max 600s)
            # Wake conditions still fire — they are Claude's own stop/scale-in triggers
            self._post_decision_until = now + max(120.0, min(clamped * 0.5, 600.0))
            self._last_wake_time = now
            logger.info("[gate] Next check in %.0fs (z-score blocked for %.0fs)",
                        clamped, self._post_decision_until - now)

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
            "zscore_blocked_for": max(0, self._post_decision_until - now),
            "wake_conditions": len(self._wake_conditions),
            "tracked_features": list(self._trackers.keys()),
            "cooldown_remaining": max(0, self._min_check_seconds - (now - self._last_wake_time)),
        }

    def _update_prev(self, features: Dict[str, float]) -> None:
        self._prev_features = dict(features)
