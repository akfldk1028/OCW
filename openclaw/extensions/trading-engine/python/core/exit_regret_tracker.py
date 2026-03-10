"""Post-exit counterfactual tracker — "exit regret" learning.

Mirrors the entry counterfactual pattern (runner._check_counterfactuals)
but tracks price AFTER an exit to learn whether we exited too early
(premature) or at the right time (validated / good exit).

Horizons are shorter than entry CF because exit effects manifest faster.
Discounts are lower (0.10-0.15 vs 0.20-0.30) because post-exit price
moves may reflect new information rather than our exit being wrong.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class ExitSnapshot:
    """Snapshot of a closed position for post-exit tracking."""

    ts: float
    ticker: str
    exit_price: float
    entry_price: float
    pnl_pct: float
    held_hours: float
    position_side: str  # "long" | "short"
    agent_signals: Dict[str, float]
    regime: str
    exit_reason: str


class ExitRegretTracker:
    """Track price after exits to learn hold_patience and risk_aversion.

    Three horizons (mirroring entry CF pattern):
      - quick:    5-15 min  (noise check, high discount)
      - medium:   30min-2h  (trend continuation)
      - extended: 2-8h      (major move)

    Callbacks:
      - on_regret(snap, horizon, price_now, additional_pnl):
            called when exiting too early cost us > threshold
      - on_validated(snap, horizon, price_now, additional_pnl):
            called when exit was correct (price moved against us)
    """

    HORIZONS = [
        # (min_age_s, max_age_s, discount, label)
        (300,   900,   0.15, "quick"),     # 5-15 min
        (1800,  7200,  0.12, "medium"),    # 30min-2h
        (7200,  28800, 0.10, "extended"),  # 2-8h
    ]

    # Minimum additional PnL to trigger learning (avoids noise)
    REGRET_THRESHOLD = 0.003   # +0.3% additional we missed
    VALIDATE_THRESHOLD = 0.003  # -0.3% we avoided by exiting

    def __init__(self) -> None:
        self._quick: deque[ExitSnapshot] = deque(maxlen=30)
        self._medium: deque[ExitSnapshot] = deque(maxlen=20)
        self._extended: deque[ExitSnapshot] = deque(maxlen=15)
        self._deques = [self._quick, self._medium, self._extended]

    def record_exit(self, snapshot: ExitSnapshot) -> None:
        """Store an exit snapshot for all three horizon queues."""
        for dq in self._deques:
            dq.append(snapshot)

    def check_exits(
        self,
        price_fetcher: Callable[[str], float],
        on_regret: Callable[[ExitSnapshot, str, float, float, float], None],
        on_validated: Callable[[ExitSnapshot, str, float, float, float], None],
    ) -> None:
        """Check all pending exit snapshots against current prices.

        Args:
            price_fetcher: ticker -> current_price (0 if unknown)
            on_regret: callback(snap, horizon, price_now, additional_pnl, discount)
            on_validated: callback(snap, horizon, price_now, additional_pnl, discount)
        """
        now = time.time()

        for dq, (min_age, max_age, discount, label) in zip(self._deques, self.HORIZONS):
            to_remove: List[ExitSnapshot] = []

            for snap in dq:
                age = now - snap.ts
                if age < min_age:
                    continue  # too recent
                if age > max_age:
                    to_remove.append(snap)
                    continue  # expired

                current_price = price_fetcher(snap.ticker)
                if current_price <= 0:
                    continue

                # Calculate hypothetical PnL if we had held
                if snap.position_side == "short":
                    hypothetical_pnl = (snap.entry_price - current_price) / snap.entry_price
                else:  # long
                    hypothetical_pnl = (current_price - snap.entry_price) / snap.entry_price

                additional_pnl = hypothetical_pnl - snap.pnl_pct

                if additional_pnl > self.REGRET_THRESHOLD:
                    # Price continued in our favor — premature exit
                    try:
                        on_regret(snap, label, current_price, additional_pnl, discount)
                    except Exception as exc:
                        logger.debug("[exit-regret] on_regret error: %s", exc)
                    to_remove.append(snap)

                elif additional_pnl < -self.VALIDATE_THRESHOLD:
                    # Price reversed — good exit
                    try:
                        on_validated(snap, label, current_price, additional_pnl, discount)
                    except Exception as exc:
                        logger.debug("[exit-regret] on_validated error: %s", exc)
                    to_remove.append(snap)

            # Remove processed snapshots
            for snap in to_remove:
                try:
                    dq.remove(snap)
                except ValueError:
                    pass

    @property
    def pending_count(self) -> int:
        return sum(len(dq) for dq in self._deques)
