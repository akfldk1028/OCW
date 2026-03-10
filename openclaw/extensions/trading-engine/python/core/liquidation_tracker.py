"""Real-time liquidation monitoring + cluster estimation for H-TS integration.

Subscribes to WS forceOrder events via EventBus.
Tracks directional liquidations (long/short) in a rolling 1h window.
Estimates leverage-based liquidation clusters from OI data.
Generates `liquidation_level` signal [-1.0, +1.0] for H-TS Level 2.

Signal interpretation:
  +1.0 = short liquidation cluster above current price (price magnet UP)
  -1.0 = long liquidation cluster below current price (price magnet DOWN)
   0.0 = no significant clusters nearby

References:
  - CryptoTrade (EMNLP 2024): derivatives microstructure data fusion
  - HedgeAgents (2025): counter-signal (liquidation cluster = directional magnet)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("trading-engine.liq_tracker")

# Estimated leverage distribution across open interest.
# Based on Binance aggregate data: most OI is 5-20x, tail at 50-100x.
LEVERAGE_DIST: List[Tuple[int, float]] = [
    (5,   0.25),  # 5x: 25% of OI
    (10,  0.30),  # 10x: 30%
    (20,  0.20),  # 20x: 20%
    (50,  0.15),  # 50x: 15%
    (100, 0.10),  # 100x: 10%
]

# Rolling window for real-time liquidation events
_WINDOW_SECONDS = 3600  # 1 hour
_RECENT_SECONDS = 900   # 15 minutes (momentum calculation)


class LiquidationTracker:
    """Real-time liquidation tracking + cluster estimation.

    Lifecycle:
        1. Runner creates instance + subscribes on_force_order to EventBus
        2. WS forceOrder events flow in via EventBus
        3. Runner calls get_context() per ticker when building MarketSnapshot
        4. Signal feeds into H-TS `liquidation_level` (constants.py line 43)
    """

    def __init__(self, tickers: List[str]) -> None:
        self._tickers = set(tickers)

        # Per-ticker rolling windows: deque of (timestamp, usd_value, price)
        self._long_liqs: Dict[str, deque] = {t: deque(maxlen=5000) for t in tickers}
        self._short_liqs: Dict[str, deque] = {t: deque(maxlen=5000) for t in tickers}

        # Track total events received (for diagnostics)
        self._total_events = 0
        self._last_event_time: float = 0.0

        logger.info("[liq_tracker] Initialized for %d tickers", len(tickers))

    # ------------------------------------------------------------------
    # Event handler (subscribed to market.force_order)
    # ------------------------------------------------------------------

    async def on_force_order(self, event) -> None:
        """Handle a real-time forceOrder event from MarketListener."""
        data = event.payload
        ticker = data.get("ticker", "")
        if ticker not in self._tickers:
            return

        side = data.get("side", "")
        price = data.get("price", 0)
        usd = data.get("usd_value", 0)
        ts = data.get("timestamp", time.time())

        if usd <= 0:
            return

        entry = (ts, usd, price)
        if side == "SELL":
            # SELL order = long position being liquidated
            self._long_liqs[ticker].append(entry)
        elif side == "BUY":
            # BUY order = short position being liquidated
            self._short_liqs[ticker].append(entry)

        self._total_events += 1
        self._last_event_time = ts

        if self._total_events % 50 == 1:
            logger.info(
                "[liq_tracker] Event #%d: %s %s liq $%.0f @ %.2f",
                self._total_events, ticker,
                "long" if side == "SELL" else "short", usd, price,
            )

    # ------------------------------------------------------------------
    # Cluster estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_clusters(
        current_price: float, oi_usd: float, funding_rate: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Estimate liquidation price clusters from OI and leverage distribution.

        Uses funding rate to split OI between long/short sides:
        - Positive funding → more longs → heavier long clusters below
        - Negative funding → more shorts → heavier short clusters above
        - Neutral → 50/50 split (symmetric, proximity ~0)

        Returns:
            (long_clusters, short_clusters) — each is [(price, estimated_usd), ...]
            long_clusters: prices where long positions get liquidated (below current)
            short_clusters: prices where short positions get liquidated (above current)
        """
        if current_price <= 0 or oi_usd <= 0:
            return [], []

        # Split OI by funding rate direction
        # funding > 0 → longs pay shorts → more longs exist
        # Clamp bias to [0.3, 0.7] so neither side goes to zero
        funding_skew = max(-0.2, min(0.2, funding_rate * 200))  # 0.001 → 0.2
        long_fraction = 0.5 + funding_skew   # [0.3, 0.7]
        short_fraction = 1.0 - long_fraction  # [0.3, 0.7]

        long_clusters = []
        short_clusters = []

        for leverage, weight in LEVERAGE_DIST:
            long_usd = oi_usd * weight * long_fraction
            short_usd = oi_usd * weight * short_fraction
            # Long liquidation price: price * (1 - 1/leverage)
            long_liq_price = current_price * (1 - 1 / leverage)
            # Short liquidation price: price * (1 + 1/leverage)
            short_liq_price = current_price * (1 + 1 / leverage)

            long_clusters.append((long_liq_price, long_usd))
            short_clusters.append((short_liq_price, short_usd))

        return long_clusters, short_clusters

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _clean_window(self, ticker: str) -> None:
        """Remove entries older than _WINDOW_SECONDS."""
        cutoff = time.time() - _WINDOW_SECONDS
        for dq in (self._long_liqs.get(ticker, deque()),
                    self._short_liqs.get(ticker, deque())):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def _sum_usd(self, dq: deque, since: float = 0) -> float:
        """Sum USD values in deque, optionally only entries after `since`."""
        return sum(usd for ts, usd, _ in dq if ts >= since)

    def get_signal(
        self,
        ticker: str,
        current_price: float,
        oi_usd: float,
        funding_rate: float,
        _already_cleaned: bool = False,
    ) -> float:
        """Compute liquidation_level signal [-1.0, +1.0].

        Components:
          1. Liquidation momentum (recent 15min vs prior 45min direction)
          2. Cluster proximity (nearest cluster = stronger magnet)
          3. Funding rate bias (crowding direction)
        """
        if not _already_cleaned:
            self._clean_window(ticker)

        now = time.time()
        recent_cutoff = now - _RECENT_SECONDS

        long_dq = self._long_liqs.get(ticker, deque())
        short_dq = self._short_liqs.get(ticker, deque())

        # --- 1. Liquidation momentum ---
        recent_long_usd = self._sum_usd(long_dq, recent_cutoff)
        recent_short_usd = self._sum_usd(short_dq, recent_cutoff)
        total_recent = recent_long_usd + recent_short_usd

        # Short liquidations > long liquidations → upward pressure → positive
        if total_recent > 0:
            momentum = (recent_short_usd - recent_long_usd) / total_recent
        else:
            momentum = 0.0

        # --- 2. Cluster proximity ---
        proximity = 0.0
        if current_price > 0 and oi_usd > 0:
            long_clusters, short_clusters = self.estimate_clusters(
                current_price, oi_usd, funding_rate)
            proximity = self._weighted_proximity(current_price, long_clusters, short_clusters)

        # --- 3. Funding rate bias ---
        # High positive funding → longs crowded → long liquidation risk → negative
        funding_bias = -funding_rate * 100  # 0.01% → -1.0 scale
        funding_bias = max(-0.3, min(0.3, funding_bias))

        # Weighted combination
        signal = 0.5 * momentum + 0.3 * proximity + 0.2 * funding_bias
        return max(-1.0, min(1.0, signal))

    @staticmethod
    def _weighted_proximity(
        current_price: float,
        long_clusters: List[Tuple[float, float]],
        short_clusters: List[Tuple[float, float]],
    ) -> float:
        """Score cluster proximity: closer clusters exert stronger pull.

        Short clusters above → positive (upward magnet)
        Long clusters below → negative (downward magnet)

        Uses inverse-distance weighting: weight = usd / distance_pct^2
        """
        if current_price <= 0:
            return 0.0

        short_pull = 0.0  # positive direction
        long_pull = 0.0   # negative direction

        for price, usd in short_clusters:
            dist_pct = abs(price - current_price) / current_price
            if dist_pct > 0.001:  # avoid division by near-zero
                short_pull += usd / (dist_pct ** 2)

        for price, usd in long_clusters:
            dist_pct = abs(current_price - price) / current_price
            if dist_pct > 0.001:
                long_pull += usd / (dist_pct ** 2)

        total = short_pull + long_pull
        if total <= 0:
            return 0.0

        # Normalize to [-1, 1]: positive = short clusters dominate (upward magnet)
        return (short_pull - long_pull) / total

    # ------------------------------------------------------------------
    # Context for MarketSnapshot
    # ------------------------------------------------------------------

    def get_context(
        self,
        ticker: str,
        current_price: float,
        oi_usd: float,
        funding_rate: float,
    ) -> Dict[str, Any]:
        """Build liquidation context dict for a single ticker.

        Called by runner._build_snapshot() for each candidate ticker.
        Falls back gracefully when no WS data is available.
        """
        self._clean_window(ticker)

        long_dq = self._long_liqs.get(ticker, deque())
        short_dq = self._short_liqs.get(ticker, deque())

        long_usd_1h = self._sum_usd(long_dq)
        short_usd_1h = self._sum_usd(short_dq)
        total_1h = long_usd_1h + short_usd_1h

        signal = self.get_signal(ticker, current_price, oi_usd, funding_rate,
                                _already_cleaned=True)

        # Dominant side
        if total_1h > 0:
            dominant_side = "long" if long_usd_1h > short_usd_1h else "short"
        else:
            dominant_side = "neutral"

        # Nearest clusters
        nearest_above: Optional[Tuple[float, float]] = None
        nearest_below: Optional[Tuple[float, float]] = None
        if current_price > 0 and oi_usd > 0:
            long_clusters, short_clusters = self.estimate_clusters(
                current_price, oi_usd, funding_rate)
            # Nearest short cluster above (sorted by distance)
            above = [(p, u) for p, u in short_clusters if p > current_price]
            if above:
                nearest_above = min(above, key=lambda x: x[0] - current_price)
            # Nearest long cluster below
            below = [(p, u) for p, u in long_clusters if p < current_price]
            if below:
                nearest_below = max(below, key=lambda x: x[0])

        # Cascade risk: >$1M in 15 min or >$5M in 1h
        now = time.time()
        recent_cutoff = now - _RECENT_SECONDS
        recent_total = (self._sum_usd(long_dq, recent_cutoff)
                        + self._sum_usd(short_dq, recent_cutoff))
        cascade_risk = recent_total > 1_000_000 or total_1h > 5_000_000

        return {
            "signal": round(signal, 4),
            "long_liqs_1h_usd": long_usd_1h,
            "short_liqs_1h_usd": short_usd_1h,
            "dominant_side": dominant_side,
            "nearest_cluster_above": nearest_above,
            "nearest_cluster_below": nearest_below,
            "cascade_risk": cascade_risk,
        }

    def has_ws_data(self, ticker: str) -> bool:
        """Check if we have any WS liquidation data for this ticker."""
        long_dq = self._long_liqs.get(ticker, deque())
        short_dq = self._short_liqs.get(ticker, deque())
        return len(long_dq) > 0 or len(short_dq) > 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Diagnostic stats."""
        return {
            "total_events": self._total_events,
            "last_event_time": self._last_event_time,
            "tickers": len(self._tickers),
        }
