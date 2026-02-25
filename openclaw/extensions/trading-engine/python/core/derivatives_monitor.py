"""Derivatives data monitor -- adaptive polling for market microstructure.

NOT a fixed timer. Polls at base interval, accelerates when extreme
values detected. Publishes events only when thresholds are breached.

Events published:
    market.funding_extreme   -- Funding rate exceeds threshold
    market.oi_spike          -- OI changed beyond threshold
    market.taker_delta       -- Taker buy/sell volume delta (CVD proxy)
    market.ls_ratio_extreme  -- Top trader long/short ratio extreme
    market.liquidation_cascade -- Large forced liquidation volume detected
    market.basis_extreme     -- Perp-spot basis spread annualized > 30%
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List

from core.event_bus import EventBus

logger = logging.getLogger("trading-engine.derivatives_monitor")


class DerivativesMonitor:
    """Adaptive derivatives data poller.

    Base interval: 5 minutes. When extreme values detected, accelerates
    to 1 minute for the next base-interval duration. This is throttling,
    not a fixed schedule.
    """

    def __init__(
        self,
        event_bus: EventBus,
        broker: Any,
        tickers: list,
        poll_base: float = 300.0,
        poll_fast: float = 60.0,
        funding_threshold: float = 0.0005,
        oi_spike_threshold: float = 0.10,
    ) -> None:
        self._event_bus = event_bus
        self._broker = broker
        self._tickers = tickers
        # Testnet: derivatives data is unreliable — poll less aggressively
        is_testnet = getattr(broker, 'paper', False)
        self._poll_base = max(poll_base, 900) if is_testnet else poll_base  # 15min on testnet
        self._poll_fast = max(poll_fast, 300) if is_testnet else poll_fast  # 5min on testnet
        self._funding_threshold = funding_threshold
        self._oi_spike_threshold = oi_spike_threshold

        # State
        self._running = False
        self._current_interval = poll_base
        self._last_oi: Dict[str, float] = {}
        self._last_funding: Dict[str, float] = {}
        self._last_taker: Dict[str, Dict[str, Any]] = {}
        self._last_ls_ratio: Dict[str, Dict[str, Any]] = {}
        self._extreme_until: float = 0.0

        # Liquidation cascade tracking — rolling 1h window
        self._liq_window: deque = deque(maxlen=5000)  # (timestamp, usd_amount)
        self._liq_seen_ts: set = set()     # dedup: timestamps already counted
        self._liq_cascade_threshold: float = 5_000_000  # $5M in 1h = cascade
        self._last_liq_context: Dict[str, Any] = {}

        # Basis spread (perp-spot premium)
        self._basis_spread: Dict[str, Dict] = {}

        # Stablecoin supply — cached, slow-moving (poll every 4h)
        self._stablecoin_cache: Dict[str, Any] = {}
        self._stablecoin_last_fetch: float = 0.0

    async def run(self) -> None:
        """Main adaptive polling loop."""
        self._running = True
        logger.info("[deriv] Starting derivatives monitor (%d tickers)", len(self._tickers))

        while self._running:
            try:
                await self._poll_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[deriv] Poll cycle failed: %s", exc)

            now = time.time()
            interval = self._poll_fast if now < self._extreme_until else self._poll_base
            self._current_interval = interval

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

        logger.info("[deriv] DerivativesMonitor stopped.")

    async def _poll_cycle(self) -> None:
        """Fetch funding rates, OI, taker delta, L/S ratio, liquidations, stablecoins."""
        # Skip entire cycle if broker is IP-banned (avoid extending ban)
        if hasattr(self._broker, '_ip_ban_until') and self._broker._ip_ban_until > time.time():
            remaining = self._broker._ip_ban_until - time.time()
            logger.info("[deriv] IP ban active (%.0fs remaining) — skipping poll cycle", remaining)
            return
        for ticker in self._tickers:
            try:
                await self._check_funding(ticker)
                await self._check_oi(ticker)
                await self._check_taker_delta(ticker)
                await self._check_long_short_ratio(ticker)
                await self._check_liquidations(ticker)
                await self._check_basis_spread(ticker)
            except Exception as exc:
                if "418" in str(exc) or "IP banned" in str(exc):
                    if hasattr(self._broker, '_detect_ip_ban'):
                        self._broker._detect_ip_ban(exc)
                    logger.warning("[deriv] IP ban detected during poll — stopping cycle")
                    return
                raise
        await self._check_stablecoin_supply()

    async def _check_funding(self, ticker: str) -> None:
        """Check funding rate for extreme values."""
        try:
            result = await asyncio.to_thread(self._broker.fetch_funding_rate, ticker)
            rate = result.get("fundingRate")
            if rate is None:
                return
            rate = float(rate)
            self._last_funding[ticker] = rate

            if abs(rate) > self._funding_threshold:
                direction = "long_crowded" if rate > 0 else "short_crowded"
                logger.info(
                    "[deriv] Funding extreme: %s rate=%.4f%% (%s)",
                    ticker, rate * 100, direction,
                )
                self._extreme_until = time.time() + self._poll_base
                await self._event_bus.publish("market.funding_extreme", {
                    "ticker": ticker,
                    "funding_rate": rate,
                    "direction": direction,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            logger.debug("[deriv] Funding fetch failed for %s: %s", ticker, exc)

    async def _check_oi(self, ticker: str) -> None:
        """Check open interest for spikes."""
        try:
            result = await asyncio.to_thread(self._broker.fetch_open_interest, ticker)
            oi = result.get("openInterestAmount")
            if oi is None:
                return
            oi = float(oi)

            prev_oi = self._last_oi.get(ticker)
            self._last_oi[ticker] = oi

            if prev_oi is None or prev_oi <= 0:
                return

            change_pct = abs(oi - prev_oi) / prev_oi
            if change_pct > self._oi_spike_threshold:
                direction = "increasing" if oi > prev_oi else "decreasing"
                logger.info(
                    "[deriv] OI spike: %s %.1f%% (%s)",
                    ticker, change_pct * 100, direction,
                )
                self._extreme_until = time.time() + self._poll_base
                await self._event_bus.publish("market.oi_spike", {
                    "ticker": ticker,
                    "open_interest": oi,
                    "previous_oi": prev_oi,
                    "change_pct": change_pct,
                    "direction": direction,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            logger.debug("[deriv] OI fetch failed for %s: %s", ticker, exc)

    async def _check_taker_delta(self, ticker: str) -> None:
        """Check taker buy/sell volume delta (CVD approximation).

        Uses Binance /futures/data/takerBuySellVol endpoint.
        Buy/sell ratio < 0.85 = sell pressure, > 1.15 = buy pressure.
        CVD momentum (recent vs prior) detects divergences.
        """
        try:
            symbol = ticker.replace("/", "").replace(":USDT", "")
            result = await asyncio.to_thread(
                self._broker.exchange.fapiPublicGetFuturesDataTakerBuySellVol,
                {"symbol": symbol, "period": "4h", "limit": 6},
            )
            if not result or len(result) < 4:
                return

            deltas = [float(r["buyVol"]) - float(r["sellVol"]) for r in result]
            cvd_recent = sum(deltas[-3:])
            cvd_prior = sum(deltas[:-3]) if len(deltas) > 3 else 0.0
            ratio_latest = float(result[-1]["buySellRatio"])
            cvd_momentum = cvd_recent - cvd_prior

            self._last_taker[ticker] = {
                "buy_sell_ratio": ratio_latest,
                "cvd_12h": cvd_recent,
                "cvd_momentum": cvd_momentum,
                "timestamp": time.time(),
            }

            if ratio_latest < 0.85 or ratio_latest > 1.15:
                direction = "buy_pressure" if ratio_latest > 1.15 else "sell_pressure"
                logger.info(
                    "[deriv] Taker delta extreme: %s ratio=%.2f (%s)",
                    ticker, ratio_latest, direction,
                )
                await self._event_bus.publish("market.taker_delta", {
                    "ticker": ticker,
                    "buy_sell_ratio": ratio_latest,
                    "cvd_12h": cvd_recent,
                    "cvd_momentum": cvd_momentum,
                    "direction": direction,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            logger.debug("[deriv] Taker delta fetch failed for %s: %s", ticker, exc)

    async def _check_long_short_ratio(self, ticker: str) -> None:
        """Check top trader long/short position ratio.

        Contrarian signal: extreme crowding often precedes a squeeze.
        Z-score > 1.5 = too many longs (bearish), < -1.5 = too many shorts (bullish).
        """
        try:
            symbol = ticker.replace("/", "").replace(":USDT", "")
            result = await asyncio.to_thread(
                self._broker.exchange.fapiPublicGetFuturesDataTopLongShortPositionRatio,
                {"symbol": symbol, "period": "4h", "limit": 6},
            )
            if not result or len(result) < 4:
                return

            ratios = [float(r["longShortRatio"]) for r in result]
            latest = ratios[-1]
            avg = sum(ratios) / len(ratios)
            std = (sum((x - avg) ** 2 for x in ratios) / len(ratios)) ** 0.5
            z_score = (latest - avg) / max(std, 0.01)

            self._last_ls_ratio[ticker] = {
                "long_short_ratio": latest,
                "z_score": z_score,
                "timestamp": time.time(),
            }

            if abs(z_score) > 1.5:
                direction = "long_crowded" if z_score > 0 else "short_crowded"
                signal = -0.5 * min(abs(z_score) / 3, 1.0) if z_score > 0 else 0.5 * min(abs(z_score) / 3, 1.0)
                logger.info(
                    "[deriv] L/S ratio extreme: %s ratio=%.2f z=%.2f (%s)",
                    ticker, latest, z_score, direction,
                )
                await self._event_bus.publish("market.ls_ratio_extreme", {
                    "ticker": ticker,
                    "long_short_ratio": latest,
                    "z_score": z_score,
                    "signal": signal,
                    "direction": direction,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            logger.debug("[deriv] L/S ratio fetch failed for %s: %s", ticker, exc)

    async def _check_liquidations(self, ticker: str) -> None:
        """Track forced liquidation volume. Large cascade = risk signal."""
        try:
            symbol = ticker.replace("/", "").replace(":USDT", "")
            result = await asyncio.to_thread(
                self._broker.exchange.fapiPublicGetForceOrders,
                {"symbol": symbol, "limit": 50},
            )
            if not result:
                return

            now = time.time()
            # Clean old entries (> 1 hour)
            while self._liq_window and self._liq_window[0][0] < now - 3600:
                old_ts = self._liq_window.popleft()[0]
                self._liq_seen_ts.discard(old_ts)
            # Periodic full sync to prevent set drift
            if len(self._liq_seen_ts) > len(self._liq_window) * 2:
                self._liq_seen_ts = {ts for ts, _ in self._liq_window}

            # Add new liquidation orders (dedup by timestamp)
            for order in result:
                ts = float(order.get("time", 0)) / 1000  # ms → s
                if ts < now - 3600:
                    continue
                if ts in self._liq_seen_ts:
                    continue
                usd = float(order.get("price", 0)) * float(order.get("origQty", 0))
                if usd > 0:
                    self._liq_window.append((ts, usd))
                    self._liq_seen_ts.add(ts)

            total_usd = sum(amt for _, amt in self._liq_window)
            count = len(self._liq_window)
            self._last_liq_context = {
                "total_usd_1h": total_usd,
                "count_1h": count,
                "is_cascade": total_usd > self._liq_cascade_threshold,
                "timestamp": now,
            }

            if total_usd > self._liq_cascade_threshold:
                logger.info(
                    "[deriv] Liquidation cascade: $%.0fM in 1h (%d orders)",
                    total_usd / 1e6, count,
                )
                self._extreme_until = now + self._poll_base
                await self._event_bus.publish("market.liquidation_cascade", {
                    "total_usd_1h": total_usd,
                    "count_1h": count,
                    "ticker": ticker,
                    "timestamp": now,
                })
        except Exception as exc:
            logger.debug("[deriv] Liquidation fetch failed for %s: %s", ticker, exc)

    async def _check_basis_spread(self, ticker: str) -> None:
        """Check perp-spot basis spread (premium/discount).

        basis_pct = (futures_mark - spot) / spot
        Annualized assuming 3x daily funding: basis_ann = basis_pct * 365 * 3
        |annualized| > 30% is extreme (carry trade signal or squeeze risk).
        """
        try:
            symbol = ticker.replace("/", "").replace(":USDT", "")

            # Spot price
            spot_ticker = ticker.replace(":USDT", "")
            spot_result = await asyncio.to_thread(
                self._broker.exchange.fetch_ticker, spot_ticker
            )
            spot_price = float(spot_result.get("last", 0))
            if spot_price <= 0:
                return

            # Futures mark price via premium index
            premium_result = await asyncio.to_thread(
                self._broker.exchange.fapiPublicGetPremiumIndex,
                {"symbol": symbol},
            )
            # Response can be a dict (single symbol) or list
            if isinstance(premium_result, list):
                premium_result = premium_result[0] if premium_result else {}
            futures_price = float(premium_result.get("markPrice", 0))
            if futures_price <= 0:
                return

            basis_pct = (futures_price - spot_price) / spot_price
            basis_annualized = basis_pct * 365 * 3

            self._basis_spread[ticker] = {
                "spot_price": spot_price,
                "futures_price": futures_price,
                "basis_pct": basis_pct,
                "basis_annualized": basis_annualized,
                "timestamp": time.time(),
            }

            if abs(basis_annualized) > 0.30:
                direction = "premium" if basis_annualized > 0 else "discount"
                logger.info(
                    "[deriv] Basis extreme: %s %.2f%% annualized (%s)",
                    ticker, basis_annualized * 100, direction,
                )
                self._extreme_until = time.time() + self._poll_base
                await self._event_bus.publish("market.basis_extreme", {
                    "ticker": ticker,
                    "basis_pct": basis_pct,
                    "basis_annualized": basis_annualized,
                    "spot_price": spot_price,
                    "futures_price": futures_price,
                    "direction": direction,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            logger.debug("[deriv] Basis spread fetch failed for %s: %s", ticker, exc)

    async def _check_stablecoin_supply(self) -> None:
        """Fetch stablecoin supply from DeFiLlama. Cached for 4 hours."""
        now = time.time()
        if now - self._stablecoin_last_fetch < 14400 and self._stablecoin_cache:
            return
        try:
            import requests
            resp = await asyncio.to_thread(
                requests.get,
                "https://stablecoins.llama.fi/stablecoins?includePrices=true",
                timeout=10,
            )
            data = resp.json()
            stablecoins = data.get("peggedAssets", [])

            total_mcap = 0.0
            usdt_mcap = 0.0
            usdc_mcap = 0.0
            for sc in stablecoins:
                symbol = sc.get("symbol", "")
                chains = sc.get("chainCirculating", {})
                mcap = sum(
                    v.get("current", {}).get("peggedUSD", 0)
                    for v in chains.values()
                )
                total_mcap += mcap
                if symbol == "USDT":
                    usdt_mcap = mcap
                elif symbol == "USDC":
                    usdc_mcap = mcap

            prev_total = self._stablecoin_cache.get("total_mcap", total_mcap)
            change_pct = (total_mcap / prev_total - 1) if prev_total > 0 else 0

            self._stablecoin_cache = {
                "total_mcap": total_mcap,
                "usdt_mcap": usdt_mcap,
                "usdc_mcap": usdc_mcap,
                "change_pct": change_pct,
                "timestamp": now,
            }
            self._stablecoin_last_fetch = now
            logger.info(
                "[deriv] Stablecoin supply: $%.1fB (USDT $%.1fB, USDC $%.1fB, chg %.2f%%)",
                total_mcap / 1e9, usdt_mcap / 1e9, usdc_mcap / 1e9, change_pct * 100,
            )
        except Exception as exc:
            logger.debug("[deriv] Stablecoin supply fetch failed: %s", exc)

    def get_context(self) -> Dict[str, Any]:
        """Return current derivatives context for pipeline injection."""
        return {
            "funding_rates": dict(self._last_funding),
            "open_interest": dict(self._last_oi),
            "taker_delta": dict(self._last_taker),
            "long_short_ratio": dict(self._last_ls_ratio),
            "liquidations": dict(self._last_liq_context),
            "basis_spread": dict(self._basis_spread),
            "stablecoin_supply": dict(self._stablecoin_cache),
            "poll_interval": self._current_interval,
            "is_extreme": time.time() < self._extreme_until,
        }

    def stop(self) -> None:
        """Signal the monitor to stop."""
        self._running = False
