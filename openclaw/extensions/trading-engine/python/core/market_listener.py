"""Binance WebSocket combined stream for real-time multi-timeframe kline events.

Subscribes to multiple kline intervals (e.g. 15m/1h/4h) for configured tickers.
Publishes events when candles close and when significant price moves occur.

Events published:
    market.tick             -- every WS message (for AdaptiveGate)
    market.candle_close     -- primary interval candle closed (triggers pipeline)
    market.significant_move -- BTC moved >1.5% since last decision
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import websockets

from core.event_bus import EventBus
from core.ohlcv_store import OHLCVStore

logger = logging.getLogger("trading-engine.market_listener")


class MarketListener:
    """Binance WebSocket kline listener -- drives event-based trading.

    NOT a fixed timer. Emits events only when market conditions warrant:
    - Primary interval candle closes (natural market rhythm)
    - BTC moves >1.5% since last decision (anomaly-driven)

    Subscribes to multiple kline intervals simultaneously for multi-TF analysis.
    Feeds closed candles into OHLCVStore (eliminates REST OHLCV polling).
    """

    # WS base URLs per environment
    WS_URLS = {
        "spot_live": "wss://stream.binance.com:9443/stream",
        "spot_testnet": "wss://stream.testnet.binance.vision/stream",
        "future_live": "wss://fstream.binance.com/stream",
        "future_testnet": "wss://fstream.binancefuture.com/stream",
    }

    def __init__(
        self,
        event_bus: EventBus,
        tickers: list,
        kline_intervals: List[str] = None,
        primary_interval: str = "15m",
        significant_move_pct: float = 0.015,
        min_decision_gap: float = 120.0,
        market: str = "spot",
        testnet: bool = True,
        ohlcv_store: Optional[OHLCVStore] = None,
        # Legacy single-interval param (backward compat)
        kline_interval: str = None,
    ) -> None:
        self._event_bus = event_bus
        self._tickers = tickers
        self._significant_move_pct = significant_move_pct
        self._min_decision_gap = min_decision_gap

        # Multi-TF: list of intervals to subscribe
        if kline_intervals:
            self._kline_intervals = kline_intervals
        elif kline_interval:
            self._kline_intervals = [kline_interval]
        else:
            self._kline_intervals = ["15m", "1h", "4h"]
        self._primary_interval = primary_interval

        # Select correct WS URL
        env = "testnet" if testnet else "live"
        self._ws_base = self.WS_URLS[f"{market}_{env}"]

        # OHLCV store — WS candle closes feed into this (replaces REST polling)
        self._ohlcv_store = ohlcv_store

        # State
        self._last_decision_time: float = 0.0
        self._last_decision_prices: Dict[str, float] = {}
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._last_message_time: float = 0.0  # for watchdog

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all tickers x all intervals."""
        streams = []
        for tic in self._tickers:
            symbol = tic.replace("/", "").lower()
            for interval in self._kline_intervals:
                streams.append(f"{symbol}@kline_{interval}")
        return f"{self._ws_base}?streams={'/'.join(streams)}"

    async def run(self) -> None:
        """Main loop: connect, listen, reconnect with exponential backoff."""
        self._running = True
        backoff = 1.0

        while self._running:
            try:
                url = self._build_stream_url()
                logger.info("[ws] Connecting to Binance stream: %d tickers", len(self._tickers))
                async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                    self._ws = ws
                    backoff = 1.0
                    logger.info("[ws] Connected. Listening for %s kline events...",
                                "/".join(self._kline_intervals))
                    await self._listen(ws)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("[ws] Disconnected: %s. Reconnecting in %.0fs...", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        logger.info("[ws] MarketListener stopped.")

    async def _listen(self, ws) -> None:
        """Process incoming WebSocket messages."""
        async for raw in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw)
                self._last_message_time = time.time()
                data = msg.get("data", {})
                if data.get("e") == "kline":
                    await self._handle_kline(data)
            except Exception as exc:
                logger.warning("[ws] Failed to process message: %s", exc)

    async def _handle_kline(self, data: dict) -> None:
        """Process a kline event from any subscribed interval."""
        kline = data.get("k", {})
        symbol = data.get("s", "")
        ticker = self._symbol_to_ticker(symbol)
        interval = kline.get("i", "")  # e.g. "15m", "1h", "4h"
        close_price = float(kline.get("c", 0))
        volume = float(kline.get("v", 0))
        is_closed = kline.get("x", False)

        if close_price <= 0:
            return

        # 0. Raw tick for AdaptiveGate (every WS message, all intervals)
        await self._event_bus.publish("market.tick", {
            "ticker": ticker,
            "price": close_price,
            "volume": volume,
            "interval": interval,
            "is_closed": is_closed,
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "timestamp": time.time(),
        })

        # 1a. Feed closed candles into OHLCV store (all intervals)
        if is_closed and self._ohlcv_store:
            kline_open_time = int(kline.get("t", 0))  # Binance kline open time (ms)
            self._ohlcv_store.append(ticker, interval, {
                "timestamp": kline_open_time / 1000.0 if kline_open_time > 1e12 else kline_open_time,
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": close_price,
                "volume": volume,
            })

        # 1b. Candle close on primary interval -> pipeline trigger
        if is_closed and interval == self._primary_interval:
            await self._emit_candle_close(ticker, close_price, kline)

        # 2. Significant move detection (BTC only, any interval)
        if "BTC" in ticker:
            await self._check_significant_move(ticker, close_price)

    async def _emit_candle_close(self, ticker: str, price: float, kline: dict) -> None:
        """Emit candle close event with cooldown debounce."""
        now = time.time()
        if now - self._last_decision_time < self._min_decision_gap:
            return

        self._last_decision_time = now
        self._last_decision_prices[ticker] = price

        logger.info("[event] Candle close (%s): %s @ %.2f", self._primary_interval, ticker, price)
        await self._event_bus.publish("market.candle_close", {
            "ticker": ticker,
            "price": price,
            "interval": self._primary_interval,
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": price,
            "volume": float(kline.get("v", 0)),
            "timestamp": time.time(),
        })

    async def _check_significant_move(self, ticker: str, current_price: float) -> None:
        """Check if BTC moved significantly since last decision."""
        last_price = self._last_decision_prices.get(ticker)
        if last_price is None or last_price <= 0:
            self._last_decision_prices[ticker] = current_price
            return

        now = time.time()
        if now - self._last_decision_time < self._min_decision_gap:
            return

        pct_change = abs(current_price - last_price) / last_price
        if pct_change >= self._significant_move_pct:
            self._last_decision_time = now
            self._last_decision_prices[ticker] = current_price
            direction = "up" if current_price > last_price else "down"

            logger.info(
                "[event] Significant move: %s %s %.1f%% (%.2f -> %.2f)",
                ticker, direction, pct_change * 100, last_price, current_price,
            )
            await self._event_bus.publish("market.significant_move", {
                "ticker": ticker,
                "price": current_price,
                "previous_price": last_price,
                "pct_change": pct_change,
                "direction": direction,
                "timestamp": time.time(),
            })

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._running = False

    @staticmethod
    def _symbol_to_ticker(symbol: str) -> str:
        """Convert BTCUSDT -> BTC/USDT."""
        symbol = symbol.upper()
        for quote in ("USDT", "BUSD", "BTC"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return f"{symbol[:-len(quote)]}/{quote}"
        return symbol
