"""Binance broker integration via ccxt.

Supports Spot and Futures trading with testnet mode.

Setup:
1. Create API key at https://www.binance.com/en/my/settings/api-management
2. Set environment variables:
   BINANCE_API_KEY=your_key
   BINANCE_SECRET_KEY=your_secret
   BINANCE_PAPER=true  (default, uses testnet)

Usage::

    from broker_binance import BinanceBroker
    broker = BinanceBroker(paper=True, market="spot")
    broker.connect()
    broker.execute_decisions(decisions)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from brokers.base import BrokerBase

logger = logging.getLogger(__name__)


class BinanceBroker(BrokerBase):
    """Binance broker via ccxt — Spot + Futures, testnet support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper: bool = True,
        market: str = "spot",
    ) -> None:
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self.secret = secret or os.environ.get("BINANCE_SECRET_KEY", "")
        self.paper = paper if os.environ.get("BINANCE_PAPER", "true").lower() != "false" else False
        self.market = market  # "spot" or "future"

        self._exchange = None
        self._connected = False
        self._ip_ban_until: float = 0.0  # epoch seconds when ban expires

    @property
    def market_type(self) -> str:
        return "crypto"

    @property
    def is_24h_market(self) -> bool:
        return True

    @property
    def exchange(self):
        """Expose ccxt exchange for direct API access (derivatives data)."""
        return self._exchange

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> Dict[str, Any]:
        """Connect to Binance via ccxt."""
        if not self.api_key or not self.secret:
            return {
                "connected": False,
                "error": "Missing API keys. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.",
            }

        try:
            import ccxt

            options = {"defaultType": self.market}

            self._exchange = ccxt.binance({
                "apiKey": self.api_key,
                "secret": self.secret,
                "options": options,
                "enableRateLimit": True,
            })

            if self.paper:
                # Futures: sandbox deprecated → use demo trading (demo-api.binance.com)
                # Spot: sandbox still works (testnet.binance.vision)
                use_demo = (
                    self.market == "future"
                    or os.environ.get("BINANCE_USE_DEMO", "").lower() in ("1", "true")
                )
                if use_demo and hasattr(self._exchange, "enable_demo_trading"):
                    self._exchange.enable_demo_trading(True)
                    logger.info("Demo trading enabled (demo-api.binance.com) for %s", self.market)
                else:
                    self._exchange.set_sandbox_mode(True)
                    logger.info("Sandbox mode enabled (testnet.binance.vision) for %s", self.market)

            # Verify connection
            balance = self._exchange.fetch_balance()
            self._connected = True

            total = balance.get("total", {})
            usdt = float(total.get("USDT", 0))

            is_demo = os.environ.get("BINANCE_USE_DEMO", "").lower() in ("1", "true")
            paper_mode = "demo" if (self.paper and is_demo) else ("testnet" if self.paper else "live")

            return {
                "connected": True,
                "mode": paper_mode,
                "market": self.market,
                "exchange": "binance",
                "usdt_balance": usdt,
                "total_assets": {k: float(v) for k, v in total.items() if float(v) > 0},
            }

        except Exception as exc:
            self._connected = False
            self._detect_ip_ban(exc)
            err = str(exc)
            is_demo = os.environ.get("BINANCE_USE_DEMO", "").lower() in ("1", "true")
            if is_demo and ("Invalid API-key" in err or "-2015" in err):
                err += (
                    " | Demo trading requires API keys from https://www.binance.com/en/demo-trading"
                    " (testnet keys do NOT work). Create demo keys and update .env."
                )
            return {"connected": False, "error": err, "exchange": "binance"}

    def get_account_status(self) -> Dict[str, Any]:
        """Get account balances and open positions."""
        if not self._ensure_connected():
            return {"error": "Not connected to Binance"}

        try:
            balance = self._exchange.fetch_balance()
            total = balance.get("total", {})
            free = balance.get("free", {})

            usdt_total = float(total.get("USDT", 0))
            usdt_free = float(free.get("USDT", 0))

            # Non-zero holdings
            holdings = []
            for symbol, amount in total.items():
                amt = float(amount)
                if amt > 0 and symbol != "USDT":
                    holdings.append({
                        "ticker": f"{symbol}/USDT",
                        "qty": amt,
                        "symbol": symbol,
                    })

            # Estimate portfolio value
            portfolio_value = usdt_total
            for h in holdings:
                try:
                    ticker_info = self._exchange.fetch_ticker(h["ticker"])
                    price = ticker_info.get("last", 0)
                    h["current_price"] = price
                    h["market_value"] = h["qty"] * price
                    portfolio_value += h["market_value"]
                except Exception:
                    h["current_price"] = 0
                    h["market_value"] = 0

            return {
                "equity": portfolio_value,
                "cash": usdt_free,
                "portfolio_value": portfolio_value,
                "positions": holdings,
                "num_positions": len(holdings),
                "mode": "testnet" if self.paper else "live",
                "market": self.market,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.error("Binance account status failed: %s", exc)
            return {"error": str(exc)}

    def execute_decisions(
        self,
        decisions: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute trading decisions on Binance."""
        if not self._ensure_connected() and not dry_run:
            return {"error": "Not connected to Binance"}

        results = []
        total_orders = 0
        successful = 0
        failed = 0

        for decision in decisions:
            action = decision.get("action", "HOLD")
            ticker = decision.get("ticker", "")
            confidence = decision.get("confidence", 0)
            position_usd = decision.get("position_size_usd", 0)
            price = decision.get("price", 0)

            if action == "HOLD" or not ticker:
                continue

            # Normalize ticker format (BTCUSDT -> BTC/USDT)
            symbol = self._normalize_symbol(ticker)

            if action in ("BUY", "COVER"):
                side = "buy"
                if action == "COVER":
                    # COVER: buy back to close short — use qty from decision
                    qty = decision.get("qty", 0)
                    if qty <= 0:
                        qty = self._get_position_qty(symbol)
                else:
                    qty = position_usd / price if price > 0 else 0
            elif action in ("SELL", "SHORT"):
                side = "sell"
                if action == "SHORT":
                    # SHORT: sell to open — use position_size_usd
                    qty = position_usd / price if price > 0 else 0
                else:
                    # SELL: close long — use qty from decision
                    qty = decision.get("qty", 0)
                    if qty <= 0:
                        qty = self._get_position_qty(symbol)
            else:
                continue

            if qty <= 0:
                logger.warning("Order skipped: %s %s qty=0 (price=%.4f, usd=%.2f)",
                               side, ticker, price, position_usd)
                continue

            # Round to exchange precision
            raw_qty = qty
            qty = self._round_qty(symbol, qty)
            if qty <= 0:
                logger.warning("Order skipped after rounding: %s %s raw_qty=%.8f rounded=0",
                               side, ticker, raw_qty)
                continue

            total_orders += 1

            order_result = {
                "ticker": ticker,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "confidence": confidence,
                "estimated_value": round(qty * price, 2),
                "reasons": decision.get("reasons", []),
            }

            if dry_run:
                order_result["status"] = "dry_run"
                order_result["message"] = f"Would {side} {qty} {symbol}"
                successful += 1
            else:
                try:
                    # Use limit order for maker fees (0.02% vs 0.05% taker)
                    order_type = "limit"
                    ticker_info = self._exchange.fetch_ticker(symbol)
                    current_price = ticker_info.get("last", price)

                    # Adaptive offset for fill probability: 0.1% base, wider during volatility
                    # Check if decision includes ATR-based info for smarter offset
                    atr_pct = decision.get("atr_pct", 0)
                    offset = max(0.001, min(atr_pct * 0.3, 0.005)) if atr_pct > 0 else 0.001
                    if side == "buy":
                        limit_price = current_price * (1 + offset)
                    else:
                        limit_price = current_price * (1 - offset)

                    limit_price = self._round_price(symbol, limit_price)

                    params = {}
                    if self.market == "future":
                        # SELL/COVER close existing positions — mark reduceOnly
                        if action in ("SELL", "COVER"):
                            params["reduceOnly"] = True
                    order = self._exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side,
                        amount=qty,
                        price=limit_price,
                        params=params,
                    )
                    order_result["status"] = "submitted"
                    order_result["order_id"] = order.get("id")
                    order_result["order_type"] = order_type
                    order_result["limit_price"] = limit_price
                    successful += 1
                    logger.info(
                        "Binance order: %s %s %s @ %.2f (conf=%.2f)",
                        side, qty, symbol, limit_price, confidence,
                    )
                except Exception as exc:
                    # Close failures: qty mismatch or below minimum — retry with fresh qty
                    err_str = str(exc)
                    if action in ("SELL", "COVER") and ("-4118" in err_str or "minimum amount" in err_str):
                        try:
                            fresh_qty = self._get_position_qty(symbol)
                            if fresh_qty > 0 and fresh_qty != qty:
                                fresh_qty = self._round_qty(symbol, fresh_qty)
                                logger.info("Retrying %s %s with fresh qty %.8f (was %.8f)", side, symbol, fresh_qty, qty)
                                retry_order = self._exchange.create_order(
                                    symbol=symbol, type=order_type, side=side,
                                    amount=fresh_qty, price=limit_price, params=params,
                                )
                                order_result["status"] = "submitted"
                                order_result["order_id"] = retry_order.get("id")
                                order_result["qty"] = fresh_qty
                                order_result["retry"] = True
                                successful += 1
                                logger.info("Retry succeeded: %s %s %s @ %.2f", side, fresh_qty, symbol, limit_price)
                                results.append(order_result)
                                continue
                            elif fresh_qty <= 0:
                                logger.warning("No position found on exchange for %s — position may be closed", symbol)
                        except Exception as retry_exc:
                            logger.error("Retry also failed for %s: %s", symbol, retry_exc)
                    order_result["status"] = "failed"
                    order_result["error"] = str(exc)
                    failed += 1
                    logger.error("Binance order failed for %s: %s (qty=%.8f, price=%.2f, notional=%.2f)", symbol, exc, qty, limit_price, qty * limit_price)

            results.append(order_result)

        return {
            "timestamp": datetime.now().isoformat(),
            "exchange": "binance",
            "mode": "testnet" if self.paper else "live",
            "market": self.market,
            "dry_run": dry_run,
            "total_orders": total_orders,
            "successful": successful,
            "failed": failed,
            "orders": results,
        }

    def get_positions_detail(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed position info for all holdings."""
        if not self._ensure_connected():
            return {}

        try:
            result = {}

            if self.market == "future":
                # Futures: fetch_positions returns actual long/short info
                try:
                    positions = self._exchange.fetch_positions()
                    for pos in positions:
                        amt = abs(float(pos.get("contracts", 0) or 0))
                        if amt <= 0:
                            continue
                        symbol = pos.get("symbol", "")
                        # Normalize: BTC/USDT:USDT → BTC/USDT for internal consistency
                        if ":" in symbol:
                            symbol = symbol.split(":")[0]
                        side = pos.get("side", "long")  # "long" or "short"
                        entry_px = float(pos.get("entryPrice", 0) or 0)
                        current_px = float(pos.get("markPrice", 0) or pos.get("liquidationPrice", 0) or 0)
                        unrealized = float(pos.get("unrealizedPnl", 0) or 0)
                        result[symbol] = {
                            "qty": amt,
                            "entry_price": entry_px,
                            "current_price": current_px,
                            "market_value": amt * current_px,
                            "unrealized_pl": unrealized,
                            "unrealized_plpc": unrealized / (entry_px * amt) if entry_px * amt > 0 else 0,
                            "side": side,
                        }
                except Exception as exc:
                    logger.warning("fetch_positions failed, falling back to balance: %s", exc)

            if not result:
                # Spot fallback: balance-based (no short info available)
                balance = self._exchange.fetch_balance()
                total = balance.get("total", {})
                for symbol, amount in total.items():
                    amt = float(amount)
                    if amt <= 0 or symbol == "USDT":
                        continue
                    pair = f"{symbol}/USDT"
                    try:
                        ticker_info = self._exchange.fetch_ticker(pair)
                        current_price = ticker_info.get("last", 0)
                    except Exception:
                        continue
                    result[pair] = {
                        "qty": amt,
                        "entry_price": 0.0,
                        "current_price": current_price,
                        "market_value": amt * current_price,
                        "unrealized_pl": 0.0,
                        "unrealized_plpc": 0.0,
                        "side": "long",
                    }

            return result

        except Exception as exc:
            logger.error("Binance positions failed: %s", exc)
            return {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 500,
    ) -> List[List]:
        """Fetch OHLCV data via ccxt.

        Returns list of [timestamp, open, high, low, close, volume].
        """
        if not self._ensure_connected():
            return []

        try:
            return self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as exc:
            logger.error("Binance OHLCV fetch failed for %s: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # Derivatives data (Funding Rate, Open Interest)
    # ------------------------------------------------------------------

    def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch current funding rate for a Futures symbol.

        Works from Spot mode too (reads public Futures data).
        Returns dict with 'fundingRate', 'fundingTimestamp', etc.
        """
        if not self._ensure_connected():
            return {}
        try:
            normalized = self._normalize_symbol(symbol)
            return self._exchange.fetch_funding_rate(normalized)
        except Exception as exc:
            logger.debug("fetch_funding_rate failed for %s: %s", symbol, exc)
            return {}

    def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetch current open interest for a Futures symbol.

        Returns dict with 'openInterestAmount', 'openInterestValue', etc.
        """
        if not self._ensure_connected():
            return {}
        try:
            normalized = self._normalize_symbol(symbol)
            return self._exchange.fetch_open_interest(normalized)
        except Exception as exc:
            logger.debug("fetch_open_interest failed for %s: %s", symbol, exc)
            return {}

    # ------------------------------------------------------------------
    # Futures leverage
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a Futures symbol. No-op for Spot."""
        if self.market != "future" or not self._ensure_connected():
            return {"skipped": True, "reason": "spot mode or not connected"}
        try:
            result = self._exchange.set_leverage(leverage, self._normalize_symbol(symbol))
            logger.info("Set leverage %dx for %s", leverage, symbol)
            return {"symbol": symbol, "leverage": leverage, "result": result}
        except Exception as exc:
            logger.warning("set_leverage failed for %s: %s", symbol, exc)
            return {"error": str(exc)}

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> Dict[str, Any]:
        """Set margin mode (isolated/cross) for Futures. No-op for Spot."""
        if self.market != "future" or not self._ensure_connected():
            return {"skipped": True}
        try:
            result = self._exchange.set_margin_mode(mode, self._normalize_symbol(symbol))
            logger.info("Set margin mode %s for %s", mode, symbol)
            return {"symbol": symbol, "mode": mode, "result": result}
        except Exception as exc:
            # Often fails if already set — not critical
            logger.debug("set_margin_mode: %s (may already be set)", exc)
            return {"already_set": True}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        # Block ALL REST calls during IP ban
        import time as _time
        if self._ip_ban_until > _time.time():
            remaining = self._ip_ban_until - _time.time()
            logger.debug("IP ban active, %d s remaining — skipping REST call", int(remaining))
            return False
        if self._connected and self._exchange is not None:
            return True
        result = self.connect()
        return result.get("connected", False)

    def _detect_ip_ban(self, exc: Exception) -> None:
        """Detect IP ban from error and set ban_until timestamp."""
        import re
        import time as _time
        err_str = str(exc)
        if "418" in err_str or "IP banned" in err_str:
            m = re.search(r"IP banned until (\d+)", err_str)
            if m:
                ban_ts = int(m.group(1)) / 1000
                self._ip_ban_until = ban_ts + 60  # +60s margin
                remaining = ban_ts - _time.time()
                logger.error("IP BANNED for %.0f more seconds (until ts=%s)", remaining, m.group(1))
            else:
                # No timestamp — assume 10 min ban
                self._ip_ban_until = _time.time() + 600
                logger.error("IP BANNED (no timestamp) — blocking REST for 600s")

    def _normalize_symbol(self, ticker: str) -> str:
        """Normalize ticker to ccxt format.

        Spot:   BTC/USDT
        Future: BTC/USDT:USDT  (ccxt unified symbol for linear perpetual)
        """
        # Strip futures suffix if present (handled below)
        ticker = ticker.upper().replace("-", "/")
        if ":USDT" in ticker:
            ticker = ticker.split(":")[0]
        if "/" not in ticker:
            # BTCUSDT -> BTC/USDT
            for quote in ("USDT", "BUSD", "BTC", "ETH"):
                if ticker.endswith(quote) and len(ticker) > len(quote):
                    ticker = f"{ticker[:-len(quote)]}/{quote}"
                    break
            else:
                ticker = f"{ticker}/USDT"
        # Append futures settlement currency
        if self.market == "future" and ":" not in ticker:
            ticker = f"{ticker}:USDT"
        return ticker

    def _get_position_qty(self, symbol: str) -> float:
        """Get current holding quantity for a symbol."""
        if not self._ensure_connected():
            return 0.0
        try:
            if self.market == "future":
                positions = self._exchange.fetch_positions([symbol])
                for p in positions:
                    contracts = abs(float(p.get("contracts", 0)))
                    if contracts > 0:
                        return contracts
                return 0.0
            else:
                base = symbol.split("/")[0]
                balance = self._exchange.fetch_balance()
                return float(balance.get("total", {}).get(base, 0))
        except Exception:
            return 0.0

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange precision using ccxt's built-in method."""
        try:
            return float(self._exchange.amount_to_precision(symbol, qty))
        except Exception:
            return round(qty, 6)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision using ccxt's built-in method."""
        try:
            return float(self._exchange.price_to_precision(symbol, price))
        except Exception:
            return round(price, 2)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    broker = BinanceBroker(paper=True)
    status = broker.connect()

    print(f"\n{'='*50}")
    print("BINANCE BROKER CONNECTION TEST")
    print(f"{'='*50}")

    if status["connected"]:
        print(f"Connected: YES ({status['mode']} mode, {status['market']})")
        print(f"USDT Balance: ${status['usdt_balance']:,.2f}")

        # Test dry run
        sample = [
            {"ticker": "BTC/USDT", "action": "BUY", "confidence": 0.5,
             "position_size_usd": 100, "price": 60000.0, "reasons": ["test"]},
        ]
        result = broker.execute_decisions(sample, dry_run=True)
        print(f"\nDry run: {result['successful']}/{result['total_orders']} orders")
    else:
        print(f"Connected: NO")
        print(f"Error: {status.get('error', 'Unknown')}")
