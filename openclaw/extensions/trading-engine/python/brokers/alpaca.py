"""Alpaca broker integration for paper/live trading.

Executes trading decisions from AutoTrader via Alpaca API.
Supports paper trading (default) and live trading.

Paper trading:
- Base URL: https://paper-api.alpaca.markets
- No real money at risk
- Full order book simulation

Setup:
1. Create account at https://alpaca.markets
2. Get API keys from dashboard
3. Set environment variables:
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_PAPER=true  (default, set false for live)

Usage::

    from broker_alpaca import AlpacaBroker
    broker = AlpacaBroker()
    status = broker.get_account_status()
    broker.execute_decisions(auto_trader_decisions)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from brokers.base import BrokerBase

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerBase):
    """Alpaca broker integration for order execution and portfolio tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = paper if os.environ.get("ALPACA_PAPER", "true").lower() != "false" else False

        self._api = None
        self._connected = False

    @property
    def market_type(self) -> str:
        return "equity"

    @property
    def is_24h_market(self) -> bool:
        return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> Dict[str, Any]:
        """Connect to Alpaca API and verify credentials."""
        if not self.api_key or not self.secret_key:
            return {
                "connected": False,
                "error": "Missing API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.",
                "setup_url": "https://alpaca.markets/docs/trading/getting_started/",
            }

        try:
            import alpaca_trade_api as tradeapi

            base_url = (
                "https://paper-api.alpaca.markets"
                if self.paper
                else "https://api.alpaca.markets"
            )

            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=base_url,
                api_version="v2",
            )

            # Verify connection
            account = self._api.get_account()
            self._connected = True

            return {
                "connected": True,
                "mode": "paper" if self.paper else "live",
                "account_id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
            }

        except Exception as exc:
            self._connected = False
            return {
                "connected": False,
                "error": str(exc),
                "mode": "paper" if self.paper else "live",
            }

    def get_account_status(self) -> Dict[str, Any]:
        """Get current account status and portfolio summary."""
        if not self._ensure_connected():
            return {"error": "Not connected to Alpaca"}

        try:
            account = self._api.get_account()
            positions = self._api.list_positions()

            position_list = []
            for pos in positions:
                position_list.append({
                    "ticker": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                    "side": pos.side,
                })

            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "positions": position_list,
                "num_positions": len(position_list),
                "total_unrealized_pl": sum(p["unrealized_pl"] for p in position_list),
                "mode": "paper" if self.paper else "live",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.error("Failed to get account status: %s", exc)
            return {"error": str(exc)}

    def execute_decisions(
        self,
        decisions: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute trading decisions from AutoTrader.

        Args:
            decisions: List of decision dicts from AutoTrader.decide()
            dry_run: If True, don't actually submit orders (just simulate)

        Returns execution report with order statuses.
        """
        if not self._ensure_connected() and not dry_run:
            return {"error": "Not connected to Alpaca"}

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

            # Determine order side and quantity
            if "BUY" in action:
                side = "buy"
                qty = int(position_usd / price) if price > 0 else 0
            elif "SELL" in action:
                side = "sell"
                # Sell existing position
                qty = self._get_position_qty(ticker)
            else:
                continue

            if qty <= 0:
                continue

            total_orders += 1

            order_result = {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "confidence": confidence,
                "estimated_value": round(qty * price, 2),
                "reasons": decision.get("reasons", []),
            }

            if dry_run:
                order_result["status"] = "dry_run"
                order_result["message"] = f"Would {side} {qty} shares of {ticker}"
                successful += 1
            else:
                try:
                    order = self._api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        type="market",
                        time_in_force="day",
                    )
                    order_result["status"] = "submitted"
                    order_result["order_id"] = order.id
                    order_result["submitted_at"] = order.submitted_at
                    successful += 1
                    logger.info(
                        "Order submitted: %s %d %s (confidence=%.2f)",
                        side, qty, ticker, confidence,
                    )
                except Exception as exc:
                    order_result["status"] = "failed"
                    order_result["error"] = str(exc)
                    failed += 1
                    logger.error("Order failed for %s: %s", ticker, exc)

            results.append(order_result)

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "paper" if self.paper else "live",
            "dry_run": dry_run,
            "total_orders": total_orders,
            "successful": successful,
            "failed": failed,
            "orders": results,
        }

    def get_positions_as_dict(self) -> Dict[str, float]:
        """Get current positions as {ticker: market_value} dict.

        Used as input for AutoTrader.decide(current_positions=...).
        """
        if not self._ensure_connected():
            return {}

        try:
            positions = self._api.list_positions()
            return {
                pos.symbol: float(pos.market_value)
                for pos in positions
            }
        except Exception as exc:
            logger.error("Failed to get positions: %s", exc)
            return {}

    def get_positions_detail(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed position info for risk checks and exit management.

        Returns {ticker: {qty, entry_price, current_price, market_value,
        unrealized_plpc, side}} for all open positions.
        """
        if not self._ensure_connected():
            return {}

        try:
            positions = self._api.list_positions()
            result = {}
            for pos in positions:
                qty = float(pos.qty)
                cost_basis = float(pos.cost_basis)
                entry_price = cost_basis / qty if qty > 0 else 0.0
                result[pos.symbol] = {
                    "qty": qty,
                    "entry_price": entry_price,
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "cost_basis": cost_basis,
                    "side": pos.side,
                }
            return result
        except Exception as exc:
            logger.error("Failed to get position details: %s", exc)
            return {}

    def get_recent_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent order history."""
        if not self._ensure_connected():
            return []

        try:
            orders = self._api.list_orders(limit=limit, status="all")
            return [
                {
                    "ticker": o.symbol,
                    "side": o.side,
                    "qty": float(o.qty) if o.qty else 0,
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "type": o.type,
                    "status": o.status,
                    "submitted_at": str(o.submitted_at),
                    "filled_at": str(o.filled_at) if o.filled_at else None,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                }
                for o in orders
            ]
        except Exception as exc:
            logger.error("Failed to get orders: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        """Ensure we have an active API connection."""
        if self._connected and self._api is not None:
            return True
        result = self.connect()
        return result.get("connected", False)

    def _get_position_qty(self, ticker: str) -> int:
        """Get current position quantity for a ticker."""
        if not self._ensure_connected():
            return 0
        try:
            pos = self._api.get_position(ticker)
            return int(float(pos.qty))
        except Exception:
            return 0


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    broker = AlpacaBroker()
    status = broker.connect()

    print(f"\n{'='*50}")
    print(f"ALPACA BROKER CONNECTION TEST")
    print(f"{'='*50}")

    if status["connected"]:
        print(f"Connected: YES ({status['mode']} mode)")
        print(f"Equity:    ${status['equity']:,.2f}")
        print(f"Cash:      ${status['cash']:,.2f}")
        print(f"Buying Power: ${status['buying_power']:,.2f}")

        # Test dry run with sample decisions
        sample_decisions = [
            {"ticker": "AAPL", "action": "BUY", "confidence": 0.5,
             "position_size_usd": 1000, "price": 185.0, "reasons": ["test"]},
        ]
        result = broker.execute_decisions(sample_decisions, dry_run=True)
        print(f"\nDry run: {result['successful']}/{result['total_orders']} orders")
        for order in result["orders"]:
            print(f"  {order['side']} {order['qty']} {order['ticker']} -> {order['status']}")
    else:
        print(f"Connected: NO")
        print(f"Error: {status.get('error', 'Unknown')}")
        print(f"\nTo set up Alpaca paper trading:")
        print(f"  1. Create account: https://alpaca.markets")
        print(f"  2. Get API keys from dashboard")
        print(f"  3. Set env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY")
