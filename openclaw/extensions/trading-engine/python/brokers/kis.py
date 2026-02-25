"""한국투자증권 broker integration via mojito2.

Supports US stock trading (overseas market).

Setup:
1. 한국투자증권 계좌 개설 + API 신청
2. Set environment variables:
   KIS_APP_KEY=your_app_key
   KIS_APP_SECRET=your_app_secret
   KIS_ACCOUNT=your_account_number  (e.g. "50123456-01")
   KIS_MOCK=true  (default, uses 모의투자)

Usage::

    from broker_kis import KISBroker
    broker = KISBroker(mock=True)
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


class KISBroker(BrokerBase):
    """한국투자증권 broker — 해외주식 (미국) 매매."""

    # KIS 해외주식 수수료 (2024 기준)
    COMMISSION_PCT = 0.0025  # 0.25%

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        account: Optional[str] = None,
        mock: bool = True,
    ) -> None:
        self.app_key = app_key or os.environ.get("KIS_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("KIS_APP_SECRET", "")
        self.account = account or os.environ.get("KIS_ACCOUNT", "")
        self.mock = mock if os.environ.get("KIS_MOCK", "true").lower() != "false" else False

        self._broker = None
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
        """Connect to KIS API via mojito2."""
        if not self.app_key or not self.app_secret or not self.account:
            return {
                "connected": False,
                "error": "Missing credentials. Set KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT.",
            }

        try:
            import mojito

            self._broker = mojito.KoreaInvestment(
                api_key=self.app_key,
                api_secret=self.app_secret,
                acc_no=self.account,
                mock=self.mock,
            )
            self._connected = True

            return {
                "connected": True,
                "mode": "mock" if self.mock else "live",
                "account": self.account,
                "exchange": "kis",
            }

        except Exception as exc:
            self._connected = False
            return {"connected": False, "error": str(exc), "exchange": "kis"}

    def get_account_status(self) -> Dict[str, Any]:
        """Get overseas stock account status."""
        if not self._ensure_connected():
            return {"error": "Not connected to KIS"}

        try:
            # 해외주식 잔고 조회 (NASDAQ + NYSE + AMEX)
            all_positions_raw = []
            cash = 0.0
            for exchange in ["NASD", "NYSE", "AMEX"]:
                try:
                    balance = self._broker.fetch_oversea_balance(exchange)
                    if "output1" in balance:
                        all_positions_raw.extend(balance["output1"])
                    if "output3" in balance and not cash:
                        cash = float(balance["output3"].get("frcr_dncl_amt_2", 0))
                except Exception:
                    continue

            positions = []
            total_value = 0.0

            for pos in all_positions_raw:
                    qty = float(pos.get("ovrs_cblc_qty", 0))
                    if qty <= 0:
                        continue
                    current_price = float(pos.get("now_pric2", 0))
                    buy_price = float(pos.get("pchs_avg_pric", 0))
                    market_value = float(pos.get("ovrs_stck_evlu_amt", 0))
                    pnl = float(pos.get("frcr_evlu_pfls_amt", 0))

                    total_value += market_value
                    positions.append({
                        "ticker": pos.get("ovrs_pdno", ""),
                        "qty": qty,
                        "entry_price": buy_price,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pl": pnl,
                    })

            return {
                "equity": total_value + cash,
                "cash": cash,
                "portfolio_value": total_value + cash,
                "positions": positions,
                "num_positions": len(positions),
                "mode": "mock" if self.mock else "live",
                "exchange": "NASD",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.error("KIS account status failed: %s", exc)
            return {"error": str(exc)}

    def execute_decisions(
        self,
        decisions: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute US stock trades via KIS."""
        if not self._ensure_connected() and not dry_run:
            return {"error": "Not connected to KIS"}

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

            if "BUY" in action:
                side = "buy"
                qty = int(position_usd / price) if price > 0 else 0
            elif "SELL" in action:
                side = "sell"
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
                "commission": round(qty * price * self.COMMISSION_PCT, 2),
                "reasons": decision.get("reasons", []),
            }

            if dry_run:
                order_result["status"] = "dry_run"
                order_result["message"] = f"Would {side} {qty} shares of {ticker}"
                successful += 1
            else:
                try:
                    rounded_price = str(round(price, 2))
                    if side == "buy":
                        resp = self._broker.create_oversea_order(
                            "NASD", ticker, "buy", str(qty), rounded_price,
                        )
                    else:
                        resp = self._broker.create_oversea_order(
                            "NASD", ticker, "sell", str(qty), rounded_price,
                        )

                    if resp.get("rt_cd") == "0":
                        order_result["status"] = "submitted"
                        order_result["order_id"] = resp.get("output", {}).get("ODNO", "")
                        successful += 1
                        logger.info(
                            "KIS order: %s %d %s @ %.2f",
                            side, qty, ticker, price,
                        )
                    else:
                        order_result["status"] = "failed"
                        order_result["error"] = resp.get("msg1", "Unknown error")
                        failed += 1

                except Exception as exc:
                    order_result["status"] = "failed"
                    order_result["error"] = str(exc)
                    failed += 1
                    logger.error("KIS order failed for %s: %s", ticker, exc)

            results.append(order_result)

        return {
            "timestamp": datetime.now().isoformat(),
            "exchange": "kis",
            "mode": "mock" if self.mock else "live",
            "dry_run": dry_run,
            "total_orders": total_orders,
            "successful": successful,
            "failed": failed,
            "orders": results,
        }

    def get_positions_detail(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed position info for all US stock holdings."""
        if not self._ensure_connected():
            return {}

        try:
            result = {}
            for exchange in ["NASD", "NYSE", "AMEX"]:
                try:
                    balance = self._broker.fetch_oversea_balance(exchange)
                except Exception:
                    continue
                if "output1" not in balance:
                    continue
                for pos in balance["output1"]:
                    qty = float(pos.get("ovrs_cblc_qty", 0))
                    if qty <= 0:
                        continue

                    ticker = pos.get("ovrs_pdno", "")
                    entry_price = float(pos.get("pchs_avg_pric", 0))
                    current_price = float(pos.get("now_pric2", 0))
                    market_value = float(pos.get("ovrs_stck_evlu_amt", 0))
                    pnl = float(pos.get("frcr_evlu_pfls_amt", 0))
                    plpc = pnl / (entry_price * qty) if entry_price * qty > 0 else 0

                    result[ticker] = {
                        "qty": qty,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pl": pnl,
                        "unrealized_plpc": plpc,
                        "side": "long",
                    }

            return result

        except Exception as exc:
            logger.error("KIS positions failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        if self._connected and self._broker is not None:
            return True
        result = self.connect()
        return result.get("connected", False)

    def _get_position_qty(self, ticker: str) -> int:
        """Get current position quantity for a ticker."""
        positions = self.get_positions_detail()
        info = positions.get(ticker, {})
        return int(info.get("qty", 0))


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    broker = KISBroker(mock=True)
    status = broker.connect()

    print(f"\n{'='*50}")
    print("KIS BROKER CONNECTION TEST")
    print(f"{'='*50}")

    if status["connected"]:
        print(f"Connected: YES ({status['mode']} mode)")
        print(f"Account: {status['account']}")
    else:
        print(f"Connected: NO")
        print(f"Error: {status.get('error', 'Unknown')}")
