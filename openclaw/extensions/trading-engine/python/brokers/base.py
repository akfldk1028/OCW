"""Abstract broker interface for multi-platform trading.

All broker implementations (Alpaca, Binance, KIS) inherit from this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BrokerBase(ABC):
    """Abstract base class for broker integrations."""

    @property
    @abstractmethod
    def market_type(self) -> str:
        """Return 'equity' or 'crypto'."""

    @property
    @abstractmethod
    def is_24h_market(self) -> bool:
        """Whether this broker's market trades 24/7."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether we have an active connection."""

    @abstractmethod
    def connect(self) -> Dict[str, Any]:
        """Connect to the broker API and verify credentials.

        Returns dict with at least {'connected': bool}.
        """

    @abstractmethod
    def get_account_status(self) -> Dict[str, Any]:
        """Get account status including equity, cash, positions."""

    @abstractmethod
    def execute_decisions(
        self,
        decisions: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute trading decisions.

        Args:
            decisions: List of decision dicts with keys:
                ticker, action (BUY/SELL/HOLD), confidence,
                position_size_usd, price, reasons
            dry_run: If True, simulate only.

        Returns execution report.
        """

    @abstractmethod
    def get_positions_detail(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed position info.

        Returns {ticker: {qty, entry_price, current_price,
                 market_value, unrealized_plpc, side}}.
        """

    def get_positions_as_dict(self) -> Dict[str, float]:
        """Get positions as {ticker: market_value}."""
        detail = self.get_positions_detail()
        return {tic: info.get("market_value", 0) for tic, info in detail.items()}
