"""Broker factory — create and manage multiple broker instances.

Usage::

    from broker_factory import BrokerRegistry

    registry = BrokerRegistry()
    registry.register("alpaca", AlpacaBroker())
    registry.register("binance", BinanceBroker(paper=True))

    # Connect all enabled brokers
    registry.connect_all()

    # Get specific broker
    binance = registry.get("binance")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from brokers.base import BrokerBase

logger = logging.getLogger(__name__)


class BrokerRegistry:
    """Registry for multiple broker instances."""

    def __init__(self) -> None:
        self._brokers: Dict[str, BrokerBase] = {}

    def register(self, name: str, broker: BrokerBase) -> None:
        """Register a broker instance."""
        self._brokers[name] = broker
        logger.info("Broker registered: %s (%s, %s)",
                     name, broker.market_type,
                     "24h" if broker.is_24h_market else "market hours")

    def get(self, name: str) -> Optional[BrokerBase]:
        """Get a registered broker by name."""
        return self._brokers.get(name)

    def get_by_market(self, market_type: str) -> List[BrokerBase]:
        """Get all brokers for a market type ('equity' or 'crypto')."""
        return [b for b in self._brokers.values() if b.market_type == market_type]

    def connect_all(self) -> Dict[str, Dict[str, Any]]:
        """Connect all registered brokers. Returns {name: connect_result}."""
        results = {}
        for name, broker in self._brokers.items():
            result = broker.connect()
            results[name] = result
            if result.get("connected"):
                logger.info("Broker %s connected", name)
            else:
                logger.warning("Broker %s failed: %s", name, result.get("error", "unknown"))
        return results

    def status_all(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered brokers."""
        statuses = {}
        for name, broker in self._brokers.items():
            if broker.is_connected:
                statuses[name] = broker.get_account_status()
            else:
                statuses[name] = {
                    "connected": False,
                    "market_type": broker.market_type,
                }
        return statuses

    @property
    def names(self) -> List[str]:
        return list(self._brokers.keys())

    @property
    def connected_brokers(self) -> Dict[str, BrokerBase]:
        return {n: b for n, b in self._brokers.items() if b.is_connected}

    def __contains__(self, name: str) -> bool:
        return name in self._brokers

    def __len__(self) -> int:
        return len(self._brokers)
