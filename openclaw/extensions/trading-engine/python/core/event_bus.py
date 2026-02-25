"""Async pub/sub event bus for multi-agent coordination.

Provides internal event routing between agents (MarketAgent, QuantAgent,
Synthesizer, etc.) and external WebSocket broadcast.  Supports both
fire-and-forget publish and request/reply with timeout.

Topics:
    market.regime   — regime change detected
    market.sector   — sector momentum update
    quant.rankings  — XGBoost cross-sectional rankings
    sentiment.score — FinBERT sentiment batch
    decision.signal — synthesized buy/sell/hold
    risk.check      — risk guard result
    order.execute   — order submitted / filled
    portfolio.update — portfolio state changed
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("trading-engine.event_bus")


@dataclass
class Event:
    """An immutable event envelope."""

    topic: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }


# Handler type: async callable that receives an Event
Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub event bus for agent coordination.

    Usage::

        bus = EventBus()

        # Subscribe
        async def on_regime(event: Event):
            print(event.payload)
        bus.subscribe("market.regime", on_regime)

        # Publish (fire-and-forget)
        await bus.publish("market.regime", {"state": "low_vol"})

        # Request/reply with timeout
        result = await bus.request("quant.rank", {"tickers": [...]}, timeout=30)
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._reply_futures: Dict[str, asyncio.Future] = {}
        self._event_log: deque = deque(maxlen=500)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register *handler* for events on *topic*."""
        self._handlers[topic].append(handler)
        logger.debug("Subscribed handler to '%s'", topic)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove *handler* from *topic* subscriptions."""
        handlers = self._handlers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, topic: str, payload: Dict[str, Any]) -> Event:
        """Publish an event to all subscribers.  Non-blocking."""
        event = Event(topic=topic, payload=payload)
        self._record(event)

        handlers = self._handlers.get(topic, [])
        if not handlers:
            logger.debug("No handlers for topic '%s'", topic)
            return event

        # Fire all handlers concurrently
        tasks = [asyncio.create_task(self._safe_call(h, event)) for h in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Resolve any pending request/reply futures
        reply_key = f"reply:{topic}:{event.event_id}"
        if reply_key in self._reply_futures:
            self._reply_futures[reply_key].set_result(payload)

        return event

    async def request(
        self, topic: str, payload: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Publish and wait for a reply on ``{topic}.reply``.

        The responder should call ``publish("{topic}.reply", result)`` to
        complete the request.  If no reply arrives within *timeout* seconds
        a ``TimeoutError`` is raised.
        """
        reply_topic = f"{topic}.reply"
        future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()

        async def _capture_reply(event: Event) -> None:
            if not future.done():
                future.set_result(event.payload)

        self.subscribe(reply_topic, _capture_reply)
        try:
            await self.publish(topic, payload)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning("Request to '%s' timed out after %.1fs", topic, timeout)
            raise
        finally:
            self.unsubscribe(reply_topic, _capture_reply)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_recent_events(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last *n* events as dicts."""
        return [e.to_dict() for e in self._event_log[-n:]]

    @property
    def topics(self) -> List[str]:
        """Return all topics that have at least one subscriber."""
        return list(self._handlers.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _safe_call(self, handler: Handler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:
            logger.exception(
                "Handler %s failed for event '%s'",
                getattr(handler, "__name__", handler),
                event.topic,
            )

    def _record(self, event: Event) -> None:
        self._event_log.append(event)
