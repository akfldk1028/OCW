"""WebSocket streaming manager for real-time trading events.

Manages connected WebSocket clients and broadcasts events from the EventBus
to all subscribers.  Clients can send subscribe/unsubscribe commands to
filter which event types they receive.

Event types broadcast:
    regime.update    — regime change detected
    sector.hot       — hot sector identified
    signal.buy       — buy signal generated
    signal.sell      — sell signal generated
    order.filled     — order executed
    risk.alert       — risk limit triggered
    portfolio.update — portfolio state changed
    agent.status     — agent pipeline status
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from core.event_bus import Event, EventBus

logger = logging.getLogger("trading-engine.ws_stream")


@dataclass
class ClientConnection:
    """Tracks a single WebSocket client and its subscriptions."""

    websocket: WebSocket
    connected_at: float = field(default_factory=time.time)
    subscribed_topics: Set[str] = field(default_factory=set)  # empty = all topics
    client_id: str = ""

    @property
    def is_filtered(self) -> bool:
        return len(self.subscribed_topics) > 0


class TradingStream:
    """WebSocket stream manager.

    Connects to the EventBus and relays events to connected WebSocket
    clients.  Supports per-client topic filtering.

    Usage::

        stream = TradingStream(event_bus)

        # In FastAPI:
        @app.websocket("/ws/stream")
        async def ws_endpoint(ws: WebSocket):
            await stream.connect(ws)
            try:
                while True:
                    data = await ws.receive_text()
                    await stream.handle_client_message(ws, data)
            except WebSocketDisconnect:
                stream.disconnect(ws)
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._clients: Dict[int, ClientConnection] = {}  # id(ws) → ClientConnection
        self._message_count = 0

        # Subscribe to all relevant EventBus topics
        for topic in [
            "market.regime",
            "market.sector",
            "market.candle_close",
            "market.significant_move",
            "market.funding_extreme",
            "market.oi_spike",
            "quant.rankings",
            "sentiment.score",
            "decision.signal",
            "risk.check",
            "order.execute",
            "portfolio.update",
            "agent.status",
        ]:
            self._event_bus.subscribe(topic, self._on_event)

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        client = ClientConnection(websocket=websocket)
        self._clients[id(websocket)] = client
        logger.info(
            "WebSocket client connected (total: %d)", len(self._clients)
        )

        # Send welcome message
        await self._send_to_client(
            websocket,
            {
                "type": "connected",
                "data": {
                    "message": "Connected to Trading Stream",
                    "total_clients": len(self._clients),
                    "available_topics": [
                        "regime.update",
                        "sector.hot",
                        "trade.alert",
                        "signal.buy",
                        "signal.sell",
                        "order.filled",
                        "risk.alert",
                        "portfolio.update",
                        "agent.status",
                    ],
                },
            },
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client."""
        ws_id = id(websocket)
        if ws_id in self._clients:
            del self._clients[ws_id]
        logger.info(
            "WebSocket client disconnected (total: %d)", len(self._clients)
        )

    # ------------------------------------------------------------------
    # Client messages (subscribe / unsubscribe)
    # ------------------------------------------------------------------

    async def handle_client_message(self, websocket: WebSocket, raw: str) -> None:
        """Process a message from a client.

        Supported commands::

            {"action": "subscribe", "topics": ["regime.update", "signal.buy"]}
            {"action": "unsubscribe", "topics": ["signal.sell"]}
            {"action": "ping"}
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_to_client(
                websocket, {"type": "error", "data": {"message": "Invalid JSON"}}
            )
            return

        action = msg.get("action", "")
        client = self._clients.get(id(websocket))
        if not client:
            return

        if action == "subscribe":
            topics = msg.get("topics", [])
            client.subscribed_topics.update(topics)
            await self._send_to_client(
                websocket,
                {
                    "type": "subscribed",
                    "data": {"topics": list(client.subscribed_topics)},
                },
            )

        elif action == "unsubscribe":
            topics = msg.get("topics", [])
            client.subscribed_topics -= set(topics)
            await self._send_to_client(
                websocket,
                {
                    "type": "unsubscribed",
                    "data": {"topics": list(client.subscribed_topics)},
                },
            )

        elif action == "ping":
            await self._send_to_client(
                websocket,
                {"type": "pong", "data": {"timestamp": time.time()}},
            )

        else:
            await self._send_to_client(
                websocket,
                {
                    "type": "error",
                    "data": {"message": f"Unknown action: {action}"},
                },
            )

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def _send_to_client(
        self, websocket: WebSocket, message: Dict[str, Any]
    ) -> None:
        """Send a JSON message to a single client, ignoring errors."""
        try:
            await websocket.send_json(message)
        except Exception:
            # Client probably disconnected; will be cleaned up later
            pass

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> int:
        """Broadcast an event to all matching clients.

        Returns the number of clients that received the message.
        """
        message = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "seq": self._message_count,
        }
        self._message_count += 1

        sent = 0
        disconnected: List[int] = []

        for ws_id, client in self._clients.items():
            # Apply topic filter
            if client.is_filtered and event_type not in client.subscribed_topics:
                continue

            try:
                await client.websocket.send_json(message)
                sent += 1
            except Exception:
                disconnected.append(ws_id)

        # Clean up dead connections
        for ws_id in disconnected:
            if ws_id in self._clients:
                del self._clients[ws_id]

        return sent

    # ------------------------------------------------------------------
    # EventBus bridge
    # ------------------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        """Bridge: EventBus event → WebSocket broadcast."""
        # Map EventBus topics to WS event types
        # market.candle_close / significant_move -> trade-alert (OpenClaw hook)
        # market.funding_extreme / oi_spike -> risk-guardian (OpenClaw hook)
        topic_map = {
            "market.regime": "regime.update",
            "market.sector": "sector.hot",
            "market.candle_close": "trade.alert",
            "market.significant_move": "trade.alert",
            "market.funding_extreme": "risk.alert",
            "market.oi_spike": "risk.alert",
            "quant.rankings": "quant.update",
            "sentiment.score": "sentiment.update",
            "decision.signal": self._decision_event_type,
            "risk.check": "risk.alert",
            "order.execute": "order.filled",
            "portfolio.update": "portfolio.update",
            "agent.status": "agent.status",
        }

        mapper = topic_map.get(event.topic)
        if mapper is None:
            return

        if callable(mapper):
            event_type = mapper(event.payload)
        else:
            event_type = mapper

        await self.broadcast(event_type, event.payload)

    @staticmethod
    def _decision_event_type(payload: Dict[str, Any]) -> str:
        """Map decision payload to signal.buy or signal.sell."""
        action = payload.get("action", "").upper()
        if action in ("BUY", "STRONG_BUY"):
            return "signal.buy"
        elif action in ("SELL", "STRONG_SELL"):
            return "signal.sell"
        return "signal.hold"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return stream status for monitoring."""
        return {
            "connected_clients": len(self._clients),
            "total_messages_sent": self._message_count,
            "clients": [
                {
                    "connected_at": c.connected_at,
                    "subscribed_topics": list(c.subscribed_topics),
                }
                for c in self._clients.values()
            ],
        }
