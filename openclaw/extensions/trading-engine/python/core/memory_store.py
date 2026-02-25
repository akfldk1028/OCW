"""Trading memory via Neo4j — cross-session learning for Claude Agent.

Stores trade outcomes, regime insights, and pattern observations in Neo4j.
Claude sees historical insights in the "Historical Insights" section of its prompt.

Uses neo4j Python driver directly (not Graphiti MCP) for 24/7 daemon compatibility.
Connection: bolt://localhost:7687, neo4j/demodemo (Docker container).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading-engine.memory_store")

_NEO4J_AVAILABLE: Optional[bool] = None


def _check_neo4j() -> bool:
    global _NEO4J_AVAILABLE
    if _NEO4J_AVAILABLE is None:
        try:
            import neo4j  # noqa: F401
            _NEO4J_AVAILABLE = True
        except ImportError:
            _NEO4J_AVAILABLE = False
            logger.info("[memory] neo4j driver not installed — memory disabled")
    return _NEO4J_AVAILABLE


class TradingMemory:
    """Trade memory via Neo4j for cross-session learning."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "demodemo",
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    def _ensure_driver(self) -> bool:
        """Lazy-init Neo4j driver."""
        if self._driver is not None:
            return True
        if not _check_neo4j():
            return False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            self._ensure_schema()
            logger.info("[memory] Connected to Neo4j at %s", self._uri)
            return True
        except Exception as exc:
            logger.warning("[memory] Neo4j connection failed: %s", exc)
            self._driver = None
            return False

    def _ensure_schema(self) -> None:
        """Create indexes/constraints on first connect."""
        with self._driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Trade) ON (t.ticker)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (i:Insight) ON (i.regime)")

    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Store a completed trade in Neo4j."""
        if not self._ensure_driver():
            return
        try:
            with self._driver.session() as session:
                session.run(
                    """
                    CREATE (t:Trade {
                        ticker: $ticker,
                        action: $action,
                        entry_price: $entry_price,
                        exit_price: $exit_price,
                        pnl_pct: $pnl_pct,
                        held_hours: $held_hours,
                        regime: $regime,
                        timestamp: $timestamp
                    })
                    """,
                    ticker=trade_data.get("ticker", ""),
                    action=trade_data.get("action", "SELL"),
                    entry_price=trade_data.get("entry_price", 0),
                    exit_price=trade_data.get("exit_price", 0),
                    pnl_pct=trade_data.get("pnl_pct", 0),
                    held_hours=trade_data.get("held_hours", 0),
                    regime=trade_data.get("regime", "unknown"),
                    timestamp=time.time(),
                )
        except Exception as exc:
            logger.debug("[memory] Failed to record trade: %s", exc)

    def record_insight(self, regime: str, insight: str) -> None:
        """Store a regime insight (from Claude's learning_note or reflection)."""
        if not self._ensure_driver():
            return
        try:
            with self._driver.session() as session:
                session.run(
                    """
                    CREATE (i:Insight {
                        regime: $regime,
                        content: $content,
                        timestamp: $timestamp
                    })
                    """,
                    regime=regime,
                    content=insight[:500],
                    timestamp=time.time(),
                )
        except Exception as exc:
            logger.debug("[memory] Failed to record insight: %s", exc)

    def get_regime_insights(self, regime: str, limit: int = 3) -> str:
        """Get recent insights for a regime."""
        if not self._ensure_driver():
            return ""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (i:Insight {regime: $regime})
                    RETURN i.content AS content
                    ORDER BY i.timestamp DESC
                    LIMIT $limit
                    """,
                    regime=regime,
                    limit=limit,
                )
                insights = [r["content"] for r in result]
                return " | ".join(insights) if insights else ""
        except Exception as exc:
            logger.debug("[memory] Failed to get insights: %s", exc)
            return ""

    def get_ticker_stats(self, ticker: str) -> str:
        """Get historical performance summary for a ticker."""
        if not self._ensure_driver():
            return ""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (t:Trade {ticker: $ticker})
                    RETURN
                        count(t) AS total,
                        sum(CASE WHEN t.pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                        avg(t.pnl_pct) AS avg_pnl,
                        avg(t.held_hours) AS avg_hours
                    """,
                    ticker=ticker,
                )
                row = result.single()
                if not row or row["total"] == 0:
                    return ""
                return (
                    f"{ticker}: {row['total']} trades, "
                    f"WR={row['wins']}/{row['total']}, "
                    f"avg PnL={row['avg_pnl']*100:+.2f}%, "
                    f"avg hold={row['avg_hours']:.0f}h"
                )
        except Exception as exc:
            logger.debug("[memory] Failed to get ticker stats: %s", exc)
            return ""

    def build_historical_insights(self, regime: str, tickers: List[str]) -> str:
        """Build combined historical insights string for Claude prompt."""
        parts = []
        regime_insights = self.get_regime_insights(regime)
        if regime_insights:
            parts.append(f"Regime({regime}): {regime_insights}")
        for ticker in tickers[:3]:  # limit to top 3
            stats = self.get_ticker_stats(ticker)
            if stats:
                parts.append(stats)
        return " | ".join(parts) if parts else ""

    def close(self) -> None:
        """Close Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None
