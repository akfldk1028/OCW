"""Live position tracker with agent-based re-evaluation.

Two layers:
1. Safety layer (30s): Price polling + rule-based TP/SL (fast, always on)
2. Agent layer (3-5min): Full context re-evaluation per position
   - Regime change since entry?
   - Momentum reversal?
   - Cross-asset correlation breakdown?
   - Sentiment shift?
   -> Weighted vote: HOLD / EXIT / ADD

This is what makes it different from hardcoded if/else rules.
The agent THINKS about each position with full market context.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import CRYPTO_RISK_CONFIG, RISK_CONFIG, TRANSACTION_COSTS

logger = logging.getLogger("trading-engine.position_tracker")


@dataclass
class TrackedPosition:
    """Rich context for a single tracked position."""

    ticker: str
    broker_name: str          # "binance", "alpaca", "kis"
    market_type: str          # "crypto" or "equity"
    entry_price: float
    entry_time: datetime
    qty: float
    side: str = "long"

    # Live state (updated by fast loop)
    current_price: float = 0.0
    trailing_high: float = 0.0
    last_update: float = 0.0

    # Context at entry (captured once)
    entry_regime: str = "unknown"
    entry_momentum_z: float = 0.0

    # Agent evaluation state (updated by agent loop)
    agent_verdict: str = "HOLD"       # HOLD / EXIT / ADD
    agent_confidence: float = 0.0
    agent_reasons: List[str] = field(default_factory=list)
    last_agent_eval: float = 0.0

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0 or self.current_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def held_seconds(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds()

    @property
    def held_hours(self) -> float:
        return self.held_seconds / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "broker": self.broker_name,
            "market_type": self.market_type,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl_pct": round(self.pnl_pct, 4),
            "qty": self.qty,
            "market_value": round(self.market_value, 2),
            "trailing_high": self.trailing_high,
            "held_hours": round(self.held_hours, 1),
            "entry_regime": self.entry_regime,
            "agent_verdict": self.agent_verdict,
            "agent_confidence": round(self.agent_confidence, 3),
            "agent_reasons": self.agent_reasons,
        }


# Type for the agent evaluation callback
AgentEvalFn = Callable[
    [TrackedPosition, Dict[str, Any]],  # position, market_context
    Dict[str, Any],                      # {"verdict": str, "confidence": float, "reasons": list}
]


class PositionTracker:
    """Live position monitor with two-layer evaluation.

    Layer 1 — Safety (30s polling):
        Rule-based TP/SL/trailing stop. Fast, no ML, always reliable.

    Layer 2 — Agent (3-5min):
        Full agent pipeline re-evaluation with market context.
        This is the intelligent layer — regime, momentum, correlation,
        sentiment all factor into hold/exit/add decisions.

    Usage::

        tracker = PositionTracker(event_bus=bus)
        tracker.set_agent_evaluator(my_eval_fn)
        tracker.set_price_fetcher("binance", binance_price_fn)

        # Register a position after BUY execution
        tracker.track("BTC/USDT", broker="binance", entry_price=60000, qty=0.01)

        # Start monitoring (runs as async background task)
        task = asyncio.create_task(tracker.run())

        # Stop
        tracker.stop()
    """

    def __init__(
        self,
        event_bus: Any,
        safety_interval: float = 30.0,     # seconds between price checks
        agent_interval: float = 180.0,      # seconds between agent evaluations
    ) -> None:
        self._bus = event_bus
        self._safety_interval = safety_interval
        self._agent_interval = agent_interval

        self._positions: Dict[str, TrackedPosition] = {}  # ticker -> position
        self._running = False

        # Pluggable: broker price fetchers {broker_name: async fn(ticker) -> float}
        self._price_fetchers: Dict[str, Callable] = {}

        # Pluggable: agent evaluation function
        self._agent_eval: Optional[AgentEvalFn] = None

        # Pluggable: market context provider (regime, correlations, etc.)
        self._context_provider: Optional[Callable] = None

        # Pluggable: exit executor {broker_name: async fn(decision) -> result}
        self._exit_executors: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_price_fetcher(self, broker_name: str, fn: Callable) -> None:
        """Register a price fetcher for a broker.

        fn signature: (ticker: str) -> float
        """
        self._price_fetchers[broker_name] = fn

    def set_agent_evaluator(self, fn: AgentEvalFn) -> None:
        """Register the agent evaluation function.

        fn signature: (position: TrackedPosition, context: dict) -> {
            "verdict": "HOLD"|"EXIT"|"ADD",
            "confidence": float,
            "reasons": ["regime changed", "momentum reversed", ...]
        }
        """
        self._agent_eval = fn

    def set_context_provider(self, fn: Callable) -> None:
        """Register market context provider.

        fn signature: () -> {"regime": str, "btc_price": float, ...}
        """
        self._context_provider = fn

    def set_exit_executor(self, broker_name: str, fn: Callable) -> None:
        """Register exit execution function for a broker.

        fn signature: async (decision: dict) -> result
        """
        self._exit_executors[broker_name] = fn

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def track(
        self,
        ticker: str,
        broker: str,
        entry_price: float,
        qty: float,
        market_type: str = "crypto",
        regime: str = "unknown",
        momentum_z: float = 0.0,
        entry_time: Optional[datetime] = None,
    ) -> TrackedPosition:
        """Start tracking a new position after BUY execution."""
        pos = TrackedPosition(
            ticker=ticker,
            broker_name=broker,
            market_type=market_type,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(),
            qty=qty,
            current_price=entry_price,
            trailing_high=entry_price,
            entry_regime=regime,
            entry_momentum_z=momentum_z,
        )
        self._positions[ticker] = pos
        logger.info(
            "[tracker] Now tracking %s: %.4f x %.6f via %s (regime=%s)",
            ticker, entry_price, qty, broker, regime,
        )
        return pos

    def untrack(self, ticker: str) -> None:
        """Stop tracking a position (after SELL execution)."""
        pos = self._positions.pop(ticker, None)
        if pos:
            logger.info(
                "[tracker] Stopped tracking %s (held %.1fh, pnl=%+.2f%%)",
                ticker, pos.held_hours, pos.pnl_pct * 100,
            )

    def get_position(self, ticker: str) -> Optional[TrackedPosition]:
        return self._positions.get(ticker)

    @property
    def active_positions(self) -> Dict[str, TrackedPosition]:
        return dict(self._positions)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main monitoring loop. Run as asyncio.create_task(tracker.run())."""
        self._running = True
        logger.info("[tracker] Started — safety@%.0fs, agent@%.0fs",
                     self._safety_interval, self._agent_interval)

        last_agent_run = 0.0

        try:
            while self._running:
                if not self._positions:
                    await asyncio.sleep(self._safety_interval)
                    continue

                # Layer 1: Safety check (every iteration)
                await self._safety_check()

                # Layer 2: Agent evaluation (every agent_interval)
                now = time.time()
                if now - last_agent_run >= self._agent_interval:
                    await self._agent_check()
                    last_agent_run = now

                await asyncio.sleep(self._safety_interval)

        except asyncio.CancelledError:
            logger.info("[tracker] Shutting down")
        except Exception as exc:
            logger.error("[tracker] Fatal error: %s", exc, exc_info=True)
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the tracker to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Layer 1: Safety (rule-based TP/SL/trailing)
    # ------------------------------------------------------------------

    async def _safety_check(self) -> None:
        """Fast price update + rule-based safety checks."""
        exits = []

        for ticker, pos in list(self._positions.items()):
            # Fetch latest price
            price = await self._fetch_price(pos)
            if price is None or price <= 0:
                continue

            prev_price = pos.current_price
            pos.current_price = price
            pos.last_update = time.time()

            # Update trailing high
            if price > pos.trailing_high:
                pos.trailing_high = price

            # Check safety rules
            reason = self._check_safety_rules(pos)
            if reason:
                exits.append((ticker, pos, reason))
                await self._publish_alert(pos, reason)

            # Publish price update if significant move (>0.5%)
            if prev_price > 0:
                move_pct = abs(price - prev_price) / prev_price
                if move_pct > 0.005:
                    await self._publish_update(pos)

        # Execute exits
        for ticker, pos, reason in exits:
            await self._execute_exit(pos, reason, source="safety")

    def _check_safety_rules(self, pos: TrackedPosition) -> Optional[str]:
        """Rule-based safety checks. Returns reason string or None.

        Safety layer is the last line of defense — these thresholds should be
        wider than Claude's strategy to avoid premature exits.
        """
        cfg = CRYPTO_RISK_CONFIG if pos.market_type == "crypto" else RISK_CONFIG
        pnl = pos.pnl_pct

        # Hard stop loss (unconditional — protects against flash crash)
        sl = cfg.get("stop_loss_pct", -0.04)
        if pnl <= sl:
            return f"stop_loss ({pnl:+.1%} <= {sl:+.1%})"

        # Trailing stop — use config trail_pct (consistent with Claude's strategy)
        # trail_pct from ACTIVE_BLEND_CONFIG (8-12%), not atr_multiplier*0.01 (2.5%)
        try:
            from crypto_config import ACTIVE_BLEND_CONFIG
            trail_pct = ACTIVE_BLEND_CONFIG.get("trail_pct", 0.12)
            trail_activation = ACTIVE_BLEND_CONFIG.get("trail_activation_pct", 0.08)
        except ImportError:
            trail_pct = cfg.get("trail_pct", 0.12)
            trail_activation = cfg.get("trail_activation_pct", 0.08)

        if pos.trailing_high > pos.entry_price:
            peak_pnl = (pos.trailing_high - pos.entry_price) / pos.entry_price
            # Only activate trailing stop after sufficient profit
            if peak_pnl >= trail_activation:
                trail_stop = pos.trailing_high * (1 - trail_pct)
                if pos.current_price <= trail_stop:
                    drop = (pos.current_price - pos.trailing_high) / pos.trailing_high
                    return f"trailing_stop ({drop:+.1%} from high {pos.trailing_high:.2f})"

            # Profit protection: don't let +2% winners become losers
            if peak_pnl >= 0.02 and pnl < 0.005:
                return f"profit_protect (peak={peak_pnl:+.1%}, now={pnl:+.1%})"

        # Time stop: swing positions shouldn't overstay
        max_hold = cfg.get("max_hold_hours", 48)
        if pos.held_hours > max_hold and pnl < 0.01:
            return f"time_stop (held {pos.held_hours:.0f}h, pnl={pnl:+.1%})"

        return None

    # ------------------------------------------------------------------
    # Layer 2: Agent evaluation (the intelligent part)
    # ------------------------------------------------------------------

    async def _agent_check(self) -> None:
        """Re-evaluate all positions through the agent pipeline.

        This is what makes it agent-based, not rule-based:
        - Current regime vs entry regime
        - Momentum direction change
        - Cross-asset correlation signals
        - Volume/sentiment anomalies
        """
        if self._agent_eval is None:
            return

        # Get current market context
        context = {}
        if self._context_provider is not None:
            try:
                ctx = self._context_provider()
                if asyncio.iscoroutine(ctx):
                    ctx = await ctx
                context = ctx or {}
            except Exception as exc:
                logger.warning("[tracker] Context provider failed: %s", exc)

        exits = []
        for ticker, pos in list(self._positions.items()):
            try:
                result = self._agent_eval(pos, context)
                if asyncio.iscoroutine(result):
                    result = await result

                verdict = result.get("verdict", "HOLD")
                confidence = result.get("confidence", 0.0)
                reasons = result.get("reasons", [])

                pos.agent_verdict = verdict
                pos.agent_confidence = confidence
                pos.agent_reasons = reasons
                pos.last_agent_eval = time.time()

                if verdict == "EXIT" and confidence > 0.3:
                    exits.append((ticker, pos, reasons))

                # Publish agent evaluation event
                await self._bus.publish("position.agent_eval", {
                    "ticker": ticker,
                    "verdict": verdict,
                    "confidence": confidence,
                    "reasons": reasons,
                    "pnl_pct": pos.pnl_pct,
                    "held_hours": pos.held_hours,
                })

                logger.info(
                    "[tracker/agent] %s: verdict=%s conf=%.2f pnl=%+.1f%% reasons=%s",
                    ticker, verdict, confidence, pos.pnl_pct * 100, reasons[:2],
                )

            except Exception as exc:
                logger.warning("[tracker/agent] Eval failed for %s: %s", ticker, exc)

        # Execute agent-recommended exits
        for ticker, pos, reasons in exits:
            reason_str = f"agent_exit: {', '.join(reasons[:3])}"
            await self._execute_exit(pos, reason_str, source="agent")

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    async def _fetch_price(self, pos: TrackedPosition) -> Optional[float]:
        """Fetch current price from the appropriate broker."""
        fetcher = self._price_fetchers.get(pos.broker_name)
        if fetcher is None:
            return None

        try:
            result = fetcher(pos.ticker)
            if asyncio.iscoroutine(result):
                result = await result
            return float(result) if result else None
        except Exception as exc:
            logger.debug("[tracker] Price fetch failed for %s: %s", pos.ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    async def _execute_exit(
        self, pos: TrackedPosition, reason: str, source: str = "safety"
    ) -> None:
        """Execute an exit and clean up the tracked position."""
        decision = {
            "ticker": pos.ticker,
            "action": "SELL",
            "confidence": -1.0,
            "position_size_usd": pos.market_value,
            "price": pos.current_price,
            "reasons": [f"{source}_{reason}"],
        }

        logger.info(
            "[tracker/%s] EXIT %s: %s (pnl=%+.2f%%, held=%.1fh)",
            source, pos.ticker, reason, pos.pnl_pct * 100, pos.held_hours,
        )

        # Execute via broker
        executor = self._exit_executors.get(pos.broker_name)
        if executor:
            try:
                result = executor(decision)
                if asyncio.iscoroutine(result):
                    result = await result
                logger.info("[tracker] Exit executed for %s: %s", pos.ticker, result)
            except Exception as exc:
                logger.error("[tracker] Exit execution failed for %s: %s", pos.ticker, exc)

        # Publish exit event
        await self._bus.publish("position.exit", {
            "ticker": pos.ticker,
            "source": source,
            "reason": reason,
            "entry_price": pos.entry_price,
            "exit_price": pos.current_price,
            "pnl_pct": pos.pnl_pct,
            "held_hours": pos.held_hours,
            "broker": pos.broker_name,
        })

        # Remove from tracking
        self.untrack(pos.ticker)

    # ------------------------------------------------------------------
    # EventBus publishing
    # ------------------------------------------------------------------

    async def _publish_update(self, pos: TrackedPosition) -> None:
        await self._bus.publish("position.update", pos.to_dict())

    async def _publish_alert(self, pos: TrackedPosition, reason: str) -> None:
        await self._bus.publish("position.alert", {
            **pos.to_dict(),
            "alert_reason": reason,
        })

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "active_positions": len(self._positions),
            "safety_interval_s": self._safety_interval,
            "agent_interval_s": self._agent_interval,
            "has_agent_evaluator": self._agent_eval is not None,
            "positions": {t: p.to_dict() for t, p in self._positions.items()},
        }
