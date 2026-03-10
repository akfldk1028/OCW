"""Scalping async runner — signal-driven with independent infrastructure.

Architecture:
    1m WS candle close → ScalpSignalEngine.evaluate()
        |score| >= 0.85 → instant execution
        0.70 <= |score| < 0.85 + 2 signals → Claude confirmation
        |score| < 0.70 → ignore

    Position management:
        - ATR-based TP/SL
        - 30-min time stop
        - Trailing stop after +0.3%
        - Daily PnL circuit breaker (-2% or +1%)

Reuses: BinanceBroker, OHLCVStore, MarketListener, PositionTracker, EventBus
Independent instances: everything except broker (stateless).
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_SCALPING_DIR = Path(__file__).resolve().parent
_BINANCE_DIR = _SCALPING_DIR.parent
_ENGINE_DIR = _BINANCE_DIR.parent
for _p in (_BINANCE_DIR, _ENGINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from brokers.binance import BinanceBroker
from core.event_bus import EventBus
from core.market_listener import MarketListener
from core.ohlcv_store import OHLCVStore
from core.position_tracker import PositionTracker
from analysis.regime_detector_crypto import CryptoRegimeDetector

from scalp_config import (
    SCALP_TICKERS,
    SCALP_EVENT_CONFIG,
    SCALP_RISK_CONFIG,
    SCALP_SIGNAL_CONFIG,
    SCALP_SIGNAL_WEIGHTS,
    SCALP_EXECUTION_CONFIG,
    SCALP_REGIME_FILTER,
    SCALP_DATA_DIR,
    SCALP_LOGS_DIR,
)
from scalp_signal import ScalpSignalEngine, calc_atr

logger = logging.getLogger(__name__)


class ScalpRunner:
    """Async scalping runner with signal-driven entries."""

    def __init__(
        self,
        broker: BinanceBroker,
        leverage: int = 3,
        initial_balance: float = 0.0,
        derivatives_monitor=None,
    ) -> None:
        self.broker = broker
        self.leverage = leverage
        self._initial_balance = initial_balance or 10_000.0
        self._derivatives_monitor = derivatives_monitor
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Independent event bus
        self.event_bus = EventBus()

        # Independent OHLCV store (maxlen=3000 for 1m bars ~ 2 days)
        self.ohlcv_store = OHLCVStore(maxlen=3000)

        # Signal engine
        self.signal_engine = ScalpSignalEngine(
            config=SCALP_SIGNAL_CONFIG,
            weights=SCALP_SIGNAL_WEIGHTS,
            execution=SCALP_EXECUTION_CONFIG,
        )

        # Regime detector (shared — stateless, cached daily)
        self.regime_detector = CryptoRegimeDetector()

        # Market listener (1m/3m/5m WS)
        ecfg = SCALP_EVENT_CONFIG
        self.market_listener = MarketListener(
            event_bus=self.event_bus,
            tickers=SCALP_TICKERS,
            kline_intervals=ecfg["kline_intervals"],
            primary_interval=ecfg["primary_interval"],
            significant_move_pct=ecfg["significant_move_pct"],
            min_decision_gap=ecfg["min_decision_gap"],
            market=broker.market,
            testnet=broker.paper,
            ohlcv_store=self.ohlcv_store,
        )

        # Position tracker (fast: 5s safety, 60s agent)
        self.position_tracker = PositionTracker(
            event_bus=self.event_bus,
            safety_interval=5.0,
            agent_interval=60.0,
        )

        # State
        self._last_tick_prices: Dict[str, float] = {}
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_reset_date: str = ""
        self._state_file = SCALP_DATA_DIR / "scalp_state.json"

        # Logging
        self._trades_csv = SCALP_LOGS_DIR / "scalp_trades.csv"
        self._signals_log = SCALP_LOGS_DIR / "scalp_signals.jsonl"
        self._init_trade_csv()

        # Setup
        self._setup_tracker()
        self._setup_event_subscriptions()
        if leverage > 1 and broker.market == "future":
            self._setup_leverage()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_leverage(self) -> None:
        for tic in SCALP_TICKERS:
            try:
                self.broker.set_margin_mode(tic, "isolated")
                self.broker.set_leverage(tic, self.leverage)
            except Exception as exc:
                logger.warning("[scalp] Leverage setup failed for %s: %s", tic, exc)
        logger.info("[scalp] Leverage set to %dx for %d tickers", self.leverage, len(SCALP_TICKERS))

    def _setup_event_subscriptions(self) -> None:
        async def _on_tick(event):
            try:
                await self._handle_tick(event.payload)
            except Exception as exc:
                logger.error("[scalp/tick] %s", exc)

        self.event_bus.subscribe("market.tick", _on_tick)

    def _setup_tracker(self) -> None:
        dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")

        async def price_fetcher(ticker: str) -> float:
            ws_price = self._last_tick_prices.get(ticker, 0.0)
            if ws_price > 0:
                return ws_price
            df = self.ohlcv_store.get_close_df([ticker], "1m")
            if not df.empty and ticker in df.columns:
                return float(df[ticker].iloc[-1])
            return 0.0

        def scalp_evaluator(pos, context):
            """Rule-based evaluator for scalp positions."""
            cfg = SCALP_RISK_CONFIG
            pnl = pos.pnl_pct
            held_min = pos.held_seconds / 60

            # Time stop
            max_hold = cfg.get("max_hold_minutes", 30)
            if held_min > max_hold:
                return {"verdict": "EXIT", "confidence": 0.9, "reasons": [f"time_stop ({held_min:.0f}min)"]}

            # Dynamic TP/SL from ATR
            bars = self.ohlcv_store.get_bars(pos.ticker, "1m", 20)
            atr = calc_atr(bars, 14) if len(bars) >= 15 else 0
            current_price = pos.current_price or pos.entry_price

            if atr > 0 and current_price > 0:
                atr_pct = atr / current_price
                tp = max(cfg["take_profit_pct"], atr_pct * 1.5)
                sl = min(cfg["stop_loss_pct"], -atr_pct * 2.0)
            else:
                tp = cfg["take_profit_pct"]
                sl = cfg["stop_loss_pct"]

            if pnl >= tp:
                return {"verdict": "EXIT", "confidence": 0.95, "reasons": [f"take_profit ({pnl:+.2%} >= {tp:+.2%})"]}
            if pnl <= sl:
                return {"verdict": "EXIT", "confidence": 0.95, "reasons": [f"stop_loss ({pnl:+.2%} <= {sl:+.2%})"]}

            # Trailing stop
            trail_act = cfg.get("trailing_activate_pct", 0.003)
            trail_width = cfg.get("trailing_width_pct", 0.002)
            if pos.trailing_high > pos.entry_price:
                peak_pnl = (pos.trailing_high - pos.entry_price) / pos.entry_price
                if peak_pnl >= trail_act and pnl < peak_pnl - trail_width:
                    return {"verdict": "EXIT", "confidence": 0.8,
                            "reasons": [f"trailing_stop (peak={peak_pnl:+.2%}, now={pnl:+.2%})"]}

            return {"verdict": "HOLD", "confidence": 0.5, "reasons": ["within_bounds"]}

        async def exit_executor(decision):
            return await asyncio.to_thread(
                self.broker.execute_decisions, [decision], dry_run
            )

        self.position_tracker.set_price_fetcher("binance", price_fetcher)
        self.position_tracker.set_agent_evaluator(scalp_evaluator)
        self.position_tracker.set_exit_executor("binance", exit_executor)

        # Track exits for daily PnL
        async def _on_exit(event):
            data = event.payload
            pnl_pct = data.get("pnl_pct", 0)
            self._daily_pnl += pnl_pct * SCALP_RISK_CONFIG.get("position_size_pct", 0.10)
            self._daily_trades += 1

            self._log_trade(
                action="SELL",
                ticker=data.get("ticker", ""),
                price=data.get("exit_price", 0),
                qty=0,
                value_usd=0,
                pnl_pct=pnl_pct,
                held_minutes=data.get("held_hours", 0) * 60,
                reason=data.get("reason", ""),
                source=data.get("source", "safety"),
            )
            logger.info("[scalp] Daily PnL: %+.3f%% (%d trades)", self._daily_pnl * 100, self._daily_trades)

        self.event_bus.subscribe("position.exit", _on_exit)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all async loops."""
        logger.info("=" * 60)
        logger.info("[scalp] Starting Scalping Runner")
        logger.info("[scalp] Tickers: %s", SCALP_TICKERS)
        logger.info("[scalp] Leverage: %dx, TP: %s, SL: %s",
                     self.leverage,
                     SCALP_RISK_CONFIG["take_profit_pct"],
                     SCALP_RISK_CONFIG["stop_loss_pct"])
        logger.info("=" * 60)

        # Bootstrap OHLCV
        await self._bootstrap_ohlcv()

        # Launch tasks
        tasks = [
            asyncio.create_task(self.market_listener.run(), name="scalp_ws"),
            asyncio.create_task(self.position_tracker.run(), name="scalp_tracker"),
        ]
        if self._derivatives_monitor:
            tasks.append(
                asyncio.create_task(self._derivatives_monitor.run(), name="scalp_deriv")
            )

        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            self.market_listener.stop()
            self.position_tracker.stop()
            if self._derivatives_monitor:
                self._derivatives_monitor.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("[scalp] Shutdown complete")

    def stop(self) -> None:
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Tick handler → Signal evaluation
    # ------------------------------------------------------------------

    async def _handle_tick(self, payload: dict) -> None:
        """1m candle close → evaluate signals → trade."""
        ticker = payload.get("ticker", "")
        price = payload.get("price", 0)
        interval = payload.get("interval", "")
        is_closed = payload.get("is_closed", False)

        if price <= 0:
            return

        self._last_tick_prices[ticker] = price

        # Only evaluate on primary (1m) candle close
        primary = SCALP_EVENT_CONFIG.get("primary_interval", "1m")
        if interval != primary or not is_closed:
            return

        # Daily reset check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_reset_date = today
            logger.info("[scalp] Daily counters reset")

        # Circuit breaker check
        if self._check_circuit_breaker():
            return

        # Check max simultaneous positions
        if len(self.position_tracker.active_positions) >= SCALP_RISK_CONFIG.get("max_simultaneous", 3):
            return

        # Skip if already have position in this ticker
        if self.position_tracker.get_position(ticker):
            return

        # Get bars
        bars_1m = self.ohlcv_store.get_bars(ticker, "1m", 100)
        bars_3m = self.ohlcv_store.get_bars(ticker, "3m", 50)
        bars_5m = self.ohlcv_store.get_bars(ticker, "5m", 30)

        if len(bars_1m) < 25:
            return

        # Regime filter
        regime_label, regime_scale = self._get_regime_filter()

        # Derivatives context
        deriv_ctx = {}
        if self._derivatives_monitor:
            deriv_ctx = self._derivatives_monitor.get_context()

        # Evaluate signals
        result = self.signal_engine.evaluate(
            ticker=ticker,
            bars_1m=bars_1m,
            bars_3m=bars_3m,
            bars_5m=bars_5m,
            derivatives_ctx=deriv_ctx,
            regime_label=regime_label,
            regime_scale=regime_scale,
        )

        # Log signal to JSONL
        self._log_signal(result)

        # Execute based on action
        if result.action == "instant":
            await self._execute_signal(result)
        elif result.action == "claude_confirm":
            # For now, execute with reduced size (Claude confirmation TBD)
            logger.info("[scalp] Claude confirm zone — executing with reduced size")
            result.regime_scale *= 0.7  # 30% size reduction without Claude confirmation
            await self._execute_signal(result)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_signal(self, result) -> None:
        """Execute a scalping trade based on signal result."""
        if self._lock.locked():
            return
        async with self._lock:
            ticker = result.ticker
            price = self._last_tick_prices.get(ticker, 0)
            if price <= 0:
                return

            dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
            cfg = SCALP_RISK_CONFIG

            # Position sizing
            pv = self._initial_balance  # simplified — use initial balance
            size_pct = cfg.get("position_size_pct", 0.10) * result.regime_scale
            position_usd = pv * size_pct * self.leverage
            qty = position_usd / price

            # Direction
            action = "BUY" if result.direction == "long" else "SELL"
            reasons = [s.reason for s in result.signals if abs(s.score) > 0.1]

            decision = {
                "action": action,
                "ticker": ticker,
                "confidence": abs(result.score),
                "position_size_usd": position_usd,
                "price": price,
                "reasons": reasons,
            }

            logger.info(
                "[scalp] EXECUTE %s %s @ $%.2f (size=$%.2f, score=%+.3f, regime_scale=%.1f)",
                action, ticker, price, position_usd, result.score, result.regime_scale,
            )

            # Execute via broker
            try:
                exec_result = await asyncio.to_thread(
                    self.broker.execute_decisions, [decision], dry_run
                )
                logger.info("[scalp] Execution result: %s", exec_result)
            except Exception as exc:
                logger.error("[scalp] Execution failed: %s", exc)
                return

            # Track position
            self.position_tracker.track(
                ticker=ticker,
                broker="binance",
                entry_price=price,
                qty=qty,
                market_type="crypto",
                regime=self._get_regime_filter()[0],
            )

            # Log trade
            self._log_trade(
                action=action,
                ticker=ticker,
                price=price,
                qty=qty,
                value_usd=position_usd,
                reason=f"signal({result.score:+.3f})",
                source="signal",
                confidence=abs(result.score),
            )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self) -> bool:
        """Check daily PnL limits."""
        cfg = SCALP_RISK_CONFIG
        daily_loss = cfg.get("daily_loss_limit_pct", -0.02)
        daily_target = cfg.get("daily_target_pct", 0.01)

        if self._daily_pnl <= daily_loss:
            logger.warning("[scalp] CIRCUIT BREAKER: daily loss limit hit (%+.3f%%)", self._daily_pnl * 100)
            return True
        if self._daily_pnl >= daily_target:
            logger.info("[scalp] Daily target reached (%+.3f%%), stopping", self._daily_pnl * 100)
            return True
        return False

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------

    def _get_regime_filter(self) -> tuple:
        """Returns (regime_label, position_scale)."""
        try:
            result = self.regime_detector.detect()
            label = result.get("regime_label", "low_volatility")
        except Exception:
            label = "low_volatility"

        scale = SCALP_REGIME_FILTER.get(label, 1.0)
        if scale == "skip":
            return label, 0.0
        return label, float(scale)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_ohlcv(self) -> None:
        """One-time REST call to load historical bars for 1m/3m/5m."""
        intervals = SCALP_EVENT_CONFIG["kline_intervals"]
        limits = {"1m": 500, "3m": 300, "5m": 200}

        for tic in SCALP_TICKERS:
            for interval in intervals:
                limit = limits.get(interval, 300)
                try:
                    ohlcv = await asyncio.to_thread(
                        self.broker.fetch_ohlcv, tic, interval, limit
                    )
                    if ohlcv:
                        self.ohlcv_store.bootstrap(tic, interval, ohlcv)
                except Exception as exc:
                    logger.warning("[scalp/bootstrap] %s/%s failed: %s", tic, interval, exc)

        total = sum(
            self.ohlcv_store.bar_count(tic, iv)
            for tic in SCALP_TICKERS
            for iv in intervals
        )
        logger.info("[scalp/bootstrap] Loaded %d total bars. WS takes over.", total)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_trade_csv(self) -> None:
        if not self._trades_csv.exists():
            with open(self._trades_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "action", "ticker", "price", "qty", "value_usd",
                    "pnl_pct", "held_minutes", "reason", "source", "confidence", "dry_run",
                ])

    def _log_trade(self, action: str, ticker: str, price: float, qty: float,
                   value_usd: float, pnl_pct: float = 0, held_minutes: float = 0,
                   reason: str = "", source: str = "signal",
                   confidence: float = 0, dry_run: bool = True) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
        try:
            with open(self._trades_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts, action, ticker, f"{price:.4f}", f"{qty:.6f}", f"{value_usd:.2f}",
                    f"{pnl_pct:.4f}", f"{held_minutes:.1f}", reason, source,
                    f"{confidence:.2f}", dry_run,
                ])
        except Exception as exc:
            logger.warning("[scalp/log] Trade write failed: %s", exc)

    def _log_signal(self, result) -> None:
        if result.action == "ignore":
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": result.ticker,
            "score": result.score,
            "direction": result.direction,
            "action": result.action,
            "agreeing": result.agreeing_count,
            "regime_scale": result.regime_scale,
            "signals": {s.name: {"score": s.score, "reason": s.reason} for s in result.signals},
        }
        try:
            with open(self._signals_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
