"""Event-driven crypto trading loop — Claude Agent as primary decision maker.

No fixed timers. Market events drive all decisions:
- Multi-TF kline (15m/1h/4h) → AdaptiveGate → Claude decides
- BTC >1.5% move → Claude re-evaluates
- Funding rate extreme → store context for next decision
- OI spike + position open → Claude re-evaluates

Architecture:
    Market Event → Multi-TF Aggregator → Build MarketSnapshot → Claude Agent → Execute
                                                                   ↑              ↓
                                                                   └── TS feedback ←┘
    Neo4j (optional) ← trade history + regime insights → Historical Insights in prompt

Rule-based pipeline is NOT used here. It exists in backtests/ only.
If Claude OAuth is unavailable, falls back to rule-based as degraded mode.
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
from typing import Dict, Optional

# Ensure parent dir is importable (independent of crypto_config import order)
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import pandas as pd

from crypto_config import (
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    ACTIVE_BLEND_CONFIG,
    EVENT_CONFIG,
)
from brokers.binance import BinanceBroker
from core.event_bus import EventBus
from core.market_listener import MarketListener
from core.derivatives_monitor import DerivativesMonitor
from core.position_tracker import PositionTracker
from core.agent_evaluator import evaluate_position, build_market_context
from core.online_learner import OnlineLearner
from core.claude_agent import ClaudeAgent, MarketSnapshot
from core.zscore_gate import AdaptiveGate
from core.multi_tf_aggregator import MultiTFAggregator
from core.ohlcv_store import OHLCVStore
from core.memory_store import TradingMemory
from analysis.regime_detector_crypto import CryptoRegimeDetector
from analysis.macro_regime import MacroRegimeDetector

logger = logging.getLogger(__name__)


class CryptoRunner:
    """24/7 Binance crypto autonomous trader — Claude Agent decides."""

    def __init__(self, broker: BinanceBroker, leverage: int = 1, initial_balance: float = 0.0) -> None:
        self.broker = broker
        self.leverage = leverage
        self._initial_balance = initial_balance or 327_000.0
        self.event_bus = EventBus()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # State
        self._entry_prices: Dict[str, float] = {}
        self._trailing_highs: Dict[str, float] = {}
        self._derivatives_context: Dict = {}
        self._agent_memory: str = ""
        self._last_tick_prices: Dict[str, float] = {}
        self._state_file = DATA_DIR / "runner_state.json"

        # OHLCV store — WS-fed, bootstrap once from REST, then zero REST calls
        self.ohlcv_store = OHLCVStore(maxlen=1500)

        # Logging directories (TRADING_LOGS_DIR env var overrides)
        self._log_dir = LOGS_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trades_csv = self._log_dir / "trades.csv"
        self._decisions_log = self._log_dir / "decisions.jsonl"
        self._init_trade_csv()

        # Components
        cfg = ACTIVE_BLEND_CONFIG
        ecfg = EVENT_CONFIG

        self.crypto_regime_detector = CryptoRegimeDetector()
        self.macro_detector = MacroRegimeDetector()
        self.claude_agent = ClaudeAgent()
        self.online_learner = OnlineLearner(
            save_path=str(MODELS_DIR / "online_learner.json"),
            min_trades_to_adapt=5,
        )
        self.online_learner.load()

        # Multi-timeframe aggregator
        kline_intervals = ecfg.get("kline_intervals", ["15m", "1h", "4h"])
        primary_interval = ecfg.get("primary_interval", "15m")
        self.multi_tf_aggregator = MultiTFAggregator(
            intervals=tuple(kline_intervals),
        )

        # Trading memory (Neo4j — optional, gracefully degrades)
        self.memory = TradingMemory()

        # Adaptive gate: Claude controls its own monitoring schedule
        gate_cfg = ecfg.get("gate", {})
        self.adaptive_gate = AdaptiveGate(
            zscore_threshold=gate_cfg.get("zscore_threshold", 2.0),
            zscore_window=gate_cfg.get("zscore_window", 50),
            max_check_seconds=gate_cfg.get("max_check_seconds", 3600),
            min_check_seconds=ecfg.get("min_decision_gap", 120),
        )

        # Market listener: WebSocket multi-TF kline events
        # Shares ohlcv_store so WS candle closes feed directly into it
        self.market_listener = MarketListener(
            event_bus=self.event_bus,
            tickers=cfg["tickers"],
            kline_intervals=kline_intervals,
            primary_interval=primary_interval,
            significant_move_pct=ecfg["significant_move_pct"],
            min_decision_gap=ecfg["min_decision_gap"],
            market=broker.market,
            testnet=broker.paper,
            ohlcv_store=self.ohlcv_store,
        )

        # Derivatives monitor: adaptive Funding/OI polling
        self.derivatives_monitor = DerivativesMonitor(
            event_bus=self.event_bus,
            broker=broker,
            tickers=cfg["tickers"],
            poll_base=ecfg["derivatives_poll_base"],
            poll_fast=ecfg["derivatives_poll_fast"],
            funding_threshold=ecfg["funding_extreme_threshold"],
            oi_spike_threshold=ecfg["oi_spike_threshold"],
        )

        # Position tracker
        self.position_tracker = PositionTracker(
            event_bus=self.event_bus,
            safety_interval=30.0,
            agent_interval=180.0,
        )
        self._setup_tracker()

        # Subscribe to market events
        self._setup_event_subscriptions()

        # Setup leverage for Futures
        if leverage > 1 and broker.market == "future":
            self._setup_leverage()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist entry_prices, entry_times, and agent_memory to disk for restart recovery."""
        # Capture entry_times from position tracker for restart fidelity
        entry_times = {}
        for ticker in self._entry_prices:
            pos = self.position_tracker.get_position(ticker)
            if pos:
                entry_times[ticker] = pos.entry_time.isoformat()
        state = {
            "entry_prices": self._entry_prices,
            "entry_times": entry_times,
            "agent_memory": self._agent_memory,
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(state), encoding="utf-8")
        except Exception as exc:
            logger.warning("[runner] Failed to save state: %s", exc)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
            self._entry_prices = state.get("entry_prices", {})
            self._entry_times = state.get("entry_times", {})
            self._agent_memory = state.get("agent_memory", "")
            if self._entry_prices:
                logger.info("[runner] Restored state: %d entry_prices, memory=%d chars",
                            len(self._entry_prices), len(self._agent_memory))
        except Exception as exc:
            logger.warning("[runner] Failed to load state: %s", exc)

    def _restore_positions(self) -> None:
        """Restore positions from broker on restart.

        Only restores positions that we previously traded (saved in entry_prices).
        In dry_run mode, restores from saved state since broker has no positions.
        """
        self._load_state()
        if not self._entry_prices:
            logger.info("[runner] No saved positions to restore")
            return

        dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")

        positions_detail = {}
        if not dry_run:
            try:
                positions_detail = self.broker.get_positions_detail()
            except Exception as exc:
                logger.warning("[runner] Failed to fetch positions for restore: %s", exc)

        restored = 0
        for ticker, entry_price in list(self._entry_prices.items()):
            if entry_price <= 0:
                continue

            # Try broker first (live mode only)
            info = (positions_detail or {}).get(ticker)
            qty = info.get("qty", 0) if info else 0

            # In dry_run, broker has no positions — estimate qty from entry price
            if qty <= 0 and dry_run:
                pv = 327_000  # approximate portfolio value
                cfg_pct = ACTIVE_BLEND_CONFIG.get("position_pct", 0.10)
                qty = (pv * cfg_pct) / entry_price if entry_price > 0 else 0

            if qty <= 0:
                continue

            if self.position_tracker.get_position(ticker) is None:
                # Restore original entry_time if available
                saved_time = None
                time_str = getattr(self, "_entry_times", {}).get(ticker)
                if time_str:
                    try:
                        from datetime import datetime as dt
                        saved_time = dt.fromisoformat(time_str)
                    except (ValueError, TypeError):
                        pass
                self.position_tracker.track(
                    ticker=ticker,
                    broker="binance",
                    entry_price=entry_price,
                    qty=qty,
                    market_type="crypto",
                    entry_time=saved_time,
                )
                restored += 1
                logger.info("[runner] Restored position: %s @ %.4f x %.6f%s",
                            ticker, entry_price, qty, " (dry_run)" if dry_run else "")

        if restored:
            logger.info("[runner] Restored %d positions", restored)
        else:
            logger.info("[runner] No positions to restore")

    # ------------------------------------------------------------------
    # Trade logging — CSV for trades, JSONL for decisions
    # ------------------------------------------------------------------

    def _init_trade_csv(self) -> None:
        """Create trades.csv with headers if it doesn't exist."""
        if not self._trades_csv.exists():
            with open(self._trades_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "action", "ticker", "price", "qty", "value_usd",
                    "pnl_pct", "held_hours", "reason", "regime", "source",
                    "confidence", "dry_run",
                ])

    def _log_trade(self, action: str, ticker: str, price: float, qty: float,
                   value_usd: float, pnl_pct: float = 0, held_hours: float = 0,
                   reason: str = "", regime: str = "", source: str = "claude",
                   confidence: float = 0, dry_run: bool = True) -> None:
        """Append a trade to CSV log."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._trades_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts, action, ticker, f"{price:.4f}", f"{qty:.6f}", f"{value_usd:.2f}",
                    f"{pnl_pct:.4f}", f"{held_hours:.1f}", reason, regime, source,
                    f"{confidence:.2f}", dry_run,
                ])
        except Exception as exc:
            logger.warning("[log] Failed to write trade: %s", exc)
        logger.info("[trade] %s %s @ $%.2f x %.4f = $%.2f (pnl=%+.2f%%, reason=%s)",
                     action, ticker, price, qty, value_usd, pnl_pct * 100, reason)

    def _log_decision(self, trigger: str, snapshot: MarketSnapshot,
                      agent_result: dict = None, decisions: list = None) -> None:
        """Append full decision context to JSONL log."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": ts,
            "trigger": trigger,
            "btc_price": snapshot.btc_price,
            "regime": snapshot.combined_regime,
            "fear_greed": snapshot.fear_greed_index,
            "portfolio_value": snapshot.portfolio_value,
            "positions": list(snapshot.positions.keys()),
        }
        if agent_result:
            entry["claude_assessment"] = agent_result.get("market_assessment", "")
            entry["next_check_s"] = agent_result.get("next_check_seconds", 0)
            entry["wake_conditions"] = agent_result.get("wake_conditions", [])
            entry["memory"] = agent_result.get("memory_update", "")
        if decisions:
            entry["trades"] = [
                {"action": d["action"], "ticker": d["ticker"],
                 "price": d.get("price", 0), "confidence": d.get("confidence", 0),
                 "reasons": d.get("reasons", [])}
                for d in decisions
            ]
        try:
            with open(self._decisions_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[log] Failed to write decision: %s", exc)

    def _setup_leverage(self) -> None:
        """Set leverage + isolated margin for all tickers."""
        cfg = ACTIVE_BLEND_CONFIG
        for tic in cfg["tickers"]:
            self.broker.set_margin_mode(tic, "isolated")
            self.broker.set_leverage(tic, self.leverage)
        logger.info("[runner] Futures leverage set to %dx", self.leverage)

    def _setup_event_subscriptions(self) -> None:
        """Wire market events through AdaptiveGate to Claude Agent."""

        async def _on_tick(event):
            try:
                await self._handle_tick(event.payload)
            except Exception as exc:
                logger.error("[tick] Failed: %s", exc)

        async def _on_funding_extreme(event):
            self._derivatives_context = self.derivatives_monitor.get_context()
            logger.info("[runner] Funding extreme stored: %s", event.payload.get("ticker"))

        async def _on_oi_spike(event):
            self._derivatives_context = self.derivatives_monitor.get_context()
            logger.info("[runner] OI spike stored: %s", event.payload.get("ticker"))

        self.event_bus.subscribe("market.tick", _on_tick)
        self.event_bus.subscribe("market.funding_extreme", _on_funding_extreme)
        self.event_bus.subscribe("market.oi_spike", _on_oi_spike)

    def _setup_tracker(self) -> None:
        """Wire position tracker with price fetcher, evaluator, and executor."""
        dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")

        async def binance_price(ticker: str) -> float:
            # Use WS tick price (zero REST). No REST fallback during normal operation.
            ws_price = self._last_tick_prices.get(ticker, 0.0)
            if ws_price > 0:
                return ws_price
            # Last resort: try OHLCV store close price
            cfg = ACTIVE_BLEND_CONFIG
            tf = cfg.get("ohlcv_timeframe", "1d")
            df = self.ohlcv_store.get_close_df([ticker], tf)
            if not df.empty and ticker in df.columns:
                return float(df[ticker].iloc[-1])
            return 0.0

        async def context_provider() -> dict:
            # Pass WS tick prices instead of letting build_market_context call REST
            btc_price = self._last_tick_prices.get("_btc_price", 0.0)
            btc_change_1h = 0.0
            # Estimate 1h change from OHLCV store if available
            cfg = ACTIVE_BLEND_CONFIG
            df = self.ohlcv_store.get_close_df([cfg["tickers"][0]], "1h")
            if not df.empty and len(df) >= 2:
                btc_col = cfg["tickers"][0]
                if btc_col in df.columns:
                    prev = float(df[btc_col].iloc[-2])
                    curr = float(df[btc_col].iloc[-1])
                    btc_change_1h = (curr - prev) / prev if prev > 0 else 0.0
            return await asyncio.to_thread(
                build_market_context,
                crypto_regime_detector=self.crypto_regime_detector,
                btc_price=btc_price,
                btc_change_1h=btc_change_1h,
            )

        async def binance_exit(decision: dict) -> dict:
            return await asyncio.to_thread(
                self.broker.execute_decisions, [decision], dry_run
            )

        self.position_tracker.set_price_fetcher("binance", binance_price)
        # agent_evaluator is rule-based fallback for when Claude is unavailable
        self.position_tracker.set_agent_evaluator(evaluate_position)
        self.position_tracker.set_context_provider(context_provider)
        self.position_tracker.set_exit_executor("binance", binance_exit)

        # Position exit -> online learner (regime-aware TS feedback)
        async def _on_exit(event):
            data = event.payload
            ticker = data.get("ticker", "")
            source = data.get("source", "unknown")
            pnl_pct = data.get("pnl_pct", 0)
            held_hours = data.get("held_hours", 0)
            exit_price = data.get("exit_price", 0)
            reason = data.get("reason", "")

            # Clean up _entry_prices for ANY exit (safety or claude)
            self._entry_prices.pop(ticker, None)
            self._save_state()

            # Log safety exits to CSV (claude exits are logged in _run_decision)
            if source != "claude_agent":
                dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                self._log_trade(
                    "SELL", ticker, exit_price, 0, 0,
                    pnl_pct=pnl_pct, held_hours=held_hours,
                    reason=reason, source=f"safety_{source}",
                    dry_run=dry_run,
                )

            # Record Claude's reasoning category as the "agent signal"
            agent_signals = data.get("agent_signals", {})
            if not agent_signals:
                for ev in self.event_bus.get_recent_events(100):
                    if (ev.get("topic") == "decision.signal"
                            and ev.get("payload", {}).get("ticker") == ticker):
                        agent_signals = ev["payload"].get("signals", {})
                        break
            if agent_signals:
                crypto_regime = self.crypto_regime_detector.detect().get("regime_label", "unknown")
                macro_regime = self.macro_detector.regime.value
                combined_regime = f"{crypto_regime}_{macro_regime}"

                self.online_learner.record_trade(
                    ticker=ticker,
                    entry_price=data.get("entry_price", 0),
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    held_hours=held_hours,
                    agent_signals=agent_signals,
                    market_type="crypto",
                    regime=combined_regime,
                )
                # Record to Neo4j for cross-session memory
                try:
                    self.memory.record_trade({
                        "ticker": ticker,
                        "entry_price": data.get("entry_price", 0),
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "held_hours": held_hours,
                        "regime": combined_regime,
                    })
                except Exception:
                    pass

        self.event_bus.subscribe("position.exit", _on_exit)

    # ------------------------------------------------------------------
    # Adaptive gate — tick handler
    # ------------------------------------------------------------------

    async def _handle_tick(self, payload: dict) -> None:
        """Every WS tick → update aggregator → gate evaluates → wake Claude if needed."""
        ticker = payload.get("ticker", "")
        price = payload.get("price", 0)
        interval = payload.get("interval", "")
        is_closed = payload.get("is_closed", False)
        primary = EVENT_CONFIG.get("primary_interval", "15m")

        if price <= 0:
            return

        # Update multi-TF aggregator with closed candles (all intervals)
        if is_closed and interval:
            self.multi_tf_aggregator.update(ticker, interval, payload)

        # Gate evaluation: primary interval only (avoid z-score pollution from
        # the same price arriving on multiple interval streams simultaneously)
        if interval and interval != primary:
            return

        features = self._compute_tick_features(ticker, price, payload)
        should_wake, reasons = self.adaptive_gate.evaluate(features, is_candle_close=is_closed)

        if should_wake:
            trigger = "candle_close" if is_closed else "gate"
            logger.info("[runner] Gate wake (%s) -> Claude deciding", ", ".join(reasons))
            try:
                await self._run_decision(trigger=trigger, event=payload, wake_reasons=reasons)
            except Exception as exc:
                logger.error("[decide] Failed on gate wake: %s", exc)

    def _compute_tick_features(self, ticker: str, price: float, payload: dict) -> Dict[str, float]:
        """Extract features from tick for gate evaluation.

        Feature names are ticker-scoped (e.g. "BTC/USDT:price_change_pct") so
        each asset gets its own z-score tracker.  Global features like btc_price
        are shared across all tickers for wake-condition evaluation.
        """
        features: Dict[str, float] = {}
        prefix = ticker  # e.g. "BTC/USDT"

        # Price change vs previous tick (per-ticker z-score)
        prev_price = self._last_tick_prices.get(ticker, price)
        if prev_price > 0:
            features[f"{prefix}:price_change_pct"] = (price - prev_price) / prev_price
        else:
            features[f"{prefix}:price_change_pct"] = 0.0
        self._last_tick_prices[ticker] = price

        # BTC price — always present (cached from last BTC tick)
        if "BTC" in ticker:
            self._last_tick_prices["_btc_price"] = price
        features["btc_price"] = self._last_tick_prices.get("_btc_price", 0.0)

        # Volume (per-ticker z-score)
        features[f"{prefix}:volume"] = payload.get("volume", 0.0)

        # Derivatives context (cached from DerivativesMonitor)
        deriv = self._derivatives_context
        if deriv:
            fr = deriv.get("funding_rates", {}).get(ticker, 0)
            if fr:
                features["funding_rate"] = float(fr)
            oi = deriv.get("open_interest", {}).get(ticker, {})
            if isinstance(oi, dict) and "change_pct" in oi:
                features["oi_change_pct"] = float(oi["change_pct"])

        return features

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    # _initial_balance: set in __init__ from broker.connect()
    _fg_cache: tuple = (0, 0, "")  # (timestamp, index, label)
    _trending_cache: tuple = (0, "")  # (timestamp, summary)
    _etf_flow_cache: Dict = {}
    _etf_flow_cache_time: float = 0.0

    def _estimate_portfolio_value(self) -> float:
        """Estimate portfolio value from WS prices (dry_run mode, no REST)."""
        # Start from initial balance (set once at connect time)
        base = self._initial_balance or 327_000.0
        position_value = 0.0
        for tic, entry_px in self._entry_prices.items():
            tracked = self.position_tracker.get_position(tic)
            if tracked and tracked.qty > 0:
                current_px = self._last_tick_prices.get(tic, entry_px)
                position_value += tracked.qty * current_px
        # Cash = base - (sum of position values at entry)
        invested = sum(
            self.position_tracker.get_position(tic).qty * entry_px
            for tic, entry_px in self._entry_prices.items()
            if self.position_tracker.get_position(tic)
        )
        return base - invested + position_value

    def _fetch_fear_greed(self) -> tuple:
        """Fetch Fear & Greed Index from alternative.me. Cached for 1 hour."""
        import requests
        now = time.time()
        if now - self._fg_cache[0] < 3600 and self._fg_cache[1] > 0:
            return self._fg_cache[1], self._fg_cache[2]
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            data = resp.json().get("data", [{}])[0]
            index = int(data.get("value", 0))
            label = data.get("value_classification", "")
            self._fg_cache = (now, index, label)
            return index, label
        except Exception as exc:
            logger.debug("[data] Fear & Greed fetch failed: %s", exc)
            return self._fg_cache[1], self._fg_cache[2]

    def _fetch_trending(self) -> str:
        """Fetch CoinGecko trending coins as market buzz. Cached for 1 hour."""
        import requests
        now = time.time()
        if now - self._trending_cache[0] < 3600 and self._trending_cache[1]:
            return self._trending_cache[1]
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=5)
            coins = resp.json().get("coins", [])[:5]
            if not coins:
                return self._trending_cache[1]
            parts = []
            for c in coins:
                item = c.get("item", {})
                name = item.get("name", "")
                symbol = item.get("symbol", "")
                rank = item.get("score", 0) + 1
                price_change = item.get("data", {}).get("price_change_percentage_24h", {}).get("usd", 0)
                if price_change:
                    parts.append(f"#{rank} {symbol}({name}) {price_change:+.1f}%")
                else:
                    parts.append(f"#{rank} {symbol}({name})")
            summary = "Trending: " + ", ".join(parts)
            self._trending_cache = (now, summary)
            return summary
        except Exception as exc:
            logger.debug("[data] Trending fetch failed: %s", exc)
            return self._trending_cache[1]

    def _fetch_etf_flow(self) -> Dict:
        """Fetch BTC ETF daily flow from CoinGlass public API. Cached for 4 hours."""
        import requests
        now = time.time()
        if now - self._etf_flow_cache_time < 14400 and self._etf_flow_cache:
            return self._etf_flow_cache

        result = {"daily_flow_usd": 0.0, "weekly_flow_usd": 0.0, "label": "neutral"}
        try:
            resp = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/etf_flow_total",
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                rows = data["data"]
                if rows:
                    latest = rows[-1] if isinstance(rows, list) else rows
                    daily = float(latest.get("totalFlow", 0) or latest.get("value", 0) or 0)
                    # Weekly: sum last 5 entries (trading days)
                    if isinstance(rows, list) and len(rows) >= 5:
                        weekly = sum(float(r.get("totalFlow", 0) or r.get("value", 0) or 0) for r in rows[-5:])
                    else:
                        weekly = daily

                    if daily > 1e6:
                        label = "inflow"
                    elif daily < -1e6:
                        label = "outflow"
                    else:
                        label = "neutral"

                    result = {"daily_flow_usd": daily, "weekly_flow_usd": weekly, "label": label}
            else:
                logger.debug("[data] ETF flow API returned code=%s", data.get("code"))
        except Exception as exc:
            logger.debug("[data] ETF flow fetch failed: %s", exc)

        self._etf_flow_cache = result
        self._etf_flow_cache_time = now
        return result

    async def _bootstrap_ohlcv(self) -> None:
        """One-time REST call to load historical OHLCV into the WS store.

        After this, MarketListener WS candle closes keep the store updated.
        Zero REST OHLCV calls after bootstrap.
        """
        cfg = ACTIVE_BLEND_CONFIG
        timeframe = cfg.get("ohlcv_timeframe", "1d")
        bars_per_day = {"4h": 6, "2h": 12, "1h": 24, "1d": 1}.get(timeframe, 1)
        limit = min(cfg["ohlcv_lookback_days"] * bars_per_day, 1000)

        for tic in cfg["tickers"]:
            try:
                ohlcv = await asyncio.to_thread(
                    self.broker.fetch_ohlcv, tic, timeframe, limit
                )
                if ohlcv:
                    self.ohlcv_store.bootstrap(tic, timeframe, ohlcv)
            except Exception as e:
                logger.warning("[bootstrap] OHLCV fetch failed for %s: %s", tic, e)

        total = sum(
            self.ohlcv_store.bar_count(tic, timeframe) for tic in cfg["tickers"]
        )
        logger.info("[bootstrap] OHLCV loaded: %d total bars (%s). REST done — WS takes over.",
                    total, timeframe)

    def _fetch_ohlcv_df(self) -> pd.DataFrame:
        """Read OHLCV from the WS-fed store. No REST calls."""
        cfg = ACTIVE_BLEND_CONFIG
        timeframe = cfg.get("ohlcv_timeframe", "1d")
        df = self.ohlcv_store.get_close_df(cfg["tickers"], timeframe)
        self._latest_volumes = self.ohlcv_store.get_latest_volumes()
        return df

    async def _build_snapshot(
        self,
        close_df: pd.DataFrame,
        positions: Dict[str, Dict],
        pv: float,
        trigger: str,
    ) -> MarketSnapshot:
        """Build a complete MarketSnapshot for Claude."""
        cfg = ACTIVE_BLEND_CONFIG
        candidates = [t for t in cfg["tickers"] if t in close_df.columns]

        # Price action
        ticker_prices = {}
        ticker_returns_4h = {}
        ticker_returns_24h = {}
        ticker_volumes = getattr(self, "_latest_volumes", {})
        for tic in candidates:
            series = close_df[tic].dropna()
            if len(series) >= 1:
                ticker_prices[tic] = float(series.iloc[-1])
            if len(series) >= 2:
                ticker_returns_4h[tic] = (float(series.iloc[-1]) / float(series.iloc[-2])) - 1
            if len(series) >= 7:
                ticker_returns_24h[tic] = (float(series.iloc[-1]) / float(series.iloc[-7])) - 1

        # BTC reference
        btc_ticker = cfg["tickers"][0]
        btc_price = ticker_prices.get(btc_ticker, 0)
        # Use WS tick price if available (avoids REST call)
        if btc_price <= 0:
            btc_price = self._last_tick_prices.get(btc_ticker, 0)
        btc_1h = 0.0
        btc_24h = ticker_returns_24h.get(btc_ticker, 0.0)
        # btc_1h estimated from 4h return (avoids separate REST call)
        btc_1h = btc_24h / 24 if btc_24h else 0.0

        # Derivatives
        deriv = self._derivatives_context or self.derivatives_monitor.get_context()

        # Regime (both blocking — run in threads to avoid blocking event loop)
        crypto_result = await asyncio.to_thread(self.crypto_regime_detector.detect)
        crypto_regime = crypto_result.get("regime_label", "unknown")
        macro_regime = await asyncio.to_thread(self.macro_detector.detect)

        # Fear & Greed Index + Trending + ETF Flow (non-blocking, cached)
        fg_index, fg_label = await asyncio.to_thread(self._fetch_fear_greed)
        trending_summary = await asyncio.to_thread(self._fetch_trending)
        etf_flow = await asyncio.to_thread(self._fetch_etf_flow)
        combined_regime = f"{crypto_regime}_{macro_regime.value}"

        # TS posteriors — feed into Claude's context
        ts_mean_weights = self.online_learner.get_mean_weights(regime=combined_regime)
        learner_status = self.online_learner.get_status()

        ts_regime_info = ""
        regime_data = learner_status.get("regime_agents", {}).get(combined_regime, {})
        if regime_data.get("using_own_weights"):
            ts_regime_info = f"Regime '{combined_regime}' has {regime_data['trade_count']} trades — using regime-specific weights"
        else:
            ts_regime_info = f"Regime '{combined_regime}' has <3 trades — using global weights as fallback"

        # Recent trades
        recent_trades = learner_status.get("recent_trades", [])

        # Position PnL enrichment (include tracker data for Claude context)
        enriched_positions = {}
        for tic, pos in positions.items():
            current_px = ticker_prices.get(tic, 0)
            entry_px = pos.get("entry_price", 0)
            pnl_pct = (current_px / entry_px - 1) if entry_px > 0 else 0
            tracked = self.position_tracker.get_position(tic)
            enriched_positions[tic] = {
                **pos,
                "current_price": current_px,
                "pnl_pct": pnl_pct,
                "trailing_high": tracked.trailing_high if tracked else current_px,
                "held_hours": tracked.held_hours if tracked else 0,
            }

        # Cash
        cash = pv - sum(
            p["qty"] * ticker_prices.get(t, 0)
            for t, p in positions.items()
        )
        cash = max(cash, 0)

        # Multi-TF summaries
        multi_tf_summary = {}
        for tic in candidates:
            tf_text = self.multi_tf_aggregator.format_for_prompt(tic)
            if tf_text:
                multi_tf_summary[tic] = tf_text

        # Historical insights from Neo4j (non-blocking)
        historical_insights = ""
        try:
            historical_insights = await asyncio.to_thread(
                self.memory.build_historical_insights, combined_regime, candidates
            )
        except Exception:
            pass

        return MarketSnapshot(
            ticker_prices=ticker_prices,
            ticker_returns_4h=ticker_returns_4h,
            ticker_returns_24h=ticker_returns_24h,
            ticker_volumes=ticker_volumes,
            btc_price=btc_price,
            btc_change_1h=btc_1h,
            btc_change_24h=btc_24h,
            funding_rates=deriv.get("funding_rates", {}),
            open_interest=deriv.get("open_interest", {}),
            taker_delta=deriv.get("taker_delta", {}),
            long_short_ratio=deriv.get("long_short_ratio", {}),
            basis_spreads={
                tic: info["basis_annualized"]
                for tic, info in deriv.get("basis_spread", {}).items()
                if "basis_annualized" in info
            },
            liquidation_usd_1h=deriv.get("liquidations", {}).get("total_usd_1h", 0),
            liquidation_cascade=deriv.get("liquidations", {}).get("is_cascade", False),
            stablecoin_mcap_b=deriv.get("stablecoin_supply", {}).get("total_mcap", 0) / 1e9,
            stablecoin_change_pct=deriv.get("stablecoin_supply", {}).get("change_pct", 0),
            crypto_regime=crypto_regime,
            macro_regime=macro_regime.value,
            combined_regime=combined_regime,
            macro_exposure_scale=self.macro_detector.exposure_scale,
            macro_trail_multiplier=self.macro_detector.trail_multiplier,
            ts_mean_weights=ts_mean_weights,
            ts_regime_info=ts_regime_info,
            ts_total_trades=learner_status.get("total_trades", 0),
            ts_cumulative_pnl_pct=learner_status.get("cumulative_pnl_pct", 0.0),
            recent_trades=recent_trades,
            positions=enriched_positions,
            portfolio_value=pv,
            cash=cash,
            candidates=candidates,
            trigger=trigger,
            fear_greed_index=fg_index,
            fear_greed_label=fg_label,
            news_summary=trending_summary,
            etf_daily_flow_usd=etf_flow.get("daily_flow_usd", 0.0),
            etf_flow_label=etf_flow.get("label", ""),
            multi_tf_summary=multi_tf_summary,
            historical_insights=historical_insights,
        )

    # ------------------------------------------------------------------
    # Decision: Claude-first, rule-based fallback
    # ------------------------------------------------------------------

    async def _run_decision(self, trigger: str = "manual", event: dict = None, wake_reasons: list = None) -> None:
        """Collect data → build snapshot → Claude decides → execute."""
        if self._lock.locked():
            logger.debug("[decide] Skipping: already running")
            return
        async with self._lock:
            if not self.broker.is_connected:
                return

            logger.info("[decide] Collecting data (trigger=%s)...", trigger)
            close_df = self._fetch_ohlcv_df()
            if close_df.empty:
                logger.warning("[decide] No OHLCV data, skipping")
                return

            dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")

            if dry_run:
                # Dry-run: use tracker + WS prices → zero REST calls
                pv = self._estimate_portfolio_value()
                positions_detail = {}
                positions: Dict[str, Dict] = {}
                for tic, entry_px in self._entry_prices.items():
                    tracked = self.position_tracker.get_position(tic)
                    if tracked and tracked.qty > 0:
                        positions[tic] = {
                            "qty": tracked.qty,
                            "entry_price": entry_px,
                        }
            else:
                # Live: use real broker data
                status = await asyncio.to_thread(self.broker.get_account_status)
                pv = status.get("portfolio_value", 1000.0)
                positions_detail = await asyncio.to_thread(self.broker.get_positions_detail)
                positions: Dict[str, Dict] = {}
                for tic, info in (positions_detail or {}).items():
                    entry_px = self._entry_prices.get(tic, info.get("entry_price", 0))
                    if entry_px > 0 and info.get("qty", 0) > 0:
                        positions[tic] = {
                            "qty": info["qty"],
                            "entry_price": entry_px,
                        }

            # Build MarketSnapshot for Claude
            snapshot = await self._build_snapshot(close_df, positions, pv, trigger)
            snapshot.gate_wake_reasons = wake_reasons or []
            snapshot.agent_memory = self._agent_memory

            # === Claude Agent decides ===
            agent_result = await self.claude_agent.decide(snapshot)

            if agent_result and "decisions" in agent_result:
                logger.info("[decide] Claude: %s", agent_result.get("market_assessment", ""))
                decisions = self._convert_claude_decisions(agent_result, snapshot)

                # Update gate with Claude's scheduling decisions
                self.adaptive_gate.update_from_claude(
                    next_check_seconds=agent_result.get("next_check_seconds"),
                    wake_conditions=agent_result.get("wake_conditions"),
                )

                # Save agent memory (truncate to 500 chars)
                mem = agent_result.get("memory_update", "")
                self._agent_memory = mem[:500] if mem else self._agent_memory

                # Record learning insight to Neo4j
                learning_note = agent_result.get("learning_note", "")
                if learning_note:
                    try:
                        self.memory.record_insight(snapshot.combined_regime, learning_note)
                    except Exception:
                        pass
            else:
                # Fallback: rule-based pipeline (degraded mode)
                logger.warning("[decide] Claude unavailable — using rule-based fallback")
                decisions = await self._run_fallback_pipeline(close_df, positions, pv, snapshot, trigger)
                # Set conservative gate schedule on fallback
                self.adaptive_gate.update_from_claude(next_check_seconds=3600)

            # Log decision context (always, even if no trades)
            self._log_decision(trigger, snapshot, agent_result, decisions)

            # === Execute ===
            if decisions:
                exec_result = await asyncio.to_thread(
                    self.broker.execute_decisions, decisions, dry_run
                )
                logger.info("[decide] Executed: %d/%d orders (%s, trigger=%s)",
                            exec_result.get("successful", 0),
                            exec_result.get("total_orders", 0),
                            "dry_run" if dry_run else "LIVE",
                            trigger)

                # Build set of successfully executed tickers
                executed_tickers = set()
                for r in exec_result.get("orders", []):
                    if r.get("status") in ("submitted", "dry_run"):
                        t = r.get("ticker", "")
                        if ":" in t:
                            t = t.split(":")[0]
                        executed_tickers.add(t)

                for d in decisions:
                    ticker = d["ticker"]
                    if d["action"] == "BUY" and d.get("price", 0) > 0:
                        if ticker not in executed_tickers:
                            logger.warning("[decide] BUY %s failed execution, skipping track", ticker)
                            continue
                        # Skip if already holding this ticker (prevent duplicate position)
                        if self.position_tracker.get_position(ticker) is not None:
                            logger.info("[decide] BUY %s skipped: already holding position", ticker)
                            continue
                        self._entry_prices[ticker] = d["price"]
                        qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                        if qty > 0:
                            self.position_tracker.track(
                                ticker=ticker,
                                broker="binance",
                                entry_price=d["price"],
                                qty=qty,
                                market_type="crypto",
                                regime=snapshot.combined_regime,
                            )
                        await self.event_bus.publish("decision.signal", {
                            "ticker": ticker,
                            "signals": d.get("agent_signals", {}),
                        })
                        # === Log BUY ===
                        self._log_trade(
                            "BUY", ticker, d["price"], qty, d.get("position_size_usd", 0),
                            reason=d.get("reasons", [""])[0] if d.get("reasons") else "",
                            regime=snapshot.combined_regime, confidence=d.get("confidence", 0),
                            dry_run=dry_run,
                        )
                        self._write_alert("BUY", f"{ticker} @ ${d['price']:,.2f} ({d.get('confidence',0):.0%})")

                    elif d["action"] == "SELL" and d.get("price", 0) > 0:
                        if ticker not in executed_tickers:
                            logger.warning("[decide] SELL %s failed execution, skipping", ticker)
                            continue
                        tracked = self.position_tracker.get_position(ticker)
                        if tracked is None:
                            logger.info("[decide] %s already exited by safety layer, skipping", ticker)
                            self._entry_prices.pop(ticker, None)
                            continue

                        entry_px = self._entry_prices.get(ticker, d["price"])
                        pnl_pct = (d["price"] - entry_px) / entry_px if entry_px > 0 else 0
                        held_hours = tracked.held_hours
                        sell_reason = d.get("reasons", ["claude_sell"])[0] if d.get("reasons") else "claude_sell"

                        self.position_tracker.untrack(ticker)
                        self._entry_prices.pop(ticker, None)

                        await self.event_bus.publish("position.exit", {
                            "ticker": ticker,
                            "source": "claude_agent",
                            "reason": sell_reason,
                            "entry_price": entry_px,
                            "exit_price": d["price"],
                            "pnl_pct": pnl_pct,
                            "held_hours": held_hours,
                            "broker": "binance",
                            "agent_signals": d.get("agent_signals", {}),
                        })
                        # === Log SELL ===
                        sell_qty = tracked.qty if tracked else 0
                        sell_value = d["price"] * sell_qty
                        self._log_trade(
                            "SELL", ticker, d["price"], sell_qty, sell_value,
                            pnl_pct=pnl_pct, held_hours=held_hours,
                            reason=sell_reason, regime=snapshot.combined_regime,
                            source="claude", confidence=d.get("confidence", 0),
                            dry_run=dry_run,
                        )
                        self._write_alert("SELL", f"{ticker} @ ${d['price']:,.2f} PnL={pnl_pct:+.2%}")

                # Persist state after any trade
                if executed_tickers:
                    self._save_state()
            else:
                logger.info("[decide] No trades (trigger=%s)", trigger)

    def _convert_claude_decisions(
        self,
        agent_result: Dict,
        snapshot: MarketSnapshot,
    ) -> list:
        """Convert Claude's JSON decisions into broker-executable format."""
        decisions = []
        for d in agent_result.get("decisions", []):
            action = d.get("action", "HOLD").upper()
            if action == "HOLD":
                continue

            ticker = d.get("ticker", "")
            # Normalize futures ticker (BTC/USDT:USDT → BTC/USDT)
            if ":" in ticker:
                ticker = ticker.split(":")[0]
            if ticker not in snapshot.candidates and action == "BUY":
                continue

            price = snapshot.ticker_prices.get(ticker, 0)
            if price <= 0:
                continue

            position_pct = min(d.get("position_pct", 0.10), 0.20)
            position_size_usd = snapshot.portfolio_value * position_pct

            # Build agent_signals: prefer Claude's explicit signal_weights, fallback to keyword extraction
            confidence = d.get("confidence", 0.5)
            reasoning = d.get("reasoning", "")
            claude_sw = d.get("signal_weights")
            if claude_sw and isinstance(claude_sw, dict):
                agent_signals = {k: float(v) for k, v in claude_sw.items() if abs(float(v)) >= 0.05}
            else:
                agent_signals = self._extract_signals_from_reasoning(reasoning, confidence)

            dec = {
                "ticker": ticker,
                "action": action,
                "confidence": confidence,
                "position_size_usd": position_size_usd,
                "price": price,
                "reasons": [reasoning],
                "agent_signals": agent_signals,
            }
            # For SELL: include qty from tracker so dry_run doesn't need REST
            if action == "SELL":
                tracked = self.position_tracker.get_position(ticker)
                if tracked and tracked.qty > 0:
                    dec["qty"] = tracked.qty
            decisions.append(dec)

        return decisions

    @staticmethod
    def _extract_signals_from_reasoning(reasoning: str, confidence: float) -> Dict[str, float]:
        """Extract signal categories from Claude's reasoning for TS feedback.

        Maps Claude's reasoning to signal types so Thompson Sampling can learn
        which reasoning patterns are reliable in which regimes.
        """
        reasoning_lower = reasoning.lower()
        signals = {}

        keywords_to_signals = {
            "momentum": "momentum",
            "trend": "momentum",
            "funding": "funding_rate",
            "cvd": "oi_signal",
            "open interest": "oi_signal",
            "long/short": "oi_signal",
            "regime": "regime",
            "macro": "macro",
            "sentiment": "sentiment",
            "volume": "market",
            "price action": "market",
            "support": "quant",
            "resistance": "quant",
            "rsi": "quant",
        }

        for keyword, signal_name in keywords_to_signals.items():
            if keyword in reasoning_lower:
                signals[signal_name] = confidence

        # Default: attribute to "market" if no keywords matched
        if not signals:
            signals["market"] = confidence

        return signals

    async def _run_fallback_pipeline(
        self,
        close_df: pd.DataFrame,
        positions: Dict,
        pv: float,
        snapshot: MarketSnapshot,
        trigger: str,
    ) -> list:
        """Rule-based fallback when Claude is unavailable. Same as backtesting pipeline."""
        # Lazy import — pipeline is for backtesting / fallback only
        from core.pipeline import (
            Pipeline,
            RegimeBlendDetectNode,
            DerivativesSignalNode,
            RegimeBlendSignalNode,
            RegimeBlendExitNode,
            RegimeBlendEntryNode,
        )

        cfg = dict(ACTIVE_BLEND_CONFIG)

        # Apply macro scaling
        if snapshot.macro_exposure_scale != 1.0:
            cfg["position_pct"] = cfg.get("position_pct", 0.20) * snapshot.macro_exposure_scale
        if snapshot.macro_trail_multiplier != 1.0:
            cfg["trail_pct"] = cfg.get("trail_pct", 0.12) * snapshot.macro_trail_multiplier

        adaptive_weights = self.online_learner.sample_weights(regime=snapshot.combined_regime)

        pipe = Pipeline([
            RegimeBlendDetectNode(),
            DerivativesSignalNode(),
            RegimeBlendSignalNode(),
            RegimeBlendExitNode(),
            RegimeBlendEntryNode(),
        ])

        ctx = pipe.run({
            "crypto_close": close_df,
            "eval_date": close_df.index[-1],
            "btc_ticker": ACTIVE_BLEND_CONFIG["tickers"][0],
            "candidates": snapshot.candidates,
            "positions": positions,
            "cash": snapshot.cash,
            "trailing_highs": dict(self._trailing_highs),
            "rb_config": cfg,
            "derivatives_context": self._derivatives_context,
            "macro_context": self.macro_detector.get_context(),
            "adaptive_weights": adaptive_weights,
            "trigger": trigger,
        })

        logger.info("[fallback] Pipeline: %s", pipe.summary())

        self._trailing_highs.clear()
        self._trailing_highs.update(ctx.get("trailing_highs", {}))

        decisions = []
        for trade in ctx.get("trade_log", []):
            dec = {
                "ticker": trade["ticker"],
                "action": trade["side"],
                "confidence": trade.get("score", 0.5) if trade["side"] == "BUY" else -1.0,
                "position_size_usd": trade["qty"] * trade["price"],
                "price": trade["price"],
                "reasons": [trade.get("reason", "rule_based_fallback")],
                "agent_signals": {},
            }
            # For SELL: include qty so dry_run doesn't need REST
            if trade["side"] == "SELL":
                dec["qty"] = trade["qty"]
            decisions.append(dec)
        return decisions

    # ------------------------------------------------------------------
    # Alerts (for Skills/Hooks monitoring)
    # ------------------------------------------------------------------

    def _write_alert(self, alert_type: str, message: str) -> None:
        """Write alert for hooks/monitoring."""
        alert = {"time": time.time(), "type": alert_type, "message": message}
        alert_path = self._log_dir / "alerts.jsonl"
        try:
            with open(alert_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("[alert] Failed to write: %s", exc)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def _watchdog_loop(self) -> None:
        """Fallback timer — if no ticks arrive, force a decision check.

        The AdaptiveGate timer is only evaluated on incoming ticks.
        If WS disconnects, ticks stop and the timer is never checked.
        This watchdog ensures Claude is still called periodically.
        """
        watchdog_interval = self.adaptive_gate._max_check_seconds  # 1h default
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=watchdog_interval
                )
                break
            except asyncio.TimeoutError:
                pass

            # Check if ticks have stopped (no tick for >5 min = likely WS issue)
            last_tick_age = time.time() - max(
                (self.market_listener._last_message_time
                 if hasattr(self.market_listener, "_last_message_time") else 0),
                0.0,
            )
            if last_tick_age > 300:
                logger.warning("[watchdog] No ticks for %.0fs — forcing decision", last_tick_age)
                try:
                    await self._run_decision(trigger="watchdog")
                except Exception as exc:
                    logger.error("[watchdog] Decision failed: %s", exc)

    async def _heartbeat_loop(self) -> None:
        """Write heartbeat file every 30s for Docker healthcheck."""
        hb_path = DATA_DIR / "heartbeat"
        while not self._shutdown_event.is_set():
            hb_path.write_text(str(time.time()))
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=30
                )
                break
            except asyncio.TimeoutError:
                pass

    async def run(self) -> None:
        """Start event-driven components."""
        # Restore positions from broker before starting loops
        self._restore_positions()

        logger.info("[runner] Initial balance: $%.2f", self._initial_balance)

        # Bootstrap OHLCV store from REST (one-time, then WS takes over)
        await self._bootstrap_ohlcv()

        mode = "CLAUDE AGENT" if self.claude_agent.is_available else "RULE-BASED FALLBACK"
        logger.info("[runner] Starting event-driven system (%s mode)", mode)
        logger.info("[runner] Components: market_listener(WS %s), derivatives_monitor(adaptive), "
                    "position_tracker(30s/180s), adaptive_gate, multi_tf_aggregator, heartbeat(30s)",
                    "/".join(EVENT_CONFIG.get("kline_intervals", ["4h"])))
        logger.info("[runner] AdaptiveGate: Claude controls its own wake schedule.")

        self._tasks = [
            asyncio.create_task(self.market_listener.run(), name="market_listener"),
            asyncio.create_task(self.derivatives_monitor.run(), name="derivatives_monitor"),
            asyncio.create_task(self.position_tracker.run(), name="position_tracker"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
        ]

        # Dashboard API (optional, non-fatal if port in use)
        try:
            from api import start_api_server
            api_port = int(os.environ.get("API_PORT", "8080"))
            self._tasks.append(
                asyncio.create_task(start_api_server(self, port=api_port), name="api_server")
            )
        except Exception as exc:
            logger.warning("[runner] Dashboard API failed to start: %s", exc)
        try:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            # Log any task that died with an exception
            for task, result in zip(self._tasks, results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error("[runner] Task '%s' crashed: %s", task.get_name(), result, exc_info=result)
        except asyncio.CancelledError:
            logger.info("[runner] Cancelled, shutting down")
        finally:
            self.market_listener.stop()
            self.derivatives_monitor.stop()
            self.position_tracker.stop()
            self.online_learner.save()
            self._save_state()
            self.memory.close()
            logger.info("[runner] Stopped. State + online learner saved.")

    def stop(self) -> None:
        """Signal all components to stop immediately."""
        self._shutdown_event.set()
        self.market_listener.stop()
        self.derivatives_monitor.stop()
        self.position_tracker.stop()
        for t in getattr(self, "_tasks", []):
            t.cancel()
