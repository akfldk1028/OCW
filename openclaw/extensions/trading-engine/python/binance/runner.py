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
from typing import Any, Dict, Optional

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
    ROUND_TRIP_FEE,
)
from brokers.binance import BinanceBroker
from core.event_bus import EventBus
from core.market_listener import MarketListener
from core.derivatives_monitor import DerivativesMonitor
from core.position_tracker import PositionTracker
from core.agent_evaluator import evaluate_position, build_market_context
from core.online_learner import HierarchicalOnlineLearner
from core.claude_agent import ClaudeAgent, MarketSnapshot
from core.zscore_gate import AdaptiveGate
from core.multi_tf_aggregator import MultiTFAggregator
from core.ohlcv_store import OHLCVStore
from core.memory_store import TradingMemory
from core.liquidation_tracker import LiquidationTracker
from core.ood_detector import OODDetector
from core.exit_regret_tracker import ExitRegretTracker, ExitSnapshot
from analysis.regime_detector_crypto import CryptoRegimeDetector
from analysis.macro_regime import MacroRegimeDetector

logger = logging.getLogger(__name__)


class CryptoRunner:
    """24/7 Binance crypto autonomous trader — Claude Agent decides."""

    def __init__(self, broker: BinanceBroker, leverage: int = 1, initial_balance: float = 0.0) -> None:
        self.broker = broker
        self.leverage = leverage
        self._initial_balance = initial_balance  # set by broker.connect(), no fallback
        self.event_bus = EventBus()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # State
        self._entry_prices: Dict[str, float] = {}
        self._trailing_highs: Dict[str, float] = {}
        self._derivatives_context: Dict = {}
        self._agent_memory: str = ""
        self._last_tick_prices: Dict[str, float] = {}
        self._last_close_prices: Dict[str, float] = {}  # candle-close prices for z-score
        self._exit_cooldowns: Dict[str, float] = {}  # {ticker: exit_timestamp} — anti-churn
        self._realized_pnl_usd: float = 0.0  # cumulative realized PnL from closed trades
        self._broker_balance: float = 0.0  # actual broker balance, synced on startup
        self._broker_balance_time: float = 0.0  # last sync time
        self._entry_meta: Dict[str, dict] = {}  # {ticker: {position_pct, confidence}} for meta-param RL
        self._state_file = DATA_DIR / "runner_state.json"
        # Counterfactual learning: snapshots of HOLD decisions for missed-opportunity tracking
        # Multi-horizon: scalp (3min), swing (30min), trend (2h) — each checked independently
        from collections import deque
        self._hold_snapshots: deque = deque(maxlen=15)  # scalp horizon — reduced to limit CF noise
        self._swing_snapshots: deque = deque(maxlen=10)  # swing horizon — reduced to limit CF noise
        self._trend_snapshots: deque = deque(maxlen=8)   # trend horizon — reduced to limit CF noise
        self._cf_count_today: int = 0                    # daily CF counter
        self._cf_count_date: str = ""                    # date for CF counter reset
        # Post-exit counterfactual: track price after exits for hold_patience learning
        self.exit_regret_tracker = ExitRegretTracker()

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
        self.online_learner = HierarchicalOnlineLearner(
            save_path=str(MODELS_DIR / "online_learner.json"),
            min_trades_to_adapt=5,
        )
        self.online_learner.load()  # auto-migrates from v1/v2 if needed

        # Multi-timeframe aggregator
        kline_intervals = ecfg.get("kline_intervals", ["15m", "1h", "4h"])
        primary_interval = ecfg.get("primary_interval", "15m")
        self.multi_tf_aggregator = MultiTFAggregator(
            intervals=tuple(kline_intervals),
        )

        # Trading memory (Neo4j — optional, gracefully degrades)
        self.memory = TradingMemory()

        # Trading diary: 3-layer self-reflection memory (FinMem/Reflexion/SAGE)
        from core.trading_diary import TradingDiary
        self.diary = TradingDiary(
            data_dir=DATA_DIR / "diary",
            md_dir=Path(os.environ.get("DIARY_MD_DIR",
                os.path.expanduser("~/Documents/trading-diary"))),
            claude_agent=self.claude_agent,
            online_learner=self.online_learner,
        )

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

        # Liquidation tracker: real-time WS forceOrder → cluster estimation → H-TS signal
        self.liquidation_tracker = LiquidationTracker(tickers=cfg["tickers"])

        # OOD detector: Mahalanobis distance for anomalous market state detection
        self.ood_detector = OODDetector(window=200, min_samples=30)

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
        # Capture trailing_highs, trailing_lows, qty, and side for restart recovery
        trailing_highs = {}
        trailing_lows = {}
        position_qtys = {}
        position_sides = {}
        for ticker in self._entry_prices:
            pos = self.position_tracker.get_position(ticker)
            if pos:
                trailing_highs[ticker] = pos.trailing_high
                trailing_lows[ticker] = pos.trailing_low
                position_qtys[ticker] = pos.qty
                position_sides[ticker] = pos.side
        state = {
            "entry_prices": self._entry_prices,
            "entry_times": entry_times,
            "trailing_highs": trailing_highs,
            "trailing_lows": trailing_lows,
            "position_qtys": position_qtys,
            "position_sides": position_sides,
            "agent_memory": self._agent_memory,
            "realized_pnl_usd": self._realized_pnl_usd,
            "entry_meta": self._entry_meta,
            "initial_balance": self._initial_balance,
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self._state_file)  # atomic on POSIX
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
            self._saved_trailing_highs = state.get("trailing_highs", {})
            self._saved_trailing_lows = state.get("trailing_lows", {})
            self._saved_position_qtys = state.get("position_qtys", {})
            self._saved_position_sides = state.get("position_sides", {})
            self._agent_memory = state.get("agent_memory", "")
            self._realized_pnl_usd = state.get("realized_pnl_usd", 0.0)
            self._entry_meta = state.get("entry_meta", {})
            # Restore initial_balance if broker returned 0 (demo session expired)
            saved_balance = state.get("initial_balance", 0.0)
            if self._initial_balance <= 0 and saved_balance > 0:
                self._initial_balance = saved_balance
                logger.info("[runner] Restored initial_balance from state: $%.2f (broker returned $0)", saved_balance)
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
            broker_qty = info.get("qty", 0) if info else 0
            saved_qty = getattr(self, "_saved_position_qtys", {}).get(ticker, 0)

            # Prefer saved qty if broker qty differs significantly (>2x mismatch)
            if broker_qty > 0 and saved_qty > 0:
                ratio = max(broker_qty, saved_qty) / min(broker_qty, saved_qty)
                if ratio > 2.0:
                    logger.warning("[runner] Position qty mismatch for %s: broker=%.8f saved=%.8f (ratio=%.1fx) — using saved", ticker, broker_qty, saved_qty, ratio)
                    qty = saved_qty
                else:
                    qty = broker_qty
            elif broker_qty > 0:
                qty = broker_qty
            else:
                qty = saved_qty

            # Last resort: estimate qty from entry price (legacy fallback)
            if qty <= 0 and dry_run:
                pv = self._initial_balance
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
                saved_side = getattr(self, "_saved_position_sides", {}).get(ticker, "long")
                pos = self.position_tracker.track(
                    ticker=ticker,
                    broker="binance",
                    entry_price=entry_price,
                    qty=qty,
                    market_type="crypto",
                    entry_time=saved_time,
                    side=saved_side,
                )
                # Restore trailing_high — but cap to current WS price to prevent
                # immediate trailing stop trigger on restart
                saved_th = getattr(self, "_saved_trailing_highs", {}).get(ticker)
                if saved_th and saved_th > entry_price:
                    ws_price = self._last_tick_prices.get(ticker, 0)
                    if ws_price > 0:
                        # Cap trailing_high to max of current price (can't trail above live price)
                        pos.trailing_high = min(saved_th, ws_price * 1.005)  # 0.5% tolerance
                    else:
                        pos.trailing_high = saved_th
                # Restore trailing_low for short positions
                saved_tl = getattr(self, "_saved_trailing_lows", {}).get(ticker)
                if saved_tl and saved_tl > 0 and saved_tl < entry_price:
                    ws_price = self._last_tick_prices.get(ticker, 0)
                    if ws_price > 0:
                        pos.trailing_low = max(saved_tl, ws_price * 0.995)
                    else:
                        pos.trailing_low = saved_tl
                restored += 1
                logger.info("[runner] Restored position: %s %s @ %.4f x %.6f (trail_high=%.4f, trail_low=%.4f)%s",
                            saved_side.upper(), ticker, entry_price, qty,
                            pos.trailing_high, pos.trailing_low,
                            " (dry_run)" if dry_run else "")

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
        """Append full decision context to JSONL + human-readable TXT log."""
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
            entry["learning_note"] = agent_result.get("learning_note", "")

        trade_entries = []
        if decisions:
            for d in decisions:
                # Converted decisions use "reasons" (list) and "agent_signals" (dict)
                reasoning = d.get("reasons", [""])[0] if d.get("reasons") else ""
                pv = snapshot.portfolio_value if snapshot.portfolio_value > 0 else 1
                trade_entries.append({
                    "action": d["action"], "ticker": d["ticker"],
                    "price": d.get("price", 0), "confidence": d.get("confidence", 0),
                    "position_pct": round(d.get("position_size_usd", 0) / pv, 4),
                    "reasoning": reasoning,
                    "signal_weights": d.get("agent_signals", {}),
                })
            entry["trades"] = trade_entries

        # JSONL log (machine-readable)
        try:
            with open(self._decisions_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[log] Failed to write decision: %s", exc)

        # Human-readable TXT log (agent_decisions.txt)
        try:
            txt_path = self._log_dir / "agent_decisions.txt"
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{ts}] Trigger: {trigger}\n")
                f.write(f"BTC: ${snapshot.btc_price:,.2f} | Regime: {snapshot.combined_regime} | F&G: {snapshot.fear_greed_index}\n")
                f.write(f"Portfolio: ${snapshot.portfolio_value:,.2f} | Positions: {', '.join(snapshot.positions.keys()) or 'none'}\n")
                if agent_result:
                    f.write(f"\n--- Claude Assessment ---\n")
                    f.write(f"{agent_result.get('market_assessment', 'N/A')}\n")
                    if agent_result.get("learning_note"):
                        f.write(f"\n--- Learning Note ---\n{agent_result['learning_note']}\n")
                if trade_entries:
                    f.write(f"\n--- Decisions ({len(trade_entries)}) ---\n")
                    for t in trade_entries:
                        f.write(f"  {t['action']} {t['ticker']} @ ${t['price']:,.2f} "
                                f"(conf={t['confidence']:.0%}, size={t['position_pct']:.0%})\n")
                        if t.get("reasoning"):
                            f.write(f"    Reason: {t['reasoning'][:300]}\n")
                        if t.get("signal_weights"):
                            f.write(f"    Signals: {t['signal_weights']}\n")
                else:
                    f.write(f"\n--- Decision: HOLD (no trades) ---\n")
                if agent_result and agent_result.get("memory_update"):
                    f.write(f"\n--- Memory ---\n{agent_result['memory_update'][:300]}\n")
                f.write(f"{'='*80}\n")
        except Exception as exc:
            logger.warning("[log] Failed to write agent_decisions.txt: %s", exc)

    # ------------------------------------------------------------------
    # Counterfactual learning helpers
    # ------------------------------------------------------------------

    def _extract_ta_signals(self, ticker: str, snapshot: MarketSnapshot) -> Dict[str, float]:
        """Extract TA-derived signals from snapshot for H-TS virtual updates.

        Used for both counterfactual snapshots and entry-time TA capture.
        Maps raw TA indicators (15m) → normalized signal values in [-1, 1].
        """
        ta_signals: Dict[str, float] = {}
        price = snapshot.ticker_prices.get(ticker, 0)
        if price <= 0:
            return ta_signals

        ta = snapshot.pre_computed_ta.get(ticker, {}).get("15m", {})
        if not ta:
            return ta_signals

        # RSI → rsi_signal
        rsi = ta.get("rsi", 50)
        if rsi > 55:
            ta_signals["rsi_signal"] = min(1.0, (rsi - 50) / 50)
        elif rsi < 45:
            ta_signals["rsi_signal"] = max(-1.0, (rsi - 50) / 50)

        # StochRSI → stoch_rsi
        stoch = ta.get("stoch_rsi", 50)
        if stoch > 60:
            ta_signals["stoch_rsi"] = min(1.0, (stoch - 50) / 50)
        elif stoch < 40:
            ta_signals["stoch_rsi"] = max(-1.0, (stoch - 50) / 50)

        # EMA cross
        ema = ta.get("ema_cross", {})
        if ema.get("status") == "bullish":
            ta_signals["ema_cross_fast"] = min(1.0, abs(ema.get("gap_pct", 0)) * 100 + 0.3)
        elif ema.get("status") == "bearish":
            ta_signals["ema_cross_fast"] = -min(1.0, abs(ema.get("gap_pct", 0)) * 100 + 0.3)

        # MACD histogram
        macd_h = ta.get("macd", {}).get("histogram", 0)
        if macd_h and isinstance(macd_h, (int, float)) and price > 0:
            macd_pct = macd_h / price
            ta_signals["macd_histogram"] = max(-1.0, min(1.0, macd_pct * 500))

        # Bollinger %B → bb_deviation
        bb_pctb = ta.get("bollinger", {}).get("pct_b", 0.5)
        if isinstance(bb_pctb, (int, float)):
            if bb_pctb > 0.8:
                ta_signals["bb_deviation"] = min(1.0, (bb_pctb - 0.5) * 2)
            elif bb_pctb < 0.2:
                ta_signals["bb_deviation"] = max(-1.0, (bb_pctb - 0.5) * 2)

        # VWAP deviation
        vwap_dev = ta.get("vwap", {}).get("deviation_pct", 0)
        if vwap_dev and isinstance(vwap_dev, (int, float)):
            ta_signals["vwap_deviation"] = max(-1.0, min(1.0, vwap_dev * 10))

        # Trend strength from multi-TF
        tf_summary = self.multi_tf_aggregator.get_summary(ticker)
        if tf_summary:
            s15m = tf_summary.get("15m", {})
            if s15m.get("ema_cross") == "bullish":
                ta_signals["trend_strength"] = 0.5
            elif s15m.get("ema_cross") == "bearish":
                ta_signals["trend_strength"] = -0.5

        # Support/resistance from TA
        sr = ta.get("support_resistance")
        if sr and isinstance(sr, dict):
            dist = sr.get("distance_pct", 0)
            if isinstance(dist, (int, float)) and abs(dist) > 0.001:
                ta_signals["support_resistance"] = max(-1.0, min(1.0, -dist * 50))

        # Volume spike
        vol = ta.get("volume", {})
        if isinstance(vol, dict):
            vol_ratio = vol.get("ratio", 1.0)
            if isinstance(vol_ratio, (int, float)):
                if vol_ratio > 1.5:
                    ta_signals["volume_spike"] = min(1.0, (vol_ratio - 1) / 2)
                elif vol_ratio < 0.5:
                    ta_signals["volume_spike"] = max(-1.0, (vol_ratio - 1))

        # -- Derivatives signals --
        # Funding rate
        fr = snapshot.funding_rates.get(ticker, 0)
        if isinstance(fr, (int, float)) and abs(fr) > 0.0001:
            # Positive funding = longs pay → bearish signal
            ta_signals["funding_rate"] = max(-1.0, min(1.0, -fr * 2000))

        # OI change (4h change from Binance OI History)
        oi = snapshot.open_interest.get(ticker, {})
        if isinstance(oi, dict):
            oi_chg = oi.get("change_pct", 0)
            if isinstance(oi_chg, (int, float)) and abs(oi_chg) > 0.002:
                ta_signals["oi_change"] = max(-1.0, min(1.0, oi_chg * 10))

        # Long/short ratio
        ls = snapshot.long_short_ratio.get(ticker, {})
        if isinstance(ls, dict):
            ratio = ls.get("long_short_ratio", ls.get("ratio", 1.0))
            if isinstance(ratio, (int, float)) and abs(ratio - 1.0) > 0.05:
                # ratio > 1 = more longs = contrarian bearish
                ta_signals["long_short_ratio"] = max(-1.0, min(1.0, -(ratio - 1.0) * 2))

        # Basis spread
        basis = snapshot.basis_spreads.get(ticker, 0)
        if isinstance(basis, (int, float)) and abs(basis) > 0.05:
            ta_signals["basis_spread"] = max(-1.0, min(1.0, basis / 0.5))

        # -- Macro signals --
        # Fear & Greed → sentiment
        fg = getattr(snapshot, 'fear_greed_index', None)
        if fg and isinstance(fg, (int, float)):
            ta_signals["fear_greed"] = (fg - 50) / 50  # 0→-1, 50→0, 100→+1

        # Market regime (from regime detector)
        regime = getattr(snapshot, 'regime', '')
        if regime:
            if 'bull' in regime:
                ta_signals["market_regime"] = 0.5
            elif 'bear' in regime or 'stag' in regime:
                ta_signals["market_regime"] = -0.5

        # --- 추가 시그널 (12개) ---

        # ema_cross_slow (EMA21 vs EMA50)
        ema_21 = ta.get("ema_cross", {}).get("ema_21", 0)
        ema_50 = ta.get("ema_50")
        if ema_21 and ema_50:
            ta_signals["ema_cross_slow"] = 0.5 if ema_21 > ema_50 else -0.5

        # supertrend
        st_dir = ta.get("supertrend_dir")
        if st_dir is not None and st_dir != 0:
            ta_signals["supertrend"] = 0.6 if st_dir > 0 else -0.6

        # bb_squeeze (Bollinger bandwidth)
        bw = ta.get("bollinger", {}).get("bandwidth", 0)
        if isinstance(bw, (int, float)) and bw > 0:
            if bw < 0.02:
                ta_signals["bb_squeeze"] = 0.5   # breakout imminent
            elif bw > 0.06:
                ta_signals["bb_squeeze"] = -0.3  # extended

        # cvd_signal (taker buy/sell ratio)
        td = getattr(snapshot, 'taker_delta', {})
        if isinstance(td, dict):
            td_data = td.get(ticker, {})
            if isinstance(td_data, dict):
                bsr = td_data.get("buy_sell_ratio", td_data.get("ratio", 1.0))
                if isinstance(bsr, (int, float)) and abs(bsr - 1.0) > 0.02:
                    ta_signals["cvd_signal"] = max(-1.0, min(1.0, (bsr - 1.0) * 5))

        # obv_divergence
        obv_dir = ta.get("obv_direction")
        if obv_dir is not None and obv_dir != 0:
            ta_signals["obv_divergence"] = max(-1.0, min(1.0, obv_dir))

        # mfi_signal (Money Flow Index — continuous, not just extremes)
        mfi = ta.get("mfi")
        if isinstance(mfi, (int, float)):
            # Continuous signal: 50=neutral, >50=bullish, <50=bearish
            ta_signals["mfi_signal"] = max(-1.0, min(1.0, (mfi - 50) / 50))

        # volume_profile (VWAP proximity)
        vwap_data = ta.get("vwap", {})
        if isinstance(vwap_data, dict) and price > 0:
            upper_1sd = vwap_data.get("upper_1sd", 0)
            lower_1sd = vwap_data.get("lower_1sd", 0)
            if upper_1sd and lower_1sd:
                in_value_area = lower_1sd <= price <= upper_1sd
                ta_signals["volume_profile"] = 0.3 if in_value_area else -0.3

        # liquidation_level
        if hasattr(self, 'liquidation_tracker'):
            fr_val = snapshot.funding_rates.get(ticker, 0)
            oi_data = snapshot.open_interest.get(ticker, {})
            oi_usd = oi_data.get("oi_usd", oi_data.get("value", 0)) if isinstance(oi_data, dict) else 0
            try:
                liq_sig = self.liquidation_tracker.get_signal(ticker, price, oi_usd, fr_val)
                if abs(liq_sig) > 0.1:
                    ta_signals["liquidation_level"] = liq_sig
            except Exception:
                pass

        # volatility_regime (ATR-based)
        atr_pct = ta.get("atr_pct", 0)
        if isinstance(atr_pct, (int, float)) and atr_pct > 0:
            atr_pct_100 = atr_pct * 100 if atr_pct < 1 else atr_pct  # handle fraction vs percent
            if atr_pct_100 > 2.0:
                ta_signals["volatility_regime"] = 0.7
            elif atr_pct_100 < 0.5:
                ta_signals["volatility_regime"] = -0.5

        # dxy_direction
        dxy_dir = getattr(snapshot, 'dxy_direction', '')
        if dxy_dir == 'strengthening':
            ta_signals["dxy_direction"] = -0.5
        elif dxy_dir == 'weakening':
            ta_signals["dxy_direction"] = 0.5

        # etf_flow
        # etf_flow (spot volume anomaly proxy for institutional flow)
        etf = getattr(snapshot, 'etf_daily_flow_usd', 0)
        if isinstance(etf, (int, float)) and abs(etf) > 0:
            # Proxy: volume excess USD. Normalize by 2B (typical daily vol ~2-3B)
            ta_signals["etf_flow"] = max(-1.0, min(1.0, etf / 2_000_000_000))

        # stablecoin_flow (USDT daily supply change)
        sc_chg = getattr(snapshot, 'stablecoin_change_pct', 0)
        if isinstance(sc_chg, (int, float)) and abs(sc_chg) > 0.0002:
            ta_signals["stablecoin_flow"] = max(-1.0, min(1.0, sc_chg * 500))

        # btc_dominance (BTC.D trend — high dominance = alt weakness)
        btc_d = getattr(snapshot, 'btc_dominance_pct', 0)
        if isinstance(btc_d, (int, float)) and btc_d > 0:
            # BTC.D > 55% = BTC strong (alt bearish), < 45% = alt season
            if btc_d > 55:
                ta_signals["btc_dominance"] = min(1.0, (btc_d - 50) / 15)
            elif btc_d < 45:
                ta_signals["btc_dominance"] = max(-1.0, (btc_d - 50) / 15)

        # social_buzz (trending coins excitement level)
        buzz = getattr(snapshot, 'social_buzz_score', 0)
        if isinstance(buzz, (int, float)) and buzz > 0:
            # High buzz (avg |change| > 10%) = extreme sentiment
            if buzz > 15:
                ta_signals["social_buzz"] = 0.7  # extreme buzz, contrarian caution
            elif buzz > 5:
                ta_signals["social_buzz"] = 0.3  # moderate excitement
            elif buzz < 2:
                ta_signals["social_buzz"] = -0.3  # dead market

        # news_sentiment (headline analysis)
        ns = getattr(snapshot, 'news_sentiment_score', 0)
        if isinstance(ns, (int, float)) and abs(ns) > 0.02:
            ta_signals["news_sentiment"] = max(-1.0, min(1.0, ns))

        # whale_activity (large BTC transactions)
        whale = getattr(snapshot, 'whale_activity_score', 0)
        if isinstance(whale, (int, float)) and whale > 0.2:
            ta_signals["whale_activity"] = min(1.0, whale)

        # exchange_flow (exchange netflow — negative = outflow = bullish)
        exflow = getattr(snapshot, 'exchange_netflow_score', 0)
        if isinstance(exflow, (int, float)) and abs(exflow) > 0.1:
            ta_signals["exchange_flow"] = max(-1.0, min(1.0, exflow))

        return ta_signals

    def _save_hold_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Save price+TA snapshot when Claude decides HOLD (no trades).

        Later, _check_counterfactuals() compares current prices to these
        snapshots and feeds missed opportunities into H-TS.
        """
        now = time.time()
        for tic in snapshot.candidates:
            price = snapshot.ticker_prices.get(tic, 0)
            if price <= 0:
                continue
            # Don't snapshot tickers we're already holding
            if tic in self._entry_prices:
                continue

            ta_signals = self._extract_ta_signals(tic, snapshot)

            if ta_signals:
                snap = {
                    "ts": now,
                    "ticker": tic,
                    "price": price,
                    "ta_signals": ta_signals,
                    "regime": snapshot.combined_regime,
                }
                self._hold_snapshots.append(snap)
                # Also save for longer horizons (swing 30min, trend 2h)
                self._swing_snapshots.append(dict(snap))
                self._trend_snapshots.append(dict(snap))

    def _check_counterfactuals(self) -> None:
        """Check HOLD snapshots across multiple time horizons.

        Three horizons capture different missed-opportunity types:
          - Scalp  (3 min,  0.30 weight): quick momentum moves
          - Swing  (30 min, 0.25 weight): dip-buy / mean-reversion
          - Trend  (2 hr,   0.20 weight): regime-shift missed entries

        Each horizon has its own deque. Snapshots expire after their
        check window so they don't accumulate indefinitely.

        Daily CF cap: max 3× real trades to prevent CF noise from
        overwhelming actual trade signals in H-TS learning.
        """
        now = time.time()
        # Daily CF counter reset
        today = time.strftime("%Y-%m-%d")
        if self._cf_count_date != today:
            self._cf_count_date = today
            self._cf_count_today = 0
        # Cap: max 3× real trades per day (minimum 15 to allow early learning)
        real_trades_today = max(5, self.online_learner.total_trades_today if hasattr(self.online_learner, 'total_trades_today') else 5)
        cf_cap = real_trades_today * 3
        if self._cf_count_today >= cf_cap:
            return
        horizons = [
            # (deque,                 min_age_s, max_age_s, discount, label)
            # Scaled for 1h primary: scalp=10-30min, swing=2-6h, trend=12-48h
            (self._hold_snapshots,    600,       1800,      0.30, "scalp"),    # 10-30 min
            (self._swing_snapshots,   7200,      21600,     0.25, "swing"),    # 2-6 hr
            (self._trend_snapshots,   43200,     172800,    0.20, "trend"),    # 12-48 hr
        ]

        for snapshots, min_age, max_age, discount, label in horizons:
            processed = []
            for snap in snapshots:
                age = now - snap["ts"]
                if age < min_age:
                    continue  # too fresh
                if age > max_age:
                    processed.append(snap)  # expired, discard
                    continue

                ticker = snap["ticker"]
                current_price = self._last_tick_prices.get(ticker, 0)
                if current_price <= 0:
                    processed.append(snap)
                    continue

                raw_pnl = (current_price - snap["price"]) / snap["price"]
                if self._cf_count_today >= cf_cap:
                    processed.append(snap)
                    continue
                try:
                    if raw_pnl > 0:
                        # Price went up → missed opportunity (should have bought)
                        self._cf_count_today += 1
                        result = self.online_learner.record_counterfactual(
                            ticker=ticker,
                            price_at_hold=snap["price"],
                            price_now=current_price,
                            ta_signals=snap["ta_signals"],
                            regime=snap["regime"],
                            discount_factor=discount,
                        )
                        if result:
                            logger.info("[counterfactual:%s] %s: hold@$%.2f → now@$%.2f, raw=%+.2f%% (missed buy)",
                                        label, ticker, snap["price"], current_price,
                                        raw_pnl * 100)
                            try:
                                self.diary.record_missed_opportunity(
                                    ticker=ticker,
                                    hold_price=snap["price"],
                                    current_price=current_price,
                                    raw_pnl_pct=raw_pnl,
                                    horizon=label,
                                    regime=snap["regime"],
                                    ta_signals=snap["ta_signals"],
                                    direction="LONG",
                                )
                            except Exception:
                                pass
                    else:
                        # Price went down → two perspectives:
                        # 1. Correct hold on long (avoided buying into drop)
                        self._cf_count_today += 1
                        result = self.online_learner.record_correct_hold(
                            ticker=ticker,
                            price_at_hold=snap["price"],
                            price_now=current_price,
                            ta_signals=snap["ta_signals"],
                            regime=snap["regime"],
                            discount_factor=discount,
                        )
                        if result:
                            logger.info("[correct_hold:%s] %s: hold@$%.2f → now@$%.2f, avoided=%+.2f%%",
                                        label, ticker, snap["price"], current_price,
                                        raw_pnl * 100)
                            try:
                                self.diary.record_correct_hold(
                                    ticker=ticker,
                                    hold_price=snap["price"],
                                    current_price=current_price,
                                    avoided_pct=raw_pnl,
                                    horizon=label,
                                    regime=snap["regime"],
                                    ta_signals=snap["ta_signals"],
                                )
                            except Exception:
                                pass

                        # 2. Missed SHORT opportunity (price dropped = short would profit)
                        #    Feed to H-TS: record_counterfactual already handles
                        #    price_went_up=False → bearish signals get aligned correctly
                        short_pnl = -raw_pnl  # short profit = price drop
                        if short_pnl > 0.005 and self._cf_count_today < cf_cap:  # >0.5% missed short
                            logger.info("[counterfactual:%s] %s: missed SHORT @$%.2f → $%.2f, would-be=%+.2f%%",
                                        label, ticker, snap["price"], current_price,
                                        short_pnl * 100)
                            # Pass original signals — alignment logic already correct:
                            # bearish signal + price_went_up=False → aligned=True → reward
                            try:
                                self.online_learner.record_counterfactual(
                                    ticker=ticker,
                                    price_at_hold=snap["price"],
                                    price_now=current_price,
                                    ta_signals=snap["ta_signals"],
                                    regime=snap["regime"],
                                    discount_factor=discount * 0.5,  # lower weight than long counterfactual
                                )
                            except Exception:
                                pass
                            try:
                                self.diary.record_missed_opportunity(
                                    ticker=ticker,
                                    hold_price=snap["price"],
                                    current_price=current_price,
                                    raw_pnl_pct=short_pnl,
                                    horizon=label,
                                    regime=snap["regime"],
                                    ta_signals=snap["ta_signals"],
                                    direction="SHORT",
                                )
                            except Exception:
                                pass
                except Exception as exc:
                    logger.debug("[counterfactual:%s] Failed: %s", label, exc)

                processed.append(snap)

            for snap in processed:
                try:
                    snapshots.remove(snap)
                except ValueError:
                    pass

    def _check_exit_regrets(self) -> None:
        """Post-exit counterfactual: check if recent exits were premature or well-timed."""

        def price_fetcher(ticker: str) -> float:
            return self._last_tick_prices.get(ticker, 0)

        def on_regret(snap: ExitSnapshot, horizon: str, price_now: float,
                      additional_pnl: float, discount: float) -> None:
            """Premature exit — price continued in our favor."""
            result = self.online_learner.record_exit_regret(
                ticker=snap.ticker,
                exit_price=snap.exit_price,
                price_now=price_now,
                pnl_at_exit=snap.pnl_pct,
                held_hours=snap.held_hours,
                agent_signals=snap.agent_signals,
                regime=snap.regime,
                position_side=snap.position_side,
                discount_factor=discount,
                was_premature=True,
            )
            if result:
                self.diary.record_exit_regret(
                    ticker=snap.ticker,
                    exit_price=snap.exit_price,
                    current_price=price_now,
                    pnl_at_exit=snap.pnl_pct,
                    additional_pnl=additional_pnl,
                    horizon=horizon,
                    regime=snap.regime,
                    agent_signals=snap.agent_signals,
                    held_hours=snap.held_hours,
                    position_side=snap.position_side,
                    was_premature=True,
                )

        def on_validated(snap: ExitSnapshot, horizon: str, price_now: float,
                         additional_pnl: float, discount: float) -> None:
            """Good exit — price reversed after we closed."""
            result = self.online_learner.record_exit_regret(
                ticker=snap.ticker,
                exit_price=snap.exit_price,
                price_now=price_now,
                pnl_at_exit=snap.pnl_pct,
                held_hours=snap.held_hours,
                agent_signals=snap.agent_signals,
                regime=snap.regime,
                position_side=snap.position_side,
                discount_factor=discount,
                was_premature=False,
            )
            if result:
                self.diary.record_exit_regret(
                    ticker=snap.ticker,
                    exit_price=snap.exit_price,
                    current_price=price_now,
                    pnl_at_exit=snap.pnl_pct,
                    additional_pnl=additional_pnl,
                    horizon=horizon,
                    regime=snap.regime,
                    agent_signals=snap.agent_signals,
                    held_hours=snap.held_hours,
                    position_side=snap.position_side,
                    was_premature=False,
                )

        self.exit_regret_tracker.check_exits(price_fetcher, on_regret, on_validated)

    def _find_entry_context(self, ticker: str) -> Dict[str, Any]:
        """Find the last BUY decision for this ticker from decisions.jsonl.

        Scans the last 100 lines in reverse for the most recent BUY entry.
        Returns {reasoning, confidence, signal_weights, learning_note}.
        """
        result: Dict[str, Any] = {}
        if not self._decisions_log.exists():
            return result
        try:
            lines = []
            with open(self._decisions_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
            # Scan last 100 lines in reverse
            for line in reversed(lines[-100:]):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for t in entry.get("trades", []):
                    if t.get("action") in ("BUY", "SHORT") and t.get("ticker", "") == ticker:
                        result["reasoning"] = t.get("reasoning", "")
                        result["confidence"] = t.get("confidence", 0)
                        result["signal_weights"] = t.get("signal_weights", {})
                        result["learning_note"] = entry.get("learning_note", "")
                        return result
        except Exception as exc:
            logger.debug("[diary] _find_entry_context failed: %s", exc)
        return result

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
            ctx = self.derivatives_monitor.get_context()
            ctx["_ts"] = time.time()  # timestamp for staleness detection
            self._derivatives_context = ctx
            logger.info("[runner] Funding extreme stored: %s", event.payload.get("ticker"))

        async def _on_oi_spike(event):
            ctx = self.derivatives_monitor.get_context()
            ctx["_ts"] = time.time()
            self._derivatives_context = ctx
            logger.info("[runner] OI spike stored: %s", event.payload.get("ticker"))

        self.event_bus.subscribe("market.tick", _on_tick)
        self.event_bus.subscribe("market.funding_extreme", _on_funding_extreme)
        self.event_bus.subscribe("market.oi_spike", _on_oi_spike)
        self.event_bus.subscribe("market.force_order", self.liquidation_tracker.on_force_order)

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
            # Lock protects _entry_prices from concurrent access after fire-and-forget EventBus
            async with self._lock:
                data = event.payload
                ticker = data.get("ticker", "")
                source = data.get("source", "unknown")
                pnl_pct = data.get("pnl_pct", 0)
                held_hours = data.get("held_hours", 0)
                exit_price = data.get("exit_price", 0)
                reason = data.get("reason", "")
                mfe = data.get("mfe", 0)
                mae = data.get("mae", 0)
                capture_ratio = data.get("capture_ratio", 0)

                # Accumulate realized PnL for portfolio value tracking
                entry_px = self._entry_prices.get(ticker, data.get("entry_price", 0))
                tracked_pos = self.position_tracker.get_position(ticker)
                qty = (tracked_pos.qty if tracked_pos and tracked_pos.qty > 0
                       else data.get("qty", 0))
                if entry_px > 0 and qty > 0:
                    position_usd = qty * entry_px
                    self._realized_pnl_usd += pnl_pct * position_usd

                # Clean up _entry_prices for ANY exit (safety or claude)
                self._entry_prices.pop(ticker, None)
                # Anti-churn cooldown for safety exits too
                self._exit_cooldowns[ticker] = time.time() + 30  # 30s cooldown — agent decides re-entry
                self._save_state()

                # Log safety exits to CSV (claude exits are logged in _run_decision)
                if source != "claude_agent":
                    dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                    exit_qty = qty  # from event payload or tracked position
                    exit_value = exit_price * exit_qty if exit_qty > 0 else 0
                    # Determine exit action label based on position side
                    exit_action = "COVER" if (tracked_pos and tracked_pos.side == "short") else "SELL"
                    self._log_trade(
                        exit_action, ticker, exit_price, exit_qty, exit_value,
                        pnl_pct=pnl_pct, held_hours=held_hours,
                        reason=reason, source=f"safety_{source}",
                        dry_run=dry_run,
                    )

                # Compute regime unconditionally (needed by neo4j + diary even without agent_signals)
                combined_regime = "unknown"
                try:
                    crypto_result = await asyncio.to_thread(self.crypto_regime_detector.detect)
                    crypto_regime = crypto_result.get("regime_label", "unknown")
                    macro_regime = self.macro_detector.regime.value
                    # Trend direction from BTC 1h (same logic as snapshot builder)
                    from core.agent_tools import _calc_ema, _calc_supertrend
                    _exit_trend = "ranging"
                    _btc_bars = self.ohlcv_store.get_bars("BTC/USDT:USDT", "1h", 200)
                    if _btc_bars and len(_btc_bars) >= 21:
                        _closes = [b[4] for b in _btc_bars if b[4] > 0]
                        if len(_closes) >= 21:
                            _ema9 = _calc_ema(_closes, 9)
                            _ema21 = _calc_ema(_closes, 21)
                            _st = _calc_supertrend(_btc_bars, period=10, multiplier=3.0)
                            if _ema9 < _ema21 and _st < 0:
                                _exit_trend = "downtrend"
                            elif _ema9 > _ema21 and _st > 0:
                                _exit_trend = "uptrend"
                    combined_regime = f"{crypto_regime}_{_exit_trend}_{macro_regime}"
                except Exception:
                    pass

                # Record Claude's reasoning category as the "agent signal"
                agent_signals = data.get("agent_signals", {})
                if not agent_signals:
                    for ev in self.event_bus.get_recent_events(100):
                        if (ev.get("topic") == "decision.signal"
                                and ev.get("payload", {}).get("ticker") == ticker):
                            agent_signals = ev["payload"].get("signals", {})
                            break
                # Use position_side from event payload FIRST (reliable: set before untrack).
                # Fallback to tracked_pos (may be None if fire-and-forget handler runs late).
                pos_side = data.get("position_side") or (tracked_pos.side if tracked_pos else "long")
                # Pop entry metadata once (used by both record_trade and exit regret)
                meta = self._entry_meta.pop(ticker, {})
                ta_snap = meta.get("ta_snapshot", {})
                # Build merged_signals unconditionally (needed by exit regret tracker)
                # For safety exits: agent_signals={}, so merged = ta_snapshot only
                merged_signals = dict(ta_snap)
                merged_signals.update(agent_signals)

                if merged_signals:
                    self.online_learner.record_trade(
                        ticker=ticker,
                        entry_price=data.get("entry_price", 0),
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        held_hours=held_hours,
                        agent_signals=merged_signals,
                        market_type="crypto",
                        regime=combined_regime,
                        position_pct_used=meta.get("position_pct", 0.0),
                        confidence_at_entry=meta.get("confidence", 0.0),
                        position_side=pos_side,
                        ta_snapshot=None,  # already merged, no virtual needed
                        mfe=mfe,
                        mae=mae,
                        capture_ratio=capture_ratio,
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

                # Layer 1: Per-trade reflection (Reflexion + Chain of Hindsight)
                try:
                    entry_context = self._find_entry_context(ticker)
                    self.diary.record_reflection(
                        ticker=ticker,
                        entry_price=data.get("entry_price", 0),
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        held_hours=held_hours,
                        regime=combined_regime,
                        exit_reason=reason,
                        exit_source=source,
                        agent_signals=agent_signals,
                        entry_context=entry_context,
                        position_side=pos_side,
                        mfe=mfe,
                        mae=mae,
                        capture_ratio=capture_ratio,
                    )
                except Exception as exc:
                    logger.debug("[diary] Reflection failed: %s", exc)

                # Post-exit counterfactual: track price after exit for regret learning
                try:
                    exit_snap = ExitSnapshot(
                        ts=time.time(),
                        ticker=ticker,
                        exit_price=exit_price,
                        entry_price=data.get("entry_price", 0),
                        pnl_pct=pnl_pct,
                        held_hours=held_hours,
                        position_side=pos_side,
                        agent_signals=dict(merged_signals),
                        regime=combined_regime,
                        exit_reason=reason,
                    )
                    self.exit_regret_tracker.record_exit(exit_snap)
                except Exception as exc:
                    logger.debug("[exit-regret] Record failed: %s", exc)

        self.event_bus.subscribe("position.exit", _on_exit)

    # ------------------------------------------------------------------
    # Adaptive gate — tick handler
    # ------------------------------------------------------------------

    async def _handle_tick(self, payload: dict) -> None:
        """Every WS tick → update aggregator → gate evaluates.

        Two evaluation paths to balance data quality vs responsiveness:
          1. Candle close: full features (price_change_pct, volume, btc_price)
          2. Between candles: BTC ticks only → btc_price z-score (flash crash/pump)

        This prevents z-score pollution (duplicates, zero-dominated windows)
        while keeping standalone z-score wakes for extreme BTC moves.
        """
        ticker = payload.get("ticker", "")
        price = payload.get("price", 0)
        interval = payload.get("interval", "")
        is_closed = payload.get("is_closed", False)

        if price <= 0:
            return

        # Always cache latest prices (position tracker, snapshot builder, etc.)
        self._last_tick_prices[ticker] = price
        if "BTC" in ticker:
            self._last_tick_prices["_btc_price"] = price

        # Update multi-TF aggregator with closed candles (all intervals)
        if is_closed and interval:
            self.multi_tf_aggregator.update(ticker, interval, payload)

        # Gate interval filter: 5m/15m + primary (1h close = full analysis)
        primary = EVENT_CONFIG.get("primary_interval", "1h")
        _gate_intervals = {"5m", "15m", primary}
        if interval and interval not in _gate_intervals:
            return

        if is_closed:
            # Path 1: Candle close — full feature evaluation
            # price_change_pct = close-to-close, volume = candle volume
            features = self._compute_tick_features(ticker, price, payload)
            should_wake, reasons = self.adaptive_gate.evaluate(
                features, is_candle_close=True)
            if should_wake:
                logger.info("[runner] Gate wake (%s) -> Claude deciding",
                            ", ".join(reasons))
                try:
                    await self._run_decision(
                        trigger="candle_close", event=payload,
                        wake_reasons=reasons)
                except Exception as exc:
                    logger.error("[decide] Failed on gate wake: %s",
                                 exc, exc_info=True)
        elif "BTC" in ticker:
            # Path 2: Between candles — BTC price z-score only
            # Only actual BTC ticks (no duplicates from ETH/SOL/PAXG cache)
            # Catches flash crashes/pumps before next candle close
            should_wake, reasons = self.adaptive_gate.evaluate(
                {"btc_price": price}, is_candle_close=False)
            if should_wake:
                logger.info("[runner] Gate wake (%s) -> Claude deciding",
                            ", ".join(reasons))
                try:
                    await self._run_decision(
                        trigger="gate", event=payload,
                        wake_reasons=reasons)
                except Exception as exc:
                    logger.error("[decide] Failed on gate wake: %s",
                                 exc, exc_info=True)

    def _compute_tick_features(self, ticker: str, price: float, payload: dict) -> Dict[str, float]:
        """Extract features from candle close for gate evaluation.

        Called ONLY on candle close — each feature represents a real market
        state change, not intermediate WS noise.

        Key fixes vs previous tick-based approach:
          - price_change_pct: close-to-close (not tick-to-tick with 90% zeros)
          - btc_price: only from BTC ticks (not N duplicates from every ticker)
        """
        features: Dict[str, float] = {}
        prefix = ticker  # e.g. "BTC/USDT"

        # Price change: candle-close to candle-close (not tick-to-tick)
        prev_close = self._last_close_prices.get(ticker, price)
        if prev_close > 0:
            features[f"{prefix}:price_change_pct"] = (price - prev_close) / prev_close
        else:
            features[f"{prefix}:price_change_pct"] = 0.0
        self._last_close_prices[ticker] = price

        # BTC price — only from actual BTC candle closes
        # (prevents same cached value being fed N times per real change)
        if "BTC" in ticker:
            features["btc_price"] = price

        # Volume (per-ticker, meaningful on candle close)
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
    _trending_cache: tuple = (0, "", 0.0)  # (timestamp, summary, buzz_score)
    _etf_flow_cache: Dict = {}
    _etf_flow_cache_time: float = 0.0
    _btc_dom_cache: tuple = (0, 0.0)  # (timestamp, dominance_pct)
    _news_sent_cache: tuple = (0, 0.0)  # (timestamp, sentiment_score)
    _whale_cache: tuple = (0, 0.0)  # (timestamp, whale_score)
    _exchange_flow_cache: tuple = (0, 0.0)  # (timestamp, netflow_score)

    def _estimate_portfolio_value(self) -> float:
        """Estimate portfolio value from WS prices (dry_run mode, no REST).

        Uses actual broker balance (synced every 10min) as base, falling back
        to initial_balance + realized_pnl if broker fetch fails.

        Side-aware:
        - Long: cash -= entry_value, current_value = qty * current_px
        - Short: cash -= margin (= entry_value), unrealized_pnl = (entry - current) * qty
          → portfolio = base - margin + margin + unrealized_pnl = base + unrealized_pnl
        """
        # Sync broker balance every 10 minutes
        now = time.time()
        if now - self._broker_balance_time > 600:
            try:
                bal = self.broker._exchange.fetch_balance()
                total = bal.get("total", {})
                usdt = float(total.get("USDT", 0))
                usdc = float(total.get("USDC", 0))
                broker_bal = usdt + usdc
                if broker_bal > 0:
                    self._broker_balance = broker_bal
                    self._broker_balance_time = now
                    logger.info("[portfolio] Synced broker balance: $%.2f (USDT=%.2f, USDC=%.2f)", broker_bal, usdt, usdc)
            except Exception as exc:
                logger.debug("[portfolio] Broker balance fetch failed: %s", exc)

        # Use broker balance if available, otherwise fallback to internal tracking
        if self._broker_balance > 0:
            base = self._broker_balance
        else:
            base = self._initial_balance + self._realized_pnl_usd
        unrealized_pnl = 0.0
        for tic, entry_px in self._entry_prices.items():
            tracked = self.position_tracker.get_position(tic)
            if tracked and tracked.qty > 0:
                current_px = tracked.current_price if tracked.current_price > 0 else self._last_tick_prices.get(tic, entry_px)
                if tracked.side == "short":
                    # Short PnL: profit when price drops
                    unrealized_pnl += (entry_px - current_px) * tracked.qty
                else:
                    # Long PnL: profit when price rises
                    unrealized_pnl += (current_px - entry_px) * tracked.qty
        return base + unrealized_pnl

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

    def _fetch_trending(self) -> tuple:
        """Fetch CoinGecko trending coins as market buzz + social_buzz score. Cached for 1 hour.

        Returns (summary_text, buzz_score).
        buzz_score: average |price_change_24h| of top 5 trending coins.
        High buzz_score = market excitement, low = calm.
        """
        import requests
        now = time.time()
        if now - self._trending_cache[0] < 3600 and self._trending_cache[1]:
            return self._trending_cache[1], self._trending_cache[2]
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=5)
            coins = resp.json().get("coins", [])[:5]
            if not coins:
                return self._trending_cache[1], self._trending_cache[2]
            parts = []
            price_changes = []
            for c in coins:
                item = c.get("item", {})
                name = item.get("name", "")
                symbol = item.get("symbol", "")
                rank = item.get("score", 0) + 1
                price_change = item.get("data", {}).get("price_change_percentage_24h", {}).get("usd", 0)
                if price_change:
                    parts.append(f"#{rank} {symbol}({name}) {price_change:+.1f}%")
                    price_changes.append(price_change)
                else:
                    parts.append(f"#{rank} {symbol}({name})")
            summary = "Trending: " + ", ".join(parts)
            # Buzz score: avg absolute price change of trending coins
            # High = market is excited (trending coins moving a lot)
            buzz_score = sum(abs(p) for p in price_changes) / len(price_changes) if price_changes else 0.0
            self._trending_cache = (now, summary, buzz_score)
            return summary, buzz_score
        except Exception as exc:
            logger.debug("[data] Trending fetch failed: %s", exc)
            return self._trending_cache[1], self._trending_cache[2]

    def _fetch_etf_flow(self) -> Dict:
        """Estimate ETF-like institutional flow via Binance spot volume anomaly.

        ETF inflows correlate with unusually high spot volume relative to recent
        average. Uses Binance BTC/USDT spot 24h volume vs 7-day average.
        Cached for 4 hours.
        """
        import requests
        now = time.time()
        if now - self._etf_flow_cache_time < 14400 and self._etf_flow_cache:
            return self._etf_flow_cache

        result = {"daily_flow_usd": 0.0, "weekly_flow_usd": 0.0, "label": "neutral"}
        try:
            # Get current 24h volume
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": "BTCUSDT"},
                timeout=5,
            )
            if resp.status_code != 200:
                return self._etf_flow_cache or result
            current_vol = float(resp.json().get("quoteVolume", 0))

            # Get 7-day klines for average daily volume
            klines = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1d", "limit": 8},
                timeout=5,
            )
            if klines.status_code == 200:
                bars = klines.json()
                if len(bars) >= 2:
                    # Use bars[:-1] as historical (exclude current incomplete day)
                    hist_vols = [float(b[7]) for b in bars[:-1]]  # quoteAssetVolume
                    avg_vol = sum(hist_vols) / len(hist_vols) if hist_vols else current_vol
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

                    # Convert volume ratio to "flow" estimate
                    # ratio > 1.2 = above-average volume = institutional inflow proxy
                    # ratio < 0.8 = below-average = outflow/reduced interest
                    excess_usd = (vol_ratio - 1.0) * avg_vol  # excess in USD
                    daily_flow = excess_usd if abs(vol_ratio - 1.0) > 0.05 else 0.0

                    if daily_flow > 1e8:
                        label = "inflow"
                    elif daily_flow < -1e8:
                        label = "outflow"
                    else:
                        label = "neutral"

                    result = {
                        "daily_flow_usd": daily_flow,
                        "weekly_flow_usd": daily_flow,
                        "label": label,
                        "vol_ratio": round(vol_ratio, 3),
                    }
        except Exception as exc:
            logger.debug("[data] ETF flow proxy fetch failed: %s", exc)

        self._etf_flow_cache = result
        self._etf_flow_cache_time = now
        return result

    def _fetch_btc_dominance(self) -> float:
        """Fetch BTC dominance % from CoinGecko /global. Cached for 1 hour. Free, no key."""
        import requests
        now = time.time()
        if now - self._btc_dom_cache[0] < 3600 and self._btc_dom_cache[1] > 0:
            return self._btc_dom_cache[1]
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
            data = resp.json().get("data", {})
            btc_d = data.get("market_cap_percentage", {}).get("btc", 0.0)
            if btc_d > 0:
                self._btc_dom_cache = (now, btc_d)
            return btc_d
        except Exception as exc:
            logger.debug("[data] BTC dominance fetch failed: %s", exc)
            return self._btc_dom_cache[1]

    def _fetch_news_sentiment(self) -> float:
        """Fetch crypto news sentiment from CryptoCompare API. Cached for 1 hour.

        Returns sentiment score [-1, +1]. Positive = bullish headlines.
        """
        import requests
        now = time.time()
        if now - self._news_sent_cache[0] < 3600:
            return self._news_sent_cache[1]
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN"},
                timeout=5,
            )
            data = resp.json() if resp.status_code == 200 else {}
            articles = data.get("Data", [])
            if not articles or not isinstance(articles, list):
                return self._news_sent_cache[1]
            # Simple sentiment: count positive/negative keywords in titles
            pos_words = {"surge", "rally", "bull", "soar", "gain", "rise", "jump", "high", "up", "breakout", "pump", "buy"}
            neg_words = {"crash", "drop", "bear", "fall", "dump", "low", "down", "sell", "fear", "plunge", "risk", "loss"}
            pos, neg = 0, 0
            for art in articles[:30]:
                title = (art.get("title", "") or "").lower()
                pos += sum(1 for w in pos_words if w in title)
                neg += sum(1 for w in neg_words if w in title)
            total = pos + neg
            score = (pos - neg) / total if total > 0 else 0.0
            self._news_sent_cache = (now, score)
            return score
        except Exception as exc:
            logger.debug("[data] News sentiment fetch failed: %s", exc)
            return self._news_sent_cache[1]

    def _fetch_whale_activity(self) -> float:
        """Detect large BTC transactions via blockchain.info mempool. Cached for 4 hours.

        Returns whale score [0, +1]. Higher = more large tx activity.
        Free, no API key, no rate limit issues.
        """
        import requests
        now = time.time()
        if now - self._whale_cache[0] < 14400:
            return self._whale_cache[1]
        try:
            resp = requests.get(
                "https://blockchain.info/unconfirmed-transactions",
                params={"format": "json"},
                timeout=10,
            )
            txs = resp.json().get("txs", []) if resp.status_code == 200 else []
            if not txs:
                return self._whale_cache[1]
            # Count large transactions (>1 BTC output total in satoshi)
            large_count = sum(
                1 for tx in txs
                if sum(o.get("value", 0) for o in tx.get("out", [])) > 100_000_000  # >1 BTC
            )
            # Score: 0-20+ large txs in recent mempool → 0.0-1.0
            score = min(1.0, large_count / 10.0)
            self._whale_cache = (now, score)
            return score
        except Exception as exc:
            logger.debug("[data] Whale activity fetch failed: %s", exc)
            return self._whale_cache[1]

    def _fetch_exchange_netflow(self) -> float:
        """Estimate exchange netflow via Binance futures/spot volume ratio.

        High futures/spot ratio = speculative activity = coins on exchange (bearish).
        Low ratio = spot accumulation = coins leaving exchange (bullish).
        Cached for 4 hours.
        """
        import requests
        now = time.time()
        if now - self._exchange_flow_cache[0] < 14400:
            return self._exchange_flow_cache[1]
        try:
            r_spot = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": "BTCUSDT"}, timeout=5,
            )
            r_fut = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                params={"symbol": "BTCUSDT"}, timeout=5,
            )
            if r_spot.status_code == 200 and r_fut.status_code == 200:
                spot_vol = float(r_spot.json().get("quoteVolume", 1))
                fut_vol = float(r_fut.json().get("quoteVolume", 1))
                ratio = fut_vol / max(spot_vol, 1e6)
                # Normal ratio ~5-8. >10 = heavy speculation (bearish), <4 = spot driven (bullish)
                # Center around 7, normalize
                score = max(-1.0, min(1.0, (ratio - 7.0) / 5.0))
                # Positive score = more futures activity = bearish netflow (coins on exchange)
                # Negative score = more spot activity = bullish (accumulation)
                self._exchange_flow_cache = (now, score)
                return score
            return self._exchange_flow_cache[1]
        except Exception as exc:
            logger.debug("[data] Exchange netflow proxy fetch failed: %s", exc)
            return self._exchange_flow_cache[1]

    async def _bootstrap_ohlcv(self) -> None:
        """One-time REST call to load historical OHLCV into the WS store.

        Bootstraps multiple timeframes:
        - Primary (1h): 1440 bars (60 days)
        - 15m: 400 bars (~4 days) — if in kline_intervals
        - 5m: 300 bars (~1 day) — if in kline_intervals

        After this, MarketListener WS candle closes keep all TFs updated.
        Zero REST OHLCV calls after bootstrap.
        """
        cfg = ACTIVE_BLEND_CONFIG
        timeframe = cfg.get("ohlcv_timeframe", "1d")
        bars_per_day = {"4h": 6, "2h": 12, "1h": 24, "1d": 1}.get(timeframe, 1)
        limit = min(cfg["ohlcv_lookback_days"] * bars_per_day, 1000)

        bootstrap_tfs = [(timeframe, limit)]
        kline_intervals = EVENT_CONFIG.get("kline_intervals", [])
        if "15m" in kline_intervals:
            bootstrap_tfs.append(("15m", 400))
        if "5m" in kline_intervals:
            bootstrap_tfs.append(("5m", 300))

        for tic in cfg["tickers"]:
            for tf, tf_limit in bootstrap_tfs:
                if self.ohlcv_store.is_bootstrapped(tic, tf):
                    continue
                try:
                    ohlcv = await asyncio.to_thread(
                        self.broker.fetch_ohlcv, tic, tf, tf_limit
                    )
                    if ohlcv:
                        self.ohlcv_store.bootstrap(tic, tf, ohlcv)
                except Exception as e:
                    logger.warning("[bootstrap] OHLCV %s/%s failed: %s", tic, tf, e)

        total = sum(
            self.ohlcv_store.bar_count(tic, tf)
            for tic in cfg["tickers"] for tf, _ in bootstrap_tfs
        )
        logger.info("[bootstrap] OHLCV: %d bars (%s). WS takes over.",
                    total, "/".join(tf for tf, _ in bootstrap_tfs))

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

        # Price action — use real-time WS prices as PRIMARY source
        # OHLCV candle close is for returns calculation only
        ticker_prices = {}
        ticker_returns_4h = {}
        ticker_returns_24h = {}
        ticker_volumes = getattr(self, "_latest_volumes", {})
        # Determine bar duration to compute correct lookback indices
        tf = cfg.get("ohlcv_timeframe", "1h")
        _tf_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
        bar_hours = _tf_hours.get(tf, 1)
        bars_4h = max(1, int(4 / bar_hours))    # 1h bars → 4
        bars_24h = max(1, int(24 / bar_hours))   # 1h bars → 24
        for tic in candidates:
            series = close_df[tic].dropna()
            # Real-time WS price first, candle close as fallback
            ws_price = self._last_tick_prices.get(tic, 0)
            ohlcv_price = float(series.iloc[-1]) if len(series) >= 1 else 0

            # Cross-validate WS vs OHLCV — reject corrupted WS price
            if ws_price > 0 and ohlcv_price > 0:
                divergence = abs(ws_price - ohlcv_price) / ohlcv_price
                if divergence > 0.05:  # >5% divergence = WS price likely corrupted
                    logger.warning(
                        "[snapshot] PRICE DIVERGENCE %s: WS=%.2f OHLCV=%.2f (%.1f%%) — using OHLCV",
                        tic, ws_price, ohlcv_price, divergence * 100,
                    )
                    ticker_prices[tic] = ohlcv_price  # trust candle close over corrupted WS
                else:
                    ticker_prices[tic] = ws_price  # normal: use real-time WS
            elif ws_price > 0:
                ticker_prices[tic] = ws_price
            else:
                ticker_prices[tic] = ohlcv_price
            # Returns use candle close — correct lookback for actual timeframe
            if len(series) > bars_4h:
                ticker_returns_4h[tic] = (float(series.iloc[-1]) / float(series.iloc[-1 - bars_4h])) - 1
            if len(series) > bars_24h:
                ticker_returns_24h[tic] = (float(series.iloc[-1]) / float(series.iloc[-1 - bars_24h])) - 1

        # BTC reference
        btc_ticker = cfg["tickers"][0]
        btc_price = ticker_prices.get(btc_ticker, 0)
        if btc_price <= 0:
            btc_price = self._last_tick_prices.get(btc_ticker, 0)
        btc_24h = ticker_returns_24h.get(btc_ticker, 0.0)
        # Compute btc_1h from OHLCV directly (1 bar back for 1h timeframe)
        btc_series = close_df[btc_ticker].dropna() if btc_ticker in close_df.columns else pd.Series()
        bars_1h = max(1, int(1 / bar_hours))
        if len(btc_series) > bars_1h:
            btc_1h = (float(btc_series.iloc[-1]) / float(btc_series.iloc[-1 - bars_1h])) - 1
        else:
            btc_1h = 0.0

        # Derivatives — expire stale context (>5 min old)
        deriv = self._derivatives_context
        if deriv:
            ctx_age = time.time() - deriv.get("_ts", 0)
            if ctx_age > 300:  # >5 min
                logger.debug("[snapshot] Derivatives context expired (%.0fs old), refreshing", ctx_age)
                deriv = self.derivatives_monitor.get_context()
        else:
            deriv = self.derivatives_monitor.get_context()

        # Regime + sentiment — all independent, run in parallel
        _gather_results = await asyncio.gather(
            asyncio.to_thread(self.crypto_regime_detector.detect),   # [0]
            asyncio.to_thread(self.macro_detector.detect),           # [1]
            asyncio.to_thread(self._fetch_fear_greed),               # [2]
            asyncio.to_thread(self._fetch_trending),                 # [3]
            asyncio.to_thread(self._fetch_etf_flow),                 # [4]
            asyncio.to_thread(self._fetch_btc_dominance),            # [5]
            asyncio.to_thread(self._fetch_news_sentiment),           # [6]
            asyncio.to_thread(self._fetch_whale_activity),           # [7]
            asyncio.to_thread(self._fetch_exchange_netflow),         # [8]
            return_exceptions=True,
        )
        # Safely unpack — use defaults if any sub-task failed
        crypto_result = _gather_results[0] if not isinstance(_gather_results[0], Exception) else {"regime_label": "unknown"}
        macro_regime = _gather_results[1] if not isinstance(_gather_results[1], Exception) else type("R", (), {"value": "unknown"})()
        fg_result = _gather_results[2] if not isinstance(_gather_results[2], Exception) else (0, "")
        fg_index, fg_label = fg_result if isinstance(fg_result, tuple) else (0, "")
        trending_result = _gather_results[3] if not isinstance(_gather_results[3], Exception) else ("", 0.0)
        trending_summary, social_buzz_score = trending_result if isinstance(trending_result, tuple) else (trending_result or "", 0.0)
        etf_flow = _gather_results[4] if not isinstance(_gather_results[4], Exception) else {}
        btc_dominance = _gather_results[5] if not isinstance(_gather_results[5], (Exception, type(None))) else 0.0
        news_sentiment = _gather_results[6] if not isinstance(_gather_results[6], (Exception, type(None))) else 0.0
        whale_activity = _gather_results[7] if not isinstance(_gather_results[7], (Exception, type(None))) else 0.0
        exchange_netflow = _gather_results[8] if not isinstance(_gather_results[8], (Exception, type(None))) else 0.0
        # Log failures
        for _i, _r in enumerate(_gather_results):
            if isinstance(_r, Exception):
                logger.warning("[snapshot] gather task %d failed: %s", _i, _r)
        crypto_regime = crypto_result.get("regime_label", "unknown")
        macro_ctx = self.macro_detector.get_context()

        # H-TS sampling deferred to after TA computation (trend-aware regime at line ~1922)
        learner_status = self.online_learner.get_status()

        # Recent trades
        recent_trades = learner_status.get("recent_trades", [])

        # Position PnL enrichment (include tracker data for Claude context)
        enriched_positions = {}
        for tic, pos in positions.items():
            current_px = ticker_prices.get(tic, 0) or self._last_tick_prices.get(tic, 0)
            entry_px = pos.get("entry_price", 0)
            tracked = self.position_tracker.get_position(tic)
            # Side-aware PnL
            if tracked and tracked.side == "short":
                gross_pnl = (entry_px - current_px) / entry_px if entry_px > 0 else 0
            else:
                gross_pnl = (current_px / entry_px - 1) if entry_px > 0 else 0
            pnl_pct = gross_pnl - ROUND_TRIP_FEE  # net after fees
            enriched_positions[tic] = {
                **pos,
                "side": tracked.side if tracked else "long",
                "current_price": current_px,
                "pnl_pct": pnl_pct,
                "trailing_high": tracked.trailing_high if tracked else current_px,
                "trailing_low": tracked.trailing_low if tracked else current_px,
                "held_hours": tracked.held_hours if tracked else 0,
                "mfe": tracked.mfe if tracked else 0,
                "mae": tracked.mae if tracked else 0,
                "capture_ratio": tracked.capture_ratio if tracked else 0,
            }

        # Cash
        cash = pv - sum(
            p["qty"] * (ticker_prices.get(t, 0) or self._last_tick_prices.get(t, 0))
            for t, p in positions.items()
        )
        cash = max(cash, 0)

        # Multi-TF summaries
        multi_tf_summary = {}
        for tic in candidates:
            tf_text = self.multi_tf_aggregator.format_for_prompt(tic)
            if tf_text:
                multi_tf_summary[tic] = tf_text

        # Pre-computed TA indicators (eliminates 2-3 tool calls per decision)
        from core.agent_tools import (
            _calc_rsi, _calc_stoch_rsi, _calc_macd,
            _calc_ema, _calc_bollinger, _calc_atr, _calc_vwap,
            _calc_supertrend, _calc_mfi, _calc_obv_direction,
        )
        pre_computed_ta: Dict[str, Dict] = {}
        for tic in candidates:
            tic_ta: Dict[str, Dict] = {}
            for interval in ("5m", "15m", "1h"):
                bars = self.ohlcv_store.get_bars(tic, interval, 200)
                if not bars or len(bars) < 5:
                    continue
                closes = [b[4] for b in bars if b[4] > 0]
                if len(closes) < 5:
                    continue
                ind: Dict[str, Any] = {}
                ind["rsi"] = round(_calc_rsi(closes, 7), 2)
                rsi_14 = round(_calc_rsi(closes, 14), 2)
                ind["rsi_14"] = rsi_14
                ind["rsi_divergence"] = round(ind["rsi"] - rsi_14, 2)
                ind["stoch_rsi"] = round(_calc_stoch_rsi(closes), 2)
                ind["macd"] = _calc_macd(closes)
                ema9 = _calc_ema(closes, 9)
                ema21 = _calc_ema(closes, 21)
                gap = (ema9 - ema21) / ema21 if ema21 > 0 else 0
                ind["ema_cross"] = {
                    "status": "bullish" if ema9 > ema21 else "bearish",
                    "ema_9": round(ema9, 2), "ema_21": round(ema21, 2),
                    "gap_pct": round(gap, 6),
                }
                ind["bollinger"] = _calc_bollinger(closes)
                atr = _calc_atr(bars)
                ind["atr"] = round(atr, 2)
                ind["atr_pct"] = round(atr / closes[-1], 6) if closes[-1] > 0 else 0
                atr_pct_val = ind["atr_pct"]
                if atr_pct_val > 0:
                    ind["atr_sl"] = round(1.5 * atr_pct_val, 6)
                    ind["atr_tp"] = round(2.0 * atr_pct_val, 6)
                    ind["atr_trail"] = round(1.0 * atr_pct_val, 6)
                    ind["fee_atr_ratio"] = round(0.001 / atr_pct_val, 3)
                ind["vwap"] = _calc_vwap(bars, n=50)
                # EMA(50) for ema_cross_slow signal
                ind["ema_50"] = round(_calc_ema(closes, 50), 2) if len(closes) >= 50 else None
                # Supertrend (ATR×3 based)
                ind["supertrend_dir"] = _calc_supertrend(bars, period=10, multiplier=3.0)
                # MFI (Money Flow Index)
                ind["mfi"] = round(_calc_mfi(bars, period=14), 2)
                # OBV direction (10-bar divergence)
                volumes = [b[5] for b in bars if b[5] >= 0]
                ind["obv_direction"] = _calc_obv_direction(closes, volumes, lookback=10)
                # Volume ratio (current bar vs 20-bar avg)
                if len(volumes) >= 20 and volumes[-1] > 0:
                    avg_vol = sum(volumes[-20:]) / 20
                    ind["volume"] = {"ratio": round(volumes[-1] / avg_vol, 3) if avg_vol > 0 else 1.0}
                ind["last_close"] = round(closes[-1], 2)
                # Sparkline data: last 20 close prices + RSI(14) trajectory
                # (Agent Trading Arena, EMNLP 2025 — LLMs reason better with visual patterns)
                _spark_n = 20
                ind["recent_closes"] = [round(c, 2) for c in closes[-_spark_n:]]
                if len(closes) >= 14 + _spark_n:
                    # O(N) single-pass RSI trajectory (was O(N²) calling _calc_rsi 20x)
                    _period = 14
                    _deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
                    _ag = sum(max(d, 0) for d in _deltas[:_period]) / _period
                    _al = sum(max(-d, 0) for d in _deltas[:_period]) / _period
                    _emit_start = len(_deltas) - _spark_n
                    _rsi_hist = []
                    for _j, _d in enumerate(_deltas[_period:], start=_period):
                        _ag = (_ag * (_period - 1) + max(_d, 0)) / _period
                        _al = (_al * (_period - 1) + max(-_d, 0)) / _period
                        if _j >= _emit_start:
                            _rval = 100.0 if _al == 0 else 100.0 - 100.0 / (1.0 + _ag / _al)
                            _rsi_hist.append(round(_rval, 1))
                    if _rsi_hist:
                        ind["rsi_history"] = _rsi_hist
                tic_ta[interval] = ind
            if tic_ta:
                pre_computed_ta[tic] = tic_ta

        # Trend direction from BTC 1h TA → inject into combined_regime
        # This lets H-TS learn separate strategies for uptrend vs downtrend
        btc_1h_ta = pre_computed_ta.get("BTC/USDT:USDT", {}).get("1h", {})
        btc_ema_status = btc_1h_ta.get("ema_cross", {}).get("status", "unknown")
        btc_st_dir = btc_1h_ta.get("supertrend_dir", 0)
        if btc_ema_status == "bearish" and btc_st_dir < 0:
            trend_dir = "downtrend"
        elif btc_ema_status == "bullish" and btc_st_dir > 0:
            trend_dir = "uptrend"
        else:
            trend_dir = "ranging"
        combined_regime = f"{crypto_regime}_{trend_dir}_{macro_regime.value}"
        self._last_combined_regime = combined_regime
        # Re-sample H-TS with trend-aware regime (more granular learning)
        ts_mean_weights = self.online_learner.get_signal_reliability(regime=combined_regime)
        ts_group_weights = self.online_learner.get_group_weights(regime=combined_regime)
        ts_meta_params = self.online_learner.sample_meta_params(regime=combined_regime)

        # Regime info for prompt
        ts_regime_info = ""
        regime_data = learner_status.get("regime_info", {}).get(combined_regime, {})
        if regime_data.get("using_own_weights"):
            ts_regime_info = f"Regime '{combined_regime}' has {regime_data['trade_count']} trades — using regime-specific weights"
        else:
            ts_regime_info = f"Regime '{combined_regime}' has <3 trades — using global weights as fallback"

        # Historical insights from Neo4j (non-blocking)
        historical_insights = ""
        try:
            historical_insights = await asyncio.to_thread(
                self.memory.build_historical_insights, combined_regime, candidates
            )
        except Exception:
            pass

        # Trading diary context (self-reflection memory)
        diary_context = ""
        try:
            diary_context = self.diary.get_prompt_context(
                regime=combined_regime, tickers=candidates)
        except Exception:
            pass

        # Liquidation heatmap context (per-ticker, from LiquidationTracker)
        liq_contexts: Dict[str, Dict] = {}
        for tic in candidates:
            oi_data = deriv.get("open_interest", {}).get(tic, {})
            oi_usd = oi_data.get("value", 0) if isinstance(oi_data, dict) else 0
            funding = deriv.get("funding_rates", {}).get(tic, 0)
            current_px = ticker_prices.get(tic, 0)
            if current_px > 0:
                liq_contexts[tic] = self.liquidation_tracker.get_context(
                    tic, current_px, oi_usd, funding,
                )

        # OOD detection: score each candidate's TA features
        ood_scores: Dict[str, float] = {}
        for tic in candidates:
            ta_signals = self._extract_ta_signals(tic, MarketSnapshot(
                ticker_prices=ticker_prices,
                pre_computed_ta=pre_computed_ta,
                funding_rates=deriv.get("funding_rates", {}),
                open_interest=deriv.get("open_interest", {}),
                long_short_ratio=deriv.get("long_short_ratio", {}),
                basis_spreads={
                    t: info["basis_annualized"]
                    for t, info in deriv.get("basis_spread", {}).items()
                    if "basis_annualized" in info
                },
                candidates=candidates,
            ))
            if ta_signals:
                self.ood_detector.update(ta_signals)
                is_ood, dist = self.ood_detector.is_ood(ta_signals)
                if dist > 0:
                    ood_scores[tic] = round(dist, 2)
                if is_ood:
                    logger.info("[ood] %s: Mahalanobis=%.2f (OOD)", tic, dist)

        # Safety config for dynamic prompt generation (no hardcoded values in prompt)
        from config import CRYPTO_RISK_CONFIG
        safety_config = {
            # Fallback defaults match current catastrophic policy
            "stop_loss_pct": CRYPTO_RISK_CONFIG.get("stop_loss_pct", -0.05),
            "take_profit_pct": CRYPTO_RISK_CONFIG.get("take_profit_pct", 0.05),
            "max_hold_hours": CRYPTO_RISK_CONFIG.get("max_hold_hours", 4.0),
            "max_position_pct": CRYPTO_RISK_CONFIG.get("max_position_pct", 0.15),
            "max_exposure_pct": CRYPTO_RISK_CONFIG.get("max_exposure_pct", 0.60),
            "dca_stop_loss_pct": CRYPTO_RISK_CONFIG.get("dca_stop_loss_pct", -0.05),
            "dca_max_hold_hours": CRYPTO_RISK_CONFIG.get("dca_max_hold_hours", 4.0),
            "scalp_trail_activation_pct": CRYPTO_RISK_CONFIG.get("scalp_trail_activation_pct", 0.02),
            "scalp_trail_width_pct": CRYPTO_RISK_CONFIG.get("scalp_trail_width_pct", 0.01),
            "trail_activation_pct": cfg.get("trail_activation_pct", 0.04),
            "trail_pct": cfg.get("trail_pct", 0.06),
            "round_trip_fee": ROUND_TRIP_FEE,
        }

        return MarketSnapshot(
            ticker_prices=ticker_prices,
            ticker_returns_4h=ticker_returns_4h,
            ticker_returns_24h=ticker_returns_24h,
            ticker_volumes=ticker_volumes,
            btc_price=btc_price,
            btc_change_1h=btc_1h,
            btc_change_24h=btc_24h,
            safety_config=safety_config,
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
            # Macro indicators (FRED + DXY) — cache get_context() (was 7 separate calls)
            dxy=macro_ctx.get("dxy", 0.0) or 0.0,
            dxy_direction=macro_ctx.get("dxy_direction", "unknown"),
            dxy_1m_pct=macro_ctx.get("dxy_1m_pct", 0.0) or 0.0,
            net_liquidity_direction=macro_ctx.get("net_liquidity_direction", "unknown"),
            net_liquidity_delta_pct=macro_ctx.get("net_liquidity_delta_pct", 0.0) or 0.0,
            financial_stress=macro_ctx.get("financial_stress", "unknown"),
            nfci=macro_ctx.get("nfci", 0.0) or 0.0,
            ts_mean_weights=ts_mean_weights,
            ts_group_weights=ts_group_weights,
            ts_meta_params=ts_meta_params,
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
            btc_dominance_pct=btc_dominance if isinstance(btc_dominance, (int, float)) else 0.0,
            social_buzz_score=social_buzz_score if isinstance(social_buzz_score, (int, float)) else 0.0,
            news_sentiment_score=news_sentiment if isinstance(news_sentiment, (int, float)) else 0.0,
            whale_activity_score=whale_activity if isinstance(whale_activity, (int, float)) else 0.0,
            exchange_netflow_score=exchange_netflow if isinstance(exchange_netflow, (int, float)) else 0.0,
            multi_tf_summary=multi_tf_summary,
            pre_computed_ta=pre_computed_ta,
            historical_insights=historical_insights,
            diary_context=diary_context,
            liquidation_contexts=liq_contexts,
            ood_scores=ood_scores,
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

            # WS warmup check — block decisions until we have live prices
            cfg = ACTIVE_BLEND_CONFIG
            missing_tickers = [t for t in cfg["tickers"] if t not in self._last_tick_prices]
            if missing_tickers:
                logger.warning(
                    "[decide] WS not warmed up — missing prices for %s, skipping decision",
                    missing_tickers,
                )
                return

            # Counterfactual learning: check old HOLD snapshots for missed opportunities
            self._check_counterfactuals()
            # Post-exit counterfactual: check if we exited too early or at right time
            self._check_exit_regrets()

            logger.info("[decide] Collecting data (trigger=%s)...", trigger)
            close_df = await asyncio.to_thread(self._fetch_ohlcv_df)
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
                            "side": tracked.side,
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
                # Only update memory if decisions led to actual trades or pure HOLD.
                # When Claude hallucinates SELL/COVER that gets filtered, don't save
                # the stale memory — it perpetuates the hallucination loop.
                mem = agent_result.get("memory_update", "")
                if mem:
                    raw_decisions = agent_result.get("decisions", [])
                    had_filtered_sells = any(
                        d.get("action", "").upper() in ("SELL", "COVER")
                        for d in raw_decisions
                    ) and not decisions  # all got filtered
                    if had_filtered_sells and not self._entry_prices:
                        # Force clear memory to break SELL hallucination loop
                        logger.info("[decide] Filtered SELL/COVER with no positions — resetting memory")
                        self._agent_memory = "[NO POSITIONS OPEN] Cash only. 신규 BUY/SHORT만 가능."
                    else:
                        open_tickers = list(self._entry_prices.keys())
                        if not open_tickers:
                            mem = "[NO POSITIONS OPEN] " + mem
                        self._agent_memory = mem[:500]

                # Record learning insight to Neo4j
                learning_note = agent_result.get("learning_note", "")
                if learning_note:
                    try:
                        self.memory.record_insight(snapshot.combined_regime, learning_note)
                    except Exception:
                        pass
            else:
                # Claude unavailable — do NOTHING (no fallback trading)
                logger.warning("[decide] Claude unavailable — skipping (no fallback trades)")
                decisions = []
                # Retry sooner (5 min) instead of waiting 1h
                self.adaptive_gate.update_from_claude(next_check_seconds=300)

            # Log decision context (always, even if no trades)
            self._log_decision(trigger, snapshot, agent_result, decisions)

            # Layer 2: Daily digest check (non-blocking, runs after decision)
            try:
                await self.diary.maybe_generate_digest()
            except Exception as exc:
                logger.debug("[diary] Digest check failed: %s", exc)

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
                    is_add = d.get("is_add", False)  # ADD→BUY conversion flag
                    if d["action"] == "BUY" and d.get("price", 0) > 0:
                        if ticker not in executed_tickers:
                            logger.warning("[decide] BUY %s failed execution, skipping track", ticker)
                            continue

                        existing_pos = self.position_tracker.get_position(ticker)
                        if is_add and existing_pos is not None:
                            # ADD: increase existing position, update average entry price
                            # Converts position to DCA mode → wider SL (-1.5%) and longer hold (2h)
                            add_qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                            if add_qty > 0:
                                old_entry = self._entry_prices.get(ticker, d["price"])
                                old_qty = existing_pos.qty
                                new_qty = old_qty + add_qty
                                # Weighted average entry price
                                new_avg_entry = (old_entry * old_qty + d["price"] * add_qty) / new_qty
                                self._entry_prices[ticker] = new_avg_entry
                                existing_pos.qty = new_qty
                                existing_pos.entry_price = new_avg_entry  # tracker uses this for PnL
                                existing_pos.trailing_high = max(new_avg_entry, existing_pos.current_price)
                                # Mark as DCA → safety layer uses wider stops
                                existing_pos.is_dca = True
                                logger.info("[decide] ADD %s: +%.4f qty @ $%.2f → avg entry $%.2f, total %.4f (DCA mode: SL=-1.5%%, hold=2h)",
                                            ticker, add_qty, d["price"], new_avg_entry, new_qty)
                            await self.event_bus.publish("decision.signal", {
                                "ticker": ticker,
                                "signals": d.get("agent_signals", {}),
                            })
                            self._log_trade(
                                "ADD", ticker, d["price"], add_qty, d.get("position_size_usd", 0),
                                reason=d.get("reasons", [""])[0] if d.get("reasons") else "",
                                regime=snapshot.combined_regime, confidence=d.get("confidence", 0),
                                dry_run=dry_run,
                            )
                            self._write_alert("ADD", f"{ticker} @ ${d['price']:,.2f} (avg→${new_avg_entry:,.2f})")
                            continue

                        # Normal BUY: skip if already holding this ticker
                        if existing_pos is not None:
                            logger.info("[decide] BUY %s skipped: already holding position", ticker)
                            continue
                        # Anti-churn: block re-entry within 10 minutes of selling same ticker
                        cooldown_until = self._exit_cooldowns.get(ticker, 0)
                        if time.time() < cooldown_until:
                            remaining = int(cooldown_until - time.time())
                            logger.info("[decide] BUY %s blocked: anti-churn cooldown (%ds left)", ticker, remaining)
                            continue
                        self._entry_prices[ticker] = d["price"]
                        # Save entry metadata for Level 0 meta-parameter learning
                        pv_at_entry = snapshot.portfolio_value if snapshot.portfolio_value > 0 else 1
                        self._entry_meta[ticker] = {
                            "position_pct": d.get("position_size_usd", 0) / pv_at_entry,
                            "confidence": d.get("confidence", 0),
                            "ta_snapshot": self._extract_ta_signals(ticker, snapshot),
                        }
                        qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                        if qty > 0:
                            self.position_tracker.track(
                                ticker=ticker,
                                broker="binance",
                                entry_price=d["price"],
                                qty=qty,
                                market_type="crypto",
                                regime=snapshot.combined_regime,
                                side="long",
                            )
                        await self.event_bus.publish("decision.signal", {
                            "ticker": ticker,
                            "signals": d.get("agent_signals", {}),
                        })
                        # === Log BUY ===
                        decision_src = d.get("decision_source", "claude_agent")
                        self._log_trade(
                            "BUY", ticker, d["price"], qty, d.get("position_size_usd", 0),
                            reason=d.get("reasons", [""])[0] if d.get("reasons") else "",
                            regime=snapshot.combined_regime, confidence=d.get("confidence", 0),
                            source=decision_src,
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

                        # === PRICE SANITY GUARD ===
                        # Cross-validate snapshot price against tracker's live price
                        # Prevents phantom PnL from corrupted/stale WS data
                        entry_px = self._entry_prices.get(ticker, d["price"])
                        tracker_px = tracked.current_price
                        if tracker_px > 0 and entry_px > 0:
                            # Side-aware PnL comparison
                            if tracked.side == "short":
                                snapshot_pnl = (entry_px - d["price"]) / entry_px
                                tracker_pnl = (entry_px - tracker_px) / entry_px
                            else:
                                snapshot_pnl = (d["price"] - entry_px) / entry_px
                                tracker_pnl = (tracker_px - entry_px) / entry_px
                            pnl_gap = abs(snapshot_pnl - tracker_pnl)
                            if pnl_gap > 0.05:  # >5% PnL discrepancy
                                logger.error(
                                    "[decide] PRICE SANITY REJECT %s: snapshot_px=%.2f (pnl=%.2f%%) vs "
                                    "tracker_px=%.2f (pnl=%.2f%%) — gap=%.2f%%, using tracker price",
                                    ticker, d["price"], snapshot_pnl * 100,
                                    tracker_px, tracker_pnl * 100, pnl_gap * 100,
                                )
                                d["price"] = tracker_px  # override with trusted tracker price

                        # PnL: side-aware (short positions exit via COVER, not SELL, but guard anyway)
                        if tracked and tracked.side == "short":
                            gross_pnl = (entry_px - d["price"]) / entry_px if entry_px > 0 else 0
                        else:
                            gross_pnl = (d["price"] - entry_px) / entry_px if entry_px > 0 else 0
                        pnl_pct = gross_pnl - ROUND_TRIP_FEE  # net after fees
                        held_hours = tracked.held_hours
                        sell_reason = d.get("reasons", ["claude_sell"])[0] if d.get("reasons") else "claude_sell"

                        sell_qty = tracked.qty if tracked else 0
                        # Anti-churn: set 10-minute cooldown for this ticker
                        self._exit_cooldowns[ticker] = time.time() + 30  # 30s cooldown — agent decides re-entry

                        decision_src = d.get("decision_source", "claude_agent")
                        _sell_side = tracked.side if tracked else "long"
                        await self.event_bus.publish("position.exit", {
                            "ticker": ticker,
                            "source": decision_src,
                            "reason": sell_reason,
                            "entry_price": entry_px,
                            "exit_price": d["price"],
                            "pnl_pct": pnl_pct,
                            "held_hours": held_hours,
                            "qty": sell_qty,
                            "broker": "binance",
                            "agent_signals": d.get("agent_signals", {}),
                            "position_side": _sell_side,
                            "mfe": tracked.mfe if tracked else 0,
                            "mae": tracked.mae if tracked else 0,
                            "capture_ratio": tracked.capture_ratio if tracked else 0,
                        })
                        # Untrack AFTER event publish so _on_exit can read position
                        self.position_tracker.untrack(ticker)
                        self._entry_prices.pop(ticker, None)
                        # === Log SELL ===
                        sell_value = d["price"] * sell_qty
                        self._log_trade(
                            "SELL", ticker, d["price"], sell_qty, sell_value,
                            pnl_pct=pnl_pct, held_hours=held_hours,
                            reason=sell_reason, regime=snapshot.combined_regime,
                            source=decision_src, confidence=d.get("confidence", 0),
                            dry_run=dry_run,
                        )
                        self._write_alert("SELL", f"{ticker} @ ${d['price']:,.2f} PnL={pnl_pct:+.2%}")

                    elif d["action"] == "SHORT" and d.get("price", 0) > 0:
                        # SHORT confidence gate: require >= 0.75 confidence (31.9% WR at lower thresholds)
                        short_conf = d.get("confidence", 0)
                        if short_conf < 0.75:
                            logger.info("[decide] SHORT %s blocked: confidence %.2f < 0.75 threshold", ticker, short_conf)
                            continue
                        if ticker not in executed_tickers:
                            logger.warning("[decide] SHORT %s failed execution, skipping track", ticker)
                            continue

                        existing_pos = self.position_tracker.get_position(ticker)
                        if existing_pos is not None:
                            logger.info("[decide] SHORT %s skipped: already holding position", ticker)
                            continue
                        # Anti-churn: block re-entry within cooldown
                        cooldown_until = self._exit_cooldowns.get(ticker, 0)
                        if time.time() < cooldown_until:
                            remaining = int(cooldown_until - time.time())
                            logger.info("[decide] SHORT %s blocked: anti-churn cooldown (%ds left)", ticker, remaining)
                            continue
                        self._entry_prices[ticker] = d["price"]
                        pv_at_entry = snapshot.portfolio_value if snapshot.portfolio_value > 0 else 1
                        self._entry_meta[ticker] = {
                            "position_pct": d.get("position_size_usd", 0) / pv_at_entry,
                            "confidence": d.get("confidence", 0),
                            "ta_snapshot": self._extract_ta_signals(ticker, snapshot),
                        }
                        qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                        if qty > 0:
                            self.position_tracker.track(
                                ticker=ticker,
                                broker="binance",
                                entry_price=d["price"],
                                qty=qty,
                                market_type="crypto",
                                regime=snapshot.combined_regime,
                                side="short",
                            )
                        await self.event_bus.publish("decision.signal", {
                            "ticker": ticker,
                            "signals": d.get("agent_signals", {}),
                        })
                        decision_src = d.get("decision_source", "claude_agent")
                        self._log_trade(
                            "SHORT", ticker, d["price"], qty, d.get("position_size_usd", 0),
                            reason=d.get("reasons", [""])[0] if d.get("reasons") else "",
                            regime=snapshot.combined_regime, confidence=d.get("confidence", 0),
                            source=decision_src,
                            dry_run=dry_run,
                        )
                        self._write_alert("SHORT", f"{ticker} @ ${d['price']:,.2f} ({d.get('confidence',0):.0%})")

                    elif d["action"] == "COVER" and d.get("price", 0) > 0:
                        if ticker not in executed_tickers:
                            # Check if position still exists on exchange
                            try:
                                fresh_qty = self.broker._get_position_qty(
                                    self.broker._normalize_symbol(ticker))
                                if fresh_qty <= 0:
                                    logger.warning("[decide] COVER %s failed — no position on exchange, cleaning up tracker", ticker)
                                    self.position_tracker.untrack(ticker)
                                    self._entry_prices.pop(ticker, None)
                                    self._save_state()
                                else:
                                    logger.warning("[decide] COVER %s failed execution (exchange has %.8f), skipping", ticker, fresh_qty)
                            except Exception:
                                logger.warning("[decide] COVER %s failed execution, skipping", ticker)
                            continue
                        tracked = self.position_tracker.get_position(ticker)
                        if tracked is None:
                            logger.info("[decide] %s already exited by safety layer, skipping COVER", ticker)
                            self._entry_prices.pop(ticker, None)
                            continue

                        entry_px = self._entry_prices.get(ticker, d["price"])
                        # SHORT PnL: (entry - exit) / entry
                        if tracked.side == "short":
                            gross_pnl = (entry_px - d["price"]) / entry_px if entry_px > 0 else 0
                        else:
                            gross_pnl = (d["price"] - entry_px) / entry_px if entry_px > 0 else 0
                        pnl_pct = gross_pnl - ROUND_TRIP_FEE
                        held_hours = tracked.held_hours
                        cover_reason = d.get("reasons", ["claude_cover"])[0] if d.get("reasons") else "claude_cover"

                        cover_qty = tracked.qty if tracked else 0
                        self._exit_cooldowns[ticker] = time.time() + 30

                        decision_src = d.get("decision_source", "claude_agent")
                        _cover_side = tracked.side if tracked else "short"
                        await self.event_bus.publish("position.exit", {
                            "ticker": ticker,
                            "source": decision_src,
                            "reason": cover_reason,
                            "entry_price": entry_px,
                            "exit_price": d["price"],
                            "pnl_pct": pnl_pct,
                            "held_hours": held_hours,
                            "qty": cover_qty,
                            "broker": "binance",
                            "agent_signals": d.get("agent_signals", {}),
                            "position_side": _cover_side,
                            "mfe": tracked.mfe if tracked else 0,
                            "mae": tracked.mae if tracked else 0,
                            "capture_ratio": tracked.capture_ratio if tracked else 0,
                        })
                        self.position_tracker.untrack(ticker)
                        self._entry_prices.pop(ticker, None)
                        cover_value = d["price"] * cover_qty
                        self._log_trade(
                            "COVER", ticker, d["price"], cover_qty, cover_value,
                            pnl_pct=pnl_pct, held_hours=held_hours,
                            reason=cover_reason, regime=snapshot.combined_regime,
                            source=decision_src, confidence=d.get("confidence", 0),
                            dry_run=dry_run,
                        )
                        self._write_alert("COVER", f"{ticker} @ ${d['price']:,.2f} PnL={pnl_pct:+.2%}")

                # Persist state after any trade
                if executed_tickers:
                    self._save_state()
            else:
                logger.info("[decide] No trades (trigger=%s)", trigger)
                # Counterfactual: save snapshot for missed-opportunity learning
                self._save_hold_snapshot(snapshot)

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
            # ADD is treated as BUY for execution (additional allocation to existing position)
            is_add = False
            if action == "ADD":
                if self.position_tracker.get_position(ticker) is None:
                    logger.info("[decide] ADD %s skipped: no existing position to add to", ticker)
                    continue
                action = "BUY"  # broker executes as a BUY order
                is_add = True
            if action == "SELL":
                # SELL: close existing LONG position
                tracked = self.position_tracker.get_position(ticker)
                if tracked is None or tracked.side != "long":
                    logger.info("[decide] SELL %s skipped: no long position", ticker)
                    continue
            if action == "COVER":
                # COVER: close existing SHORT position
                tracked = self.position_tracker.get_position(ticker)
                if tracked is None or tracked.side != "short":
                    logger.info("[decide] COVER %s skipped: no short position", ticker)
                    continue
            if ticker not in snapshot.candidates and action in ("BUY", "SHORT"):
                continue

            price = snapshot.ticker_prices.get(ticker, 0)
            if price <= 0:
                continue

            from config import CRYPTO_RISK_CONFIG
            try:
                position_pct = float(d.get("position_pct", 0.10))
                # Agent decides sizing — only clamp to max_position_pct (catastrophic limit)
                max_pos = CRYPTO_RISK_CONFIG.get("max_position_pct", 0.15)
                position_pct = max(0.0, min(position_pct, max_pos))
            except (TypeError, ValueError):
                position_pct = 0.10

            # Enforce max portfolio exposure (CRYPTO_RISK_CONFIG["max_exposure_pct"])
            if action in ("BUY", "SHORT"):
                max_exposure = CRYPTO_RISK_CONFIG.get("max_exposure_pct", 0.60)
                current_exposure = 0.0
                if snapshot.portfolio_value > 0:
                    for tic_held, ep in self._entry_prices.items():
                        tracked_pos = self.position_tracker.get_position(tic_held)
                        if tracked_pos and tracked_pos.qty > 0:
                            current_px = snapshot.ticker_prices.get(tic_held, ep)
                            current_exposure += (tracked_pos.qty * current_px) / snapshot.portfolio_value
                    remaining = max(0.0, max_exposure - current_exposure)
                    if position_pct > remaining:
                        logger.info("[decide] Exposure cap: %.1f%% used, %.1f%% remaining, clamped %.1f%%→%.1f%%",
                                    current_exposure * 100, remaining * 100, position_pct * 100, remaining * 100)
                        position_pct = remaining
                    if position_pct <= 0.001:
                        continue  # skip: would exceed exposure limit

            position_size_usd = snapshot.portfolio_value * position_pct
            # Binance Futures minimum notional = $100
            if action in ("BUY", "SHORT") and position_size_usd < 100.0 and snapshot.portfolio_value >= 100.0:
                logger.info("[decide] position_size_usd=$%.2f < $100 minimum, bumping to $100", position_size_usd)
                position_size_usd = 100.0

            # Agent decides confidence — no hardcoded filter.
            # Claude's confidence is logged for H-TS learning, not gatekept.
            try:
                confidence = float(d.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5

            # Build agent_signals: prefer Claude's explicit signal_weights, fallback to keyword extraction
            reasoning = d.get("reasoning", "")
            claude_sw = d.get("signal_weights")
            if claude_sw and isinstance(claude_sw, dict):
                agent_signals = {}
                for k, v in claude_sw.items():
                    try:
                        fv = float(v)
                        if abs(fv) >= 0.05:
                            agent_signals[k] = fv
                    except (TypeError, ValueError):
                        pass
            else:
                agent_signals = self._extract_signals_from_reasoning(reasoning, confidence)

            dec = {
                "ticker": ticker,
                "action": action,
                "is_add": is_add,
                "confidence": confidence,
                "position_size_usd": position_size_usd,
                "price": price,
                "reasons": [reasoning],
                "agent_signals": agent_signals,
            }
            # For SELL/COVER: include qty from tracker so dry_run doesn't need REST
            if action in ("SELL", "COVER"):
                tracked = self.position_tracker.get_position(ticker)
                if tracked and tracked.qty > 0:
                    dec["qty"] = tracked.qty
            decisions.append(dec)

        return decisions

    @staticmethod
    def _extract_signals_from_reasoning(reasoning: str, confidence: float) -> Dict[str, float]:
        """Extract signal categories from Claude's reasoning for H-TS feedback.

        Maps Claude's reasoning keywords to the 28 hierarchical signal names.
        This is a FALLBACK — Claude should return explicit signal_weights in JSON.
        """
        reasoning_lower = reasoning.lower()
        signals = {}

        # Map keywords to new hierarchical signal names
        keywords_to_signals = {
            # technical_trend
            "ema": "ema_cross_fast",
            "moving average": "ema_cross_fast",
            "macd": "macd_histogram",
            "trend": "trend_strength",
            "adx": "trend_strength",
            "supertrend": "supertrend",
            # technical_reversion
            "rsi": "rsi_signal",
            "overbought": "rsi_signal",
            "oversold": "rsi_signal",
            "stochastic": "stoch_rsi",
            "bollinger": "bb_deviation",
            "squeeze": "bb_squeeze",
            "vwap": "vwap_deviation",
            "support": "support_resistance",
            "resistance": "support_resistance",
            # technical_volume
            "volume": "volume_spike",
            "cvd": "cvd_signal",
            "obv": "obv_divergence",
            "money flow": "mfi_signal",
            # derivatives
            "funding": "funding_rate",
            "open interest": "oi_change",
            "long/short": "long_short_ratio",
            "liquidat": "liquidation_level",
            "basis": "basis_spread",
            # sentiment
            "sentiment": "news_sentiment",
            "fear": "fear_greed",
            "greed": "fear_greed",
            "whale": "whale_activity",
            "exchange flow": "exchange_flow",
            # macro
            "regime": "market_regime",
            "volatility regime": "volatility_regime",
            "dominance": "btc_dominance",
            "dxy": "dxy_direction",
            "dollar": "dxy_direction",
            "etf": "etf_flow",
            "stablecoin": "stablecoin_flow",
            "macro": "market_regime",
            "momentum": "trend_strength",
            "price action": "ema_cross_fast",
        }

        for keyword, signal_name in keywords_to_signals.items():
            if keyword in reasoning_lower:
                signals[signal_name] = confidence

        # Default: attribute to trend_strength if no keywords matched
        if not signals:
            signals["trend_strength"] = confidence

        return signals

    async def _run_fallback_pipeline(
        self,
        close_df: pd.DataFrame,
        positions: Dict,
        pv: float,
        snapshot: MarketSnapshot,
        trigger: str,
    ) -> list:
        """Rule-based fallback when Claude is unavailable.

        SELL-only mode: new BUY entries are blocked when Claude is down.
        Rationale: blind rule-based BUY in a falling market caused -5.75%
        overnight loss (2026-02-28). Only exit existing positions.
        """
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
        buys_blocked = 0
        for trade in ctx.get("trade_log", []):
            ticker = trade["ticker"]
            # SELL-only: block BUY when Claude is down (blind entries lose money)
            if trade["side"] == "BUY":
                buys_blocked += 1
                continue
            # Use real-time WS price instead of candle close
            ws_price = self._last_tick_prices.get(ticker, 0)
            price = ws_price if ws_price > 0 else trade["price"]
            dec = {
                "ticker": ticker,
                "action": trade["side"],
                "confidence": trade.get("score", 0.5) if trade["side"] == "BUY" else -1.0,
                "position_size_usd": trade["qty"] * price,
                "price": price,
                "reasons": [trade.get("reason", "rule_based_fallback")],
                "agent_signals": {},
                "decision_source": "rule_based_fallback",
            }
            # For SELL: include qty so dry_run doesn't need REST
            if trade["side"] == "SELL":
                dec["qty"] = trade["qty"]
            decisions.append(dec)
        if buys_blocked:
            logger.info("[fallback] Blocked %d BUY(s) — SELL-only mode when Claude is unavailable", buys_blocked)
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

        # Inject runner state into Claude's agent tools (MCP)
        try:
            from core.agent_tools import set_runner
            set_runner(self)
        except Exception as exc:
            logger.warning("[runner] Agent tools injection failed: %s", exc)

        logger.info("[runner] Initial balance: $%.2f", self._initial_balance)

        # Sync actual broker balance on startup
        try:
            bal = self.broker._exchange.fetch_balance()
            total = bal.get("total", {})
            usdt = float(total.get("USDT", 0))
            usdc = float(total.get("USDC", 0))
            broker_bal = usdt + usdc
            if broker_bal > 0:
                self._broker_balance = broker_bal
                self._broker_balance_time = time.time()
                logger.info("[runner] Broker balance synced: $%.2f (USDT=%.2f, USDC=%.2f)", broker_bal, usdt, usdc)
                # Update initial_balance if broker has significantly more
                if broker_bal > self._initial_balance * 2:
                    logger.info("[runner] Updating initial_balance: $%.2f → $%.2f (broker has more)", self._initial_balance, broker_bal)
                    self._initial_balance = broker_bal
        except Exception as exc:
            logger.warning("[runner] Broker balance sync failed: %s", exc)

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
