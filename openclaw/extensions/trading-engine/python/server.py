"""FastAPI server for the OpenClaw Trading Engine.

Exposes endpoints for trading decisions, backtesting, real-time monitoring,
and Claude agent status. Training runs in a background thread so the
server remains responsive.

Start with::

    uvicorn server:app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from config import MODELS_DIR, SERVER_CONFIG, TRAIN_CONFIG, SECTOR_SCAN_CONFIG, REGIME_BLEND_CONFIG
from analysis.data_processor import DataProcessor
from analysis.ensemble_agent import EnsembleAgent
from analysis.auto_trader import AutoTrader
from brokers.alpaca import AlpacaBroker
from brokers.binance import BinanceBroker
from brokers.kis import KISBroker
from brokers.factory import BrokerRegistry
from analysis.regime_detector import RegimeDetector
from core.risk_manager import RiskManager
from backtests.scan_backtester import run_backtest as run_scan_backtest
from analysis.sector_scanner import SectorScanner
from analysis.sentiment_finbert import FinBERTScorer
from analysis.sentiment_scorer import SentimentScorer
from analysis.stock_ranker import StockRanker
from core.event_bus import EventBus
from core.ws_stream import TradingStream
from agents.market_agent import MarketAgent
from agents.quant_agent import QuantAgent
from agents.quant_agent_crypto import CryptoQuantAgent
from agents.synthesizer import Synthesizer, decisions_to_legacy
from analysis.regime_detector_crypto import CryptoRegimeDetector
from core.position_tracker import PositionTracker
from core.agent_evaluator import evaluate_position, build_market_context
from core.online_learner import OnlineLearner
from core.claude_agent import ClaudeAgent
from core.claude_auth import is_claude_available
from core.pipeline import (
    Pipeline, RegimeBlendDetectNode, RegimeBlendSignalNode,
    RegimeBlendExitNode, RegimeBlendEntryNode,
)

# -----------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------

_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "server.log"),
    ],
)
logger = logging.getLogger("trading-engine")

# -----------------------------------------------------------------------
# Application state (module-level singletons)
# -----------------------------------------------------------------------

data_processor = DataProcessor()
ensemble_agent = EnsembleAgent()
risk_manager = RiskManager()
sentiment_scorer = SentimentScorer()
finbert_scorer = FinBERTScorer()  # lazy-loads model on first use
regime_detector = RegimeDetector()
stock_ranker = StockRanker()
sector_scanner = SectorScanner(
    sentiment_scorer=sentiment_scorer,
    finbert_scorer=finbert_scorer,
    regime_detector=regime_detector,
    stock_ranker=stock_ranker,
)
alpaca_broker = AlpacaBroker()
binance_broker = BinanceBroker()
kis_broker = KISBroker()

broker_registry = BrokerRegistry()
broker_registry.register("alpaca", alpaca_broker)
broker_registry.register("binance", binance_broker)
broker_registry.register("kis", kis_broker)

auto_trader = AutoTrader(
    regime_detector=regime_detector,
    sector_scanner=sector_scanner,
    risk_manager=risk_manager,
    stock_ranker=stock_ranker,
)

# -----------------------------------------------------------------------
# v3: Multi-agent + EventBus + WebSocket
# -----------------------------------------------------------------------

event_bus = EventBus()
trading_stream = TradingStream(event_bus)

market_agent = MarketAgent(
    regime_detector=regime_detector,
    sector_scanner=sector_scanner,
    event_bus=event_bus,
)
quant_agent = QuantAgent(event_bus=event_bus)
crypto_regime_detector = CryptoRegimeDetector()
crypto_quant_agent = CryptoQuantAgent(event_bus=event_bus)
synthesizer = Synthesizer(
    event_bus=event_bus,
    risk_manager=risk_manager,
    finbert_scorer=finbert_scorer,
    sector_scanner=sector_scanner,
    ensemble_agent=ensemble_agent,
    data_processor=data_processor,
)

# Thread pool for CPU-heavy training
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")

# Cron lock to prevent 15m decide and 5m risk check from colliding
_cron_lock = asyncio.Lock()

# Local position entry tracking for max_hold_days
_position_entry_dates: Dict[str, datetime] = {}

# Crypto entry price tracking (ccxt Spot doesn't track cost basis)
_crypto_entry_prices: Dict[str, float] = {}

# Crypto trailing high tracking for Regime Blend trail stop
_crypto_trailing_highs: Dict[str, float] = {}

# Position tracker: continuous agent-based monitoring (not just cron rules)
position_tracker = PositionTracker(
    event_bus=event_bus,
    safety_interval=30.0,   # price check every 30s
    agent_interval=180.0,   # agent re-evaluation every 3min
)

# Online learner: Thompson Sampling weight adaptation (real-time RL)
online_learner = OnlineLearner(
    save_path=str(MODELS_DIR / "online_learner.json"),
    min_trades_to_adapt=5,
)
online_learner.load()  # restore from previous session


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class _AppState:
    """Mutable container for server-wide state."""

    training_status: TrainingStatus = TrainingStatus.IDLE
    training_started_at: Optional[str] = None
    training_completed_at: Optional[str] = None
    training_error: Optional[str] = None
    last_metrics: Dict[str, Any] = {}
    last_train_config: Dict[str, Any] = {}


app_state = _AppState()

# -----------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    training_status: str
    timestamp: str


class TrainRequest(BaseModel):
    tickers: List[str] = Field(default_factory=lambda: list(TRAIN_CONFIG["tickers"]))
    lookback_days: int = Field(default=TRAIN_CONFIG["lookback_days"], ge=30, le=3650)
    total_timesteps: int = Field(default=TRAIN_CONFIG["total_timesteps"], ge=1000, le=5_000_000)
    learning_rate: float = Field(default=TRAIN_CONFIG["learning_rate"], gt=0, lt=1)
    train_split: float = Field(default=TRAIN_CONFIG["train_split"], gt=0.1, lt=1.0)
    continue_from_checkpoint: bool = Field(default=True, description="Continue training from saved checkpoint (FreqAI-style continuous learning)")


class TrainResponse(BaseModel):
    message: str
    training_status: str
    started_at: str


class PredictRequest(BaseModel):
    tickers: Optional[List[str]] = None
    observation: Optional[List[float]] = None


class PredictAction(BaseModel):
    ticker: str
    action: float = Field(description="Action in [-1, 1]. Negative=sell, positive=buy")
    risk_check: Dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    actions: List[PredictAction]
    ensemble_weights: Dict[str, float]
    timestamp: str


class StatusResponse(BaseModel):
    model_loaded: bool
    training_status: str
    training_started_at: Optional[str]
    training_completed_at: Optional[str]
    training_error: Optional[str]
    last_train_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    risk_manager_status: Dict[str, Any]
    ensemble_weights: Dict[str, float]
    timestamp: str


class BacktestRequest(BaseModel):
    tickers: List[str] = Field(default_factory=lambda: list(TRAIN_CONFIG["tickers"]))
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = Field(default=90, ge=7, le=3650)


class BacktestResponse(BaseModel):
    metrics: Dict[str, Any]
    tickers: List[str]
    start_date: str
    end_date: str
    num_trading_days: int
    timestamp: str


class ScanRequest(BaseModel):
    top_sectors: int = Field(default=SECTOR_SCAN_CONFIG["top_sectors"], ge=1, le=14)
    stocks_per_sector: int = Field(default=SECTOR_SCAN_CONFIG["stocks_per_sector"], ge=1, le=10)
    include_sentiment: bool = Field(default=True)


class ScanResponse(BaseModel):
    scan_time: str
    benchmark: str
    regime: Optional[Dict[str, Any]] = None
    sector_rankings: List[Dict[str, Any]]
    top_sectors: List[Dict[str, Any]]
    recommended_tickers: List[Dict[str, Any]]


class AutoDecideRequest(BaseModel):
    portfolio_value: float = Field(default=100_000.0, gt=0)
    top_sectors: int = Field(default=3, ge=1, le=14)
    stocks_per_sector: int = Field(default=5, ge=1, le=10)
    include_sentiment: bool = Field(default=True)
    current_positions: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Current held positions {ticker: {qty, entry_price, current_price, ...}}",
    )


class ExecuteRequest(BaseModel):
    decisions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Decisions from /decide endpoint")
    auto_decide: bool = Field(default=True, description="Auto-run /decide first if decisions not provided")
    dry_run: bool = Field(default=True, description="Dry run (no real orders) by default for safety")
    portfolio_value: float = Field(default=100_000.0, gt=0)


class ScanBacktestRequest(BaseModel):
    months: int = Field(default=12, ge=3, le=36, description="Number of months to backtest")


class ScanBacktestResponse(BaseModel):
    backtest_months: int
    valid_periods: int
    hit_rate: float
    avg_monthly_return: float
    avg_benchmark_return: float
    avg_excess_return: float
    information_ratio: float
    cumulative_portfolio: float
    cumulative_benchmark: float
    cumulative_excess: float
    verdict: str
    monthly_results: List[Dict[str, Any]]


# -----------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycle."""
    # Startup
    logger.info("Starting OpenClaw Trading Engine server v3 ...")
    loaded = ensemble_agent.load_models()
    if loaded:
        logger.info("Pre-trained models loaded successfully.")
        app_state.training_status = TrainingStatus.COMPLETED
    else:
        logger.info("No pre-trained models found. Train via POST /train.")

    # Auto-connect all brokers at startup
    for bname in broker_registry.names:
        broker = broker_registry.get(bname)
        try:
            conn = broker.connect()
            if conn.get("connected"):
                logger.info("Broker %s connected: %s", bname, {k: v for k, v in conn.items() if k != "connected"})
            else:
                logger.info("Broker %s not configured or failed: %s", bname, conn.get("error", "no credentials"))
        except Exception as e:
            logger.info("Broker %s skipped: %s", bname, e)

    # Start cron scheduler (default enabled; set ENABLE_CRON=false to disable)
    scheduler = None
    if os.environ.get("ENABLE_CRON", "true").lower() in ("1", "true", "yes"):
        scheduler = _start_scheduler()

    # Start position tracker (agent-based continuous monitoring)
    tracker_task = None
    if os.environ.get("ENABLE_TRACKER", "true").lower() in ("1", "true", "yes"):
        _setup_position_tracker()
        tracker_task = asyncio.create_task(position_tracker.run())
        logger.info("Position tracker started (safety@30s, agent@180s)")

    yield

    # Shutdown
    if tracker_task is not None:
        position_tracker.stop()
        tracker_task.cancel()
    if scheduler is not None:
        scheduler.shutdown(wait=False)
    _executor.shutdown(wait=False)
    logger.info("Trading engine server shut down.")


# -----------------------------------------------------------------------
# Position tracker setup
# -----------------------------------------------------------------------

def _setup_position_tracker() -> None:
    """Wire PositionTracker with price fetchers, agent evaluator, and executors."""
    import os

    dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")

    # Price fetcher for Binance
    def binance_price(ticker: str) -> float:
        if not binance_broker.is_connected:
            return 0.0
        try:
            info = binance_broker._exchange.fetch_ticker(ticker)
            return float(info.get("last", 0))
        except Exception:
            return 0.0

    # Price fetcher for Alpaca
    def alpaca_price(ticker: str) -> float:
        if not alpaca_broker.is_connected:
            return 0.0
        try:
            positions = alpaca_broker.get_positions_detail()
            if ticker in positions:
                return positions[ticker].get("current_price", 0.0)
        except Exception:
            pass
        return 0.0

    # Market context provider
    def context_provider() -> Dict[str, Any]:
        return build_market_context(
            regime_detector=regime_detector,
            crypto_regime_detector=crypto_regime_detector,
            binance_broker=binance_broker,
        )

    # Exit executor for Binance
    async def binance_exit(decision: Dict[str, Any]) -> Dict[str, Any]:
        return binance_broker.execute_decisions([decision], dry_run=dry_run)

    # Exit executor for Alpaca
    async def alpaca_exit(decision: Dict[str, Any]) -> Dict[str, Any]:
        return alpaca_broker.execute_decisions([decision], dry_run=dry_run)

    position_tracker.set_price_fetcher("binance", binance_price)
    position_tracker.set_price_fetcher("alpaca", alpaca_price)
    position_tracker.set_agent_evaluator(evaluate_position)
    position_tracker.set_context_provider(context_provider)
    position_tracker.set_exit_executor("binance", binance_exit)
    position_tracker.set_exit_executor("alpaca", alpaca_exit)

    # Subscribe to position exits → feed online learner
    async def _on_position_exit(event):
        """When a position is closed, record outcome for online learning."""
        data = event.payload
        ticker = data.get("ticker", "")
        entry_price = data.get("entry_price", 0)
        exit_price = data.get("exit_price", 0)
        pnl_pct = data.get("pnl_pct", 0)
        held_hours = data.get("held_hours", 0)
        market_type = data.get("market_type", "crypto")

        # Find the agent signals that were active at entry
        # (stored in the tracked position's decision signals)
        pos = position_tracker.get_position(ticker)
        agent_signals = {}
        if pos and hasattr(pos, '_entry_signals'):
            agent_signals = pos._entry_signals
        else:
            # Look for recent decision signals from event bus
            for ev in event_bus.get_recent_events(100):
                if ev.get("topic") == "decision.signal" and ev.get("payload", {}).get("ticker") == ticker:
                    agent_signals = ev["payload"].get("signals", {})
                    break

        if agent_signals:
            online_learner.record_trade(
                ticker=ticker,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                held_hours=held_hours,
                agent_signals=agent_signals,
                market_type=market_type,
            )

    event_bus.subscribe("position.exit", _on_position_exit)

    logger.info("[tracker] Wired: price(binance,alpaca), agent_eval, context, exits, online_learner")


# -----------------------------------------------------------------------
# Cron scheduler (for standalone / Docker deployment)
# -----------------------------------------------------------------------

def _start_scheduler():
    """Start APScheduler cron jobs for autonomous trading."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()

    # Every 15 minutes (market hours): run v3 multi-agent pipeline + execute
    async def cron_decide():
        if _cron_lock.locked():
            logger.debug("[cron] Skipping decide — another cron job running")
            return
        async with _cron_lock:
            logger.info("[cron] Running 15-minute multi-agent decision...")
            try:
                # Get portfolio value and current positions from broker
                pv = 100_000.0
                positions_detail = None
                if alpaca_broker.is_connected:
                    positions_detail = alpaca_broker.get_positions_detail()
                    if positions_detail:
                        pv = sum(p.get("market_value", 0) for p in positions_detail.values())
                        try:
                            account = alpaca_broker._api.get_account()
                            pv += float(account.cash)
                        except Exception:
                            pv = max(pv, 100_000.0)
                        logger.info("[cron] Current positions: %s (PV=$%.0f)",
                                    list(positions_detail.keys()), pv)
                    else:
                        try:
                            status = alpaca_broker.get_account_status()
                            pv = status.get("portfolio_value", pv)
                        except Exception:
                            pass

                req = AutoDecideRequest(
                    portfolio_value=pv,
                    current_positions=positions_detail,
                )
                result = await multi_agent_decide(req)
                decisions = result.get("decisions", [])
                n = len(decisions)
                logger.info("[cron] Decision complete: %d signals, pipeline=%s",
                            n, result.get("pipeline", "v3"))

                # Track entry dates for max_hold_days
                for d in decisions:
                    if "BUY" in d.get("action", "") and d["ticker"] not in _position_entry_dates:
                        _position_entry_dates[d["ticker"]] = datetime.now()

                # Execute via broker (dry_run controlled by env var)
                dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                if decisions and alpaca_broker.is_connected:
                    exec_result = alpaca_broker.execute_decisions(decisions, dry_run=dry_run)
                    logger.info("[cron] Execution: %d/%d orders (%s)",
                                exec_result.get("successful", 0),
                                exec_result.get("total_orders", 0),
                                "dry_run" if dry_run else "LIVE")

                    # Register BUY executions with position tracker
                    for d in decisions:
                        if "BUY" in d.get("action", "") and d.get("price", 0) > 0:
                            qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                            if qty > 0:
                                position_tracker.track(
                                    ticker=d["ticker"],
                                    broker="alpaca",
                                    entry_price=d["price"],
                                    qty=qty,
                                    market_type="equity",
                                    regime=result.get("regime", {}).get("state", "unknown"),
                                )

                elif decisions:
                    logger.info("[cron] %d decisions but broker not connected (skipping execution)", n)
            except Exception as exc:
                logger.error("[cron] Decision failed: %s", exc)

    # Daily: regime check + event publish
    async def cron_daily_regime():
        logger.info("[cron] Running daily regime check...")
        try:
            view = await market_agent.analyze(top_n=5)
            logger.info("[cron] Regime: %s (%.0f%%), top=%s",
                        view.regime, view.regime_confidence * 100,
                        [s.get("sector", "?") for s in view.top_sectors[:3]])
        except Exception as exc:
            logger.error("[cron] Regime check failed: %s", exc)

    # Weekly (Sunday): retrain XGBoost + RL ensemble with dynamic tickers
    async def cron_weekly_retrain():
        logger.info("[cron] Running weekly XGBoost v8 retrain (18mo)...")
        try:
            result = await quant_agent.retrain(months=18)
            logger.info("[cron] XGBoost retrain: %s", result.get("status", "unknown"))
        except Exception as exc:
            logger.error("[cron] XGBoost retrain failed: %s", exc)

        # RL ensemble retrain with dynamic tickers from recent scans
        try:
            recent_events = event_bus.get_recent_events(200)
            scan_tickers: Dict[str, int] = {}
            for ev in recent_events:
                if ev.get("topic") == "decision.signal":
                    tic = ev.get("data", {}).get("ticker")
                    if tic:
                        scan_tickers[tic] = scan_tickers.get(tic, 0) + 1

            if scan_tickers:
                # Top 15 most frequently recommended tickers
                sorted_tickers = sorted(scan_tickers, key=scan_tickers.get, reverse=True)[:15]
                logger.info("[cron] RL retrain tickers (top %d): %s",
                            len(sorted_tickers), sorted_tickers)
            else:
                # Fallback to default
                sorted_tickers = list(TRAIN_CONFIG["tickers"])
                logger.info("[cron] RL retrain: no recent scans, using default tickers")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _executor,
                lambda: _train_rl_sync(sorted_tickers),
            )
        except Exception as exc:
            logger.error("[cron] RL retrain failed: %s", exc)

    # 5-minute risk check: TP/SL/trailing stop + max_hold_days for all held positions
    async def cron_risk_check():
        if _cron_lock.locked():
            return
        async with _cron_lock:
            if not alpaca_broker.is_connected:
                return
            try:
                positions = alpaca_broker.get_positions_detail()
                if not positions:
                    return

                dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                exits = []
                now = datetime.now()

                for tic, pos_info in positions.items():
                    entry_price = pos_info.get("entry_price", 0.0)
                    current_price = pos_info.get("current_price", 0.0)
                    if entry_price <= 0 or current_price <= 0:
                        continue

                    # Update trailing high
                    risk_manager.update_trailing_high(tic, current_price)

                    reason = None
                    if risk_manager.check_take_profit(tic, entry_price, current_price):
                        reason = "take_profit"
                    elif risk_manager.check_stop_loss(tic, entry_price, current_price):
                        reason = "stop_loss"
                    elif risk_manager.check_trailing_stop(tic, current_price):
                        reason = "trailing_stop"

                    # max_hold_days: force sell if held >10 days and <1% profit
                    if reason is None and tic in _position_entry_dates:
                        held_days = (now - _position_entry_dates[tic]).days
                        pnl_pct = (current_price - entry_price) / entry_price
                        if held_days > 10 and pnl_pct < 0.01:
                            reason = f"max_hold_{held_days}d_pnl_{pnl_pct:+.1%}"

                    if reason:
                        exits.append({
                            "ticker": tic,
                            "action": "SELL",
                            "confidence": -1.0,
                            "position_size_usd": pos_info.get("market_value", 0),
                            "price": current_price,
                            "reasons": [f"risk_{reason}"],
                        })
                        risk_manager.reset_trailing(tic)
                        _position_entry_dates.pop(tic, None)

                # Clean up entry dates for positions no longer held
                held_tickers = set(positions.keys())
                for tic in list(_position_entry_dates.keys()):
                    if tic not in held_tickers:
                        del _position_entry_dates[tic]

                if exits:
                    logger.info("[cron_risk] %d risk exits: %s", len(exits),
                                [(e["ticker"], e["reasons"][0]) for e in exits])
                    alpaca_broker.execute_decisions(exits, dry_run=dry_run)
                else:
                    logger.debug("[cron_risk] All %d positions OK", len(positions))

            except Exception as exc:
                logger.error("[cron_risk] Risk check failed: %s", exc)

    # --- Crypto cron: 24/7, 30-minute decisions, 10-minute risk checks ---
    # Uses validated Regime Blend pipeline (Sharpe 1.65, Alpha +9.2% vs BTC)

    def _fetch_crypto_ohlcv_df() -> "pd.DataFrame":
        """Fetch OHLCV for Regime Blend tickers -> pandas DataFrame (close prices)."""
        import pandas as pd
        cfg = REGIME_BLEND_CONFIG
        frames = {}
        for tic in cfg["tickers"]:
            try:
                ohlcv = binance_broker.fetch_ohlcv(
                    tic, timeframe="1d", limit=cfg["ohlcv_lookback_days"]
                )
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df.set_index("timestamp")
                    frames[tic] = df["close"]
            except Exception as e:
                logger.warning("[cron_crypto] OHLCV fetch failed for %s: %s", tic, e)
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)

    async def cron_crypto_decide():
        """30-minute crypto decision via Regime Blend pipeline."""
        if _cron_lock.locked():
            return
        async with _cron_lock:
            if not binance_broker.is_connected:
                return
            try:
                logger.info("[cron_crypto] Running Regime Blend pipeline...")
                import pandas as pd

                close_df = _fetch_crypto_ohlcv_df()
                if close_df.empty:
                    logger.warning("[cron_crypto] No OHLCV data, skipping")
                    return

                status = binance_broker.get_account_status()
                pv = status.get("portfolio_value", 1000.0)
                positions_detail = binance_broker.get_positions_detail()

                # Build positions dict from broker state + local entry prices
                positions: Dict[str, Dict] = {}
                for tic, info in (positions_detail or {}).items():
                    entry_px = _crypto_entry_prices.get(tic, info.get("entry_price", 0))
                    if entry_px > 0 and info.get("qty", 0) > 0:
                        positions[tic] = {
                            "qty": info["qty"],
                            "entry_price": entry_px,
                            "entry_date": "",
                        }

                eval_date = close_df.index[-1]
                cfg = REGIME_BLEND_CONFIG
                btc_ticker = cfg["tickers"][0]  # BTC/USDT

                pipe = Pipeline([
                    RegimeBlendDetectNode(),
                    RegimeBlendSignalNode(),
                    RegimeBlendExitNode(),
                    RegimeBlendEntryNode(),
                ])

                cash = pv - sum(
                    p["qty"] * float(close_df[t].iloc[-1])
                    for t, p in positions.items()
                    if t in close_df.columns
                )
                cash = max(cash, 0)

                ctx = {
                    "crypto_close": close_df,
                    "eval_date": eval_date,
                    "btc_ticker": btc_ticker,
                    "candidates": [t for t in cfg["tickers"] if t in close_df.columns],
                    "positions": positions,
                    "cash": cash,
                    "trailing_highs": dict(_crypto_trailing_highs),
                    "rb_config": cfg,
                }

                ctx = pipe.run(ctx)
                logger.info("[cron_crypto] Pipeline: %s, regime=%s",
                            pipe.summary(), ctx.get("regime", "?"))

                # Sync trailing highs back to module state
                _crypto_trailing_highs.clear()
                _crypto_trailing_highs.update(ctx.get("trailing_highs", {}))

                # Convert trade_log to broker decisions
                decisions = []
                for trade in ctx.get("trade_log", []):
                    decisions.append({
                        "ticker": trade["ticker"],
                        "action": trade["side"],
                        "confidence": trade.get("score", 0.5) if trade["side"] == "BUY" else -1.0,
                        "position_size_usd": trade["qty"] * trade["price"],
                        "price": trade["price"],
                        "reasons": [trade.get("reason", "regime_blend")],
                    })

                dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                if decisions:
                    # Track entry prices
                    for d in decisions:
                        if d["action"] == "BUY" and d.get("price", 0) > 0:
                            _crypto_entry_prices[d["ticker"]] = d["price"]
                        elif d["action"] == "SELL":
                            _crypto_entry_prices.pop(d["ticker"], None)

                    exec_result = binance_broker.execute_decisions(decisions, dry_run=dry_run)
                    logger.info("[cron_crypto] Execution: %d/%d orders (%s)",
                                exec_result.get("successful", 0),
                                exec_result.get("total_orders", 0),
                                "dry_run" if dry_run else "LIVE")

                    # Register BUY executions with position tracker
                    for d in decisions:
                        if d["action"] == "BUY" and d.get("price", 0) > 0:
                            qty = d.get("position_size_usd", 0) / d["price"] if d["price"] > 0 else 0
                            if qty > 0:
                                position_tracker.track(
                                    ticker=d["ticker"],
                                    broker="binance",
                                    entry_price=d["price"],
                                    qty=qty,
                                    market_type="crypto",
                                    regime=ctx.get("regime", "unknown"),
                                )
                else:
                    logger.info("[cron_crypto] No trades this cycle (regime=%s)", ctx.get("regime"))

            except Exception as exc:
                logger.error("[cron_crypto] Decision failed: %s", exc)

    async def cron_crypto_risk():
        """10-minute crypto risk check using Regime Blend trail stop."""
        if _cron_lock.locked():
            return
        async with _cron_lock:
            if not binance_broker.is_connected:
                return
            try:
                positions = binance_broker.get_positions_detail()
                if not positions:
                    return

                cfg = REGIME_BLEND_CONFIG
                dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
                exits = []

                for tic, pos_info in positions.items():
                    entry_price = _crypto_entry_prices.get(tic, pos_info.get("entry_price", 0.0))
                    current_price = pos_info.get("current_price", 0.0)
                    if entry_price <= 0 or current_price <= 0:
                        continue

                    # Update trailing high
                    prev_high = _crypto_trailing_highs.get(tic, entry_price)
                    if current_price > prev_high:
                        _crypto_trailing_highs[tic] = current_price
                        prev_high = current_price

                    pnl_pct = (current_price - entry_price) / entry_price
                    reason = None

                    # Trailing stop (activate after trail_activation_pct gain)
                    if prev_high > entry_price * (1 + cfg["trail_activation_pct"]):
                        trail_stop = prev_high * (1 - cfg["trail_pct"])
                        if current_price <= trail_stop:
                            reason = f"trail_stop ({pnl_pct:+.1%})"

                    # Per-position drawdown check
                    if reason is None and pnl_pct < -cfg["dd_trigger"]:
                        reason = f"dd_exit ({pnl_pct:+.1%})"

                    if reason:
                        exits.append({
                            "ticker": tic,
                            "action": "SELL",
                            "confidence": -1.0,
                            "position_size_usd": pos_info.get("market_value", 0),
                            "price": current_price,
                            "reasons": [f"risk_{reason}"],
                        })
                        _crypto_entry_prices.pop(tic, None)
                        _crypto_trailing_highs.pop(tic, None)

                if exits:
                    logger.info("[cron_crypto_risk] %d exits: %s", len(exits),
                                [(e["ticker"], e["reasons"][0]) for e in exits])
                    binance_broker.execute_decisions(exits, dry_run=dry_run)
            except Exception as exc:
                logger.error("[cron_crypto_risk] Failed: %s", exc)

    # --- Equity schedule: US market hours only ---
    scheduler.add_job(cron_decide, "cron", minute="*/15", hour="14-20",
                      day_of_week="mon-fri", id="decide_15m")
    scheduler.add_job(cron_risk_check, "cron", minute="*/5", hour="14-20",
                      day_of_week="mon-fri", id="risk_check_5m")
    # Daily at 14:00 UTC (market open)
    scheduler.add_job(cron_daily_regime, "cron", hour=14, minute=0,
                      day_of_week="mon-fri", id="daily_regime")
    # Weekly Sunday 10:00 UTC
    scheduler.add_job(cron_weekly_retrain, "cron", day_of_week="sun",
                      hour=10, minute=0, id="weekly_retrain")

    # --- Crypto schedule: 24/7 ---
    scheduler.add_job(cron_crypto_decide, "cron", minute="*/30",
                      id="crypto_decide_30m")
    scheduler.add_job(cron_crypto_risk, "cron", minute="*/10",
                      id="crypto_risk_10m")

    scheduler.start()
    logger.info("[cron] Scheduler started: equity(decide@15m, risk@5m) + crypto(decide@30m, risk@10m)")
    return scheduler


app = FastAPI(
    title="OpenClaw Trading Engine",
    description="Real-time multi-agent trading engine with WebSocket streaming, XGBoost cross-sectional ranking, and ensemble risk management",
    version="3.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------
# MCP integration — expose all endpoints as Claude-callable tools
# -----------------------------------------------------------------------
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        name="trading-engine",
        description="OpenClaw Trading Engine — market analysis, trade execution, and risk management",
    )
    mcp.mount()
    logger.info("[mcp] FastAPI-MCP mounted — all endpoints available as Claude tools")
except ImportError:
    logger.info("[mcp] fastapi-mcp not installed — MCP integration disabled. pip install fastapi-mcp")


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check returning model status."""
    return HealthResponse(
        status="ok",
        model_loaded=ensemble_agent.is_trained,
        training_status=app_state.training_status.value,
        timestamp=datetime.now(tz=None).isoformat(),
    )


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """Kick off ensemble training in a background thread.

    Returns immediately with the training status.  Poll ``GET /status``
    to check progress.
    """
    if app_state.training_status == TrainingStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    app_state.training_status = TrainingStatus.RUNNING
    app_state.training_started_at = datetime.now(tz=None).isoformat()
    app_state.training_error = None
    app_state.last_train_config = request.model_dump()

    # Run the heavy work in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _train_sync, request)

    return TrainResponse(
        message="Training started in background.",
        training_status=TrainingStatus.RUNNING.value,
        started_at=app_state.training_started_at,
    )


def _train_sync(request: TrainRequest) -> None:
    """Synchronous training function executed in the thread pool."""
    try:
        logger.info("Background training started with config: %s", request.model_dump())

        # 1. Prepare data
        data = data_processor.prepare_train_test(
            tickers=request.tickers,
            lookback_days=request.lookback_days,
            train_split=request.train_split,
        )

        train_array: np.ndarray = data["train_array"]
        test_array: np.ndarray = data["test_array"]

        # Update agent dimensions
        ensemble_agent.num_tickers = train_array.shape[1]
        ensemble_agent.num_features = train_array.shape[2]

        # 2. Train (with optional checkpoint continuation)
        train_metrics = ensemble_agent.train(
            train_array,
            total_timesteps=request.total_timesteps,
            learning_rate=request.learning_rate,
            continue_from_checkpoint=request.continue_from_checkpoint,
        )

        # 3. Backtest on test set
        if test_array.shape[0] > 1:
            test_metrics = ensemble_agent.backtest(test_array)
        else:
            test_metrics = {"note": "Not enough test data for backtesting."}

        app_state.last_metrics = {
            "train": train_metrics,
            "test": test_metrics,
            "tickers": data["tickers"],
            "train_days": int(train_array.shape[0]),
            "test_days": int(test_array.shape[0]),
        }
        app_state.training_status = TrainingStatus.COMPLETED
        app_state.training_completed_at = datetime.now(tz=None).isoformat()

        logger.info("Training completed successfully.")

    except Exception as exc:
        app_state.training_status = TrainingStatus.FAILED
        app_state.training_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Training failed: %s", exc, exc_info=True)


def _train_rl_sync(tickers: List[str]) -> None:
    """Synchronous RL ensemble retrain with dynamic tickers."""
    try:
        logger.info("RL retrain started with %d tickers: %s", len(tickers), tickers)
        data = data_processor.prepare_train_test(
            tickers=tickers,
            lookback_days=TRAIN_CONFIG["lookback_days"],
            train_split=TRAIN_CONFIG["train_split"],
        )
        train_array = data["train_array"]
        ensemble_agent.num_tickers = train_array.shape[1]
        ensemble_agent.num_features = train_array.shape[2]
        ensemble_agent.train(
            train_array,
            total_timesteps=TRAIN_CONFIG["total_timesteps"],
            learning_rate=TRAIN_CONFIG["learning_rate"],
            continue_from_checkpoint=True,
        )
        logger.info("RL retrain completed for %d tickers.", len(tickers))
    except Exception as exc:
        logger.error("RL retrain failed: %s", exc, exc_info=True)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict trading actions.

    Either provide a raw ``observation`` vector or ``tickers`` (in which
    case the latest market data is fetched and processed automatically).
    """
    if not ensemble_agent.is_trained:
        raise HTTPException(status_code=503, detail="No trained model available. Train first via POST /train.")

    try:
        if request.observation is not None:
            obs = np.array(request.observation, dtype=np.float32)
        elif request.tickers is not None:
            # Fetch latest data and build observation
            from datetime import timedelta

            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

            raw_df = data_processor.fetch_data(request.tickers, start, end, save_raw=False)
            enriched_df = data_processor.add_technical_indicators(raw_df)
            feature_array, _ = data_processor.create_feature_array(enriched_df)

            if feature_array.shape[0] == 0:
                raise HTTPException(status_code=400, detail="No data available for the requested tickers.")

            # Use the last day's data as the observation, build a full obs
            from analysis.ensemble_agent import TradingEnv

            env = TradingEnv(feature_array)
            # Fast-forward to the last step
            obs, _ = env.reset()
            for step_idx in range(feature_array.shape[0] - 1):
                zero_action = np.zeros(env.num_tickers, dtype=np.float32)
                obs, _, done, _, _ = env.step(zero_action)
                if done:
                    break
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'tickers' or 'observation' in the request.",
            )

        # Get ensemble prediction
        raw_actions = ensemble_agent.predict(obs)

        # Determine ticker names
        tickers = request.tickers or TRAIN_CONFIG["tickers"]
        # Pad/trim to match action size
        if len(tickers) < len(raw_actions):
            tickers = tickers + [f"TICKER_{i}" for i in range(len(tickers), len(raw_actions))]
        elif len(tickers) > len(raw_actions):
            tickers = tickers[: len(raw_actions)]

        # Apply risk checks to each action
        actions: List[PredictAction] = []
        portfolio_state: Dict[str, Any] = {
            "portfolio_value": 1_000_000.0,
            "cash": 500_000.0,
            "positions": {},
            "win_rate": 0.55,
            "avg_win": 0.02,
            "avg_loss": 0.01,
        }

        for i, (tic, act_val) in enumerate(zip(tickers, raw_actions)):
            risk_result = risk_manager.evaluate_action(float(act_val), tic, portfolio_state)
            actions.append(
                PredictAction(
                    ticker=tic,
                    action=float(risk_result["adjusted_action"]),
                    risk_check={
                        "allowed": risk_result["allowed"],
                        "reason": risk_result["reason"],
                        "position_size": risk_result["position_size"],
                        "signals": risk_result["signals"],
                    },
                )
            )

        return PredictResponse(
            actions=actions,
            ensemble_weights=ensemble_agent.weights,
            timestamp=datetime.now(tz=None).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return comprehensive status of the trading engine."""
    return StatusResponse(
        model_loaded=ensemble_agent.is_trained,
        training_status=app_state.training_status.value,
        training_started_at=app_state.training_started_at,
        training_completed_at=app_state.training_completed_at,
        training_error=app_state.training_error,
        last_train_config=app_state.last_train_config,
        performance_metrics=app_state.last_metrics,
        risk_manager_status=risk_manager.get_status(),
        ensemble_weights=ensemble_agent.weights,
        timestamp=datetime.now(tz=None).isoformat(),
    )


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a backtest on historical data.

    If ``start_date`` and ``end_date`` are provided they are used
    directly.  Otherwise ``lookback_days`` from today is used.
    """
    if not ensemble_agent.is_trained:
        raise HTTPException(status_code=503, detail="No trained model available. Train first via POST /train.")

    try:
        from datetime import timedelta

        if request.end_date:
            end = request.end_date
        else:
            end = datetime.now().strftime("%Y-%m-%d")

        if request.start_date:
            start = request.start_date
        else:
            start = (datetime.now() - timedelta(days=request.lookback_days)).strftime("%Y-%m-%d")

        # Fetch and process data
        raw_df = data_processor.fetch_data(request.tickers, start, end, save_raw=False)
        enriched_df = data_processor.add_technical_indicators(raw_df)
        feature_array, _ = data_processor.create_feature_array(enriched_df)

        if feature_array.shape[0] < 2:
            raise HTTPException(status_code=400, detail="Not enough data for backtesting. Try a longer date range.")

        # Resize agent if tickers don't match training data
        if feature_array.shape[1] != ensemble_agent.num_tickers:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Ticker count mismatch: backtest has {feature_array.shape[1]} tickers "
                    f"but model was trained on {ensemble_agent.num_tickers}. "
                    f"Use the same tickers as training."
                ),
            )

        metrics = ensemble_agent.backtest(feature_array)

        return BacktestResponse(
            metrics=metrics,
            tickers=request.tickers,
            start_date=start,
            end_date=end,
            num_trading_days=int(feature_array.shape[0]),
            timestamp=datetime.now(tz=None).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc


@app.post("/scan", response_model=ScanResponse)
async def scan_sectors(request: ScanRequest) -> ScanResponse:
    """Scan sector ETFs for relative momentum and pick top stocks.

    Does NOT require a trained model — uses pure market data analysis
    with momentum, volume, RSI, and news sentiment scoring.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: sector_scanner.full_pipeline(
                top_sectors=request.top_sectors,
                stocks_per_sector=request.stocks_per_sector,
                include_sentiment=request.include_sentiment,
            ),
        )
        return ScanResponse(**result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Sector scan failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sector scan failed: {exc}") from exc


@app.post("/decide")
async def auto_decide(request: AutoDecideRequest) -> Dict[str, Any]:
    """Autonomous trading decision.

    Synthesizes regime, sector momentum, stock factors, FinBERT sentiment
    into actionable buy/sell/hold decisions with confidence scores and
    position sizing.  This is the "brain" of the auto-trader.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: auto_trader.decide(
                portfolio_value=request.portfolio_value,
                top_sectors=request.top_sectors,
                stocks_per_sector=request.stocks_per_sector,
                include_sentiment=request.include_sentiment,
            ),
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auto decide failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Auto decide failed: {exc}") from exc


@app.post("/execute")
async def execute_trades(request: ExecuteRequest) -> Dict[str, Any]:
    """Execute trading decisions via Alpaca broker.

    By default runs in dry_run mode (no real orders).
    Set dry_run=false to actually submit orders (requires Alpaca API keys).
    """
    try:
        decisions = request.decisions

        # Auto-decide if no decisions provided (uses v3 multi-agent pipeline)
        if decisions is None and request.auto_decide:
            req = AutoDecideRequest(portfolio_value=request.portfolio_value)
            decide_result = await multi_agent_decide(req)
            decisions = decide_result.get("decisions", [])

        if not decisions:
            return {
                "message": "No decisions to execute",
                "dry_run": request.dry_run,
            }

        # Execute via broker
        result = alpaca_broker.execute_decisions(decisions, dry_run=request.dry_run)
        return result

    except Exception as exc:
        logger.error("Trade execution failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {exc}") from exc


@app.get("/broker/status")
async def broker_status() -> Dict[str, Any]:
    """Get status of all registered brokers."""
    return broker_registry.status_all()


@app.get("/tracker/status")
async def tracker_status() -> Dict[str, Any]:
    """Get position tracker status — all actively monitored positions."""
    return position_tracker.get_status()


@app.get("/learner/status")
async def learner_status() -> Dict[str, Any]:
    """Get online learner status — Thompson Sampling agent weight adaptation."""
    return online_learner.get_status()


@app.get("/learner/weights")
async def learner_weights() -> Dict[str, Any]:
    """Get current adapted agent weights (sampled from posteriors)."""
    return {
        "sampled_weights": online_learner.sample_weights(),
        "mean_weights": online_learner.get_mean_weights(),
        "has_enough_data": online_learner.has_enough_data,
        "total_trades": online_learner.total_trades,
    }


@app.get("/claude/status")
async def claude_status() -> Dict[str, Any]:
    """Check Claude agent status (OpenClaw OAuth, SDK availability)."""
    agent = ClaudeAgent()
    return {
        "oauth_token_found": is_claude_available(),
        "agent_available": agent.is_available,
        "mode": "primary_decision_maker",
        "note": "Claude IS the trader. Rule-based pipeline is backtesting fallback only.",
    }


@app.get("/regime")
async def get_regime() -> Dict[str, Any]:
    """Detect the current market regime using HMM.

    Returns regime state (low_vol/high_vol), confidence,
    and strategy adjustments.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            regime_detector.detect,
        )
        adjustments = regime_detector.get_adjustments(result)
        return {**result, "adjustments": adjustments}

    except Exception as exc:
        logger.error("Regime detection failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Regime detection failed: {exc}") from exc


@app.post("/scan/backtest", response_model=ScanBacktestResponse)
async def scan_backtest(request: ScanBacktestRequest) -> ScanBacktestResponse:
    """Backtest the sector scanner over N months.

    Simulates what the scanner would have recommended at each monthly
    evaluation point and measures forward returns vs SPY.
    Does NOT require a trained model.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_scan_backtest(months=request.months),
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return ScanBacktestResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Scan backtest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scan backtest failed: {exc}") from exc


# -----------------------------------------------------------------------
# v3: WebSocket streaming endpoint
# -----------------------------------------------------------------------


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """Real-time trading event stream.

    Clients receive events as JSON:
        {"type": "regime.update", "data": {...}, "timestamp": ..., "seq": ...}

    Clients can send commands:
        {"action": "subscribe", "topics": ["regime.update", "signal.buy"]}
        {"action": "unsubscribe", "topics": ["signal.sell"]}
        {"action": "ping"}
    """
    await trading_stream.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await trading_stream.handle_client_message(websocket, data)
    except WebSocketDisconnect:
        trading_stream.disconnect(websocket)


@app.get("/ws/status")
async def ws_status() -> Dict[str, Any]:
    """Get WebSocket stream status."""
    return trading_stream.get_status()


@app.get("/events/recent")
async def recent_events(n: int = 20) -> Dict[str, Any]:
    """Get recent EventBus events."""
    return {
        "events": event_bus.get_recent_events(n),
        "topics": event_bus.topics,
    }


# -----------------------------------------------------------------------
# v3: Multi-agent decision endpoint
# -----------------------------------------------------------------------


@app.post("/decide/v3")
async def multi_agent_decide(request: AutoDecideRequest) -> Dict[str, Any]:
    """Multi-agent trading decision (v3 pipeline).

    Runs the full multi-agent pipeline:
        1. MarketAgent → regime + sector analysis
        2. QuantAgent → cross-sectional XGBoost rankings
        3. Synthesizer → weighted voting → final decisions

    Falls back to v2 auto_trader.decide() on error.
    """
    try:
        # Auto-fetch positions from Alpaca if not provided (tool calls)
        if request.current_positions is None and alpaca_broker.is_connected:
            fetched = alpaca_broker.get_positions_detail()
            if fetched:
                request.current_positions = fetched
                logger.info("[decide/v3] Auto-fetched %d positions from Alpaca", len(fetched))

        # Publish pipeline start
        await event_bus.publish("agent.status", {
            "stage": "pipeline_start",
            "portfolio_value": request.portfolio_value,
        })

        # Step 1: MarketAgent — regime + sector analysis
        market_view = await market_agent.analyze(top_n=request.top_sectors)

        # Step 2: Gather candidate tickers from top sectors
        from agents.quant_agent import SECTOR_MAP
        candidates = []
        for sec_info in market_view.top_sectors:
            sector_name = sec_info.get("sector", "")
            if sector_name in SECTOR_MAP:
                candidates.extend(SECTOR_MAP[sector_name]["stocks"])
        # Expand to more sectors if few candidates
        if len(candidates) < 15:
            for sec_info in market_view.sector_scores[request.top_sectors:]:
                sector_name = sec_info.get("sector", "")
                if sector_name in SECTOR_MAP:
                    candidates.extend(SECTOR_MAP[sector_name]["stocks"])
                if len(candidates) >= 30:
                    break
        candidates = list(set(candidates))

        # Step 3: QuantAgent — z-scored ranking (v8)
        # Pass sector scores from MarketAgent for sector_momentum feature
        sector_score_map = {}
        for sec_info in (market_view.sector_scores or []):
            name = sec_info.get("sector", "")
            score = sec_info.get("composite_score", sec_info.get("score", 0.0))
            sector_score_map[name] = score

        quant_rankings = await quant_agent.rank_stocks(candidates, sector_scores=sector_score_map)

        # Step 4: Synthesizer — weighted voting (with position awareness)
        # Use online-learned weights if available (Thompson Sampling RL)
        adapted_weights = None
        if online_learner.has_enough_data:
            adapted_weights = online_learner.sample_weights()
            logger.info("[decide/v3] Using adapted weights: %s",
                        {k: f"{v:.2f}" for k, v in adapted_weights.items()})

        decisions = await synthesizer.synthesize(
            market_view=market_view,
            quant_rankings=quant_rankings,
            portfolio_value=request.portfolio_value,
            current_positions=request.current_positions,
            include_sentiment=request.include_sentiment,
            adaptive_weights=adapted_weights,
        )

        # Convert to legacy format for backwards compatibility
        result = decisions_to_legacy(decisions, market_view, request.portfolio_value)
        if adapted_weights:
            result["adaptive_weights"] = {k: round(v, 3) for k, v in adapted_weights.items()}
            result["online_learner_trades"] = online_learner.total_trades

        # Publish pipeline complete
        await event_bus.publish("agent.status", {
            "stage": "pipeline_complete",
            "total_decisions": len(decisions),
        })

        return result

    except Exception as exc:
        logger.error("Multi-agent pipeline failed: %s", exc, exc_info=True)
        # Fallback to v2 pipeline
        logger.info("Falling back to v2 auto_trader.decide()")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: auto_trader.decide(
                    portfolio_value=request.portfolio_value,
                    top_sectors=request.top_sectors,
                    stocks_per_sector=request.stocks_per_sector,
                    include_sentiment=request.include_sentiment,
                ),
            )
            result["pipeline"] = "v2_fallback"
            result["v3_error"] = str(exc)
            return result
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Both v3 and v2 pipelines failed. v3: {exc}, v2: {fallback_exc}",
            ) from fallback_exc


@app.post("/quant/rank")
async def quant_rank(request: ScanRequest) -> Dict[str, Any]:
    """Run QuantAgent cross-sectional ranking independently."""
    try:
        from agents.quant_agent import SECTOR_MAP
        # Gather all stocks from top sectors
        candidates = []
        for name, info in SECTOR_MAP.items():
            candidates.extend(info["stocks"])
        candidates = list(set(candidates))

        rankings = await quant_agent.rank_stocks(candidates)
        return {
            "rankings": [r.to_dict() for r in rankings[:20]],
            "total_ranked": len(rankings),
            "model_trained": quant_agent.is_trained,
        }
    except Exception as exc:
        logger.error("Quant ranking failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quant ranking failed: {exc}") from exc


@app.post("/quant/retrain")
async def quant_retrain() -> Dict[str, Any]:
    """Retrain QuantAgent XGBoost v8 model (18-month window)."""
    try:
        result = await quant_agent.retrain(months=18)
        return result
    except Exception as exc:
        logger.error("Quant retrain failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quant retrain failed: {exc}") from exc


# -----------------------------------------------------------------------
# Risk check endpoint (exposes cron_risk_check logic for manual trigger)
# -----------------------------------------------------------------------


class RiskCheckResponse(BaseModel):
    checked: int = Field(description="Number of positions checked")
    exits: List[Dict[str, Any]] = Field(default_factory=list)
    dry_run: bool
    broker_connected: bool


@app.post("/risk/check", response_model=RiskCheckResponse)
async def risk_check() -> RiskCheckResponse:
    """Manually trigger risk check (TP/SL/trailing stop) for all held positions."""
    if not alpaca_broker.is_connected:
        return RiskCheckResponse(checked=0, exits=[], dry_run=True, broker_connected=False)

    try:
        positions = alpaca_broker.get_positions_detail()
        if not positions:
            return RiskCheckResponse(checked=0, exits=[], dry_run=True, broker_connected=True)

        dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
        exits = []
        for tic, pos_info in positions.items():
            entry_price = pos_info.get("entry_price", 0.0)
            current_price = pos_info.get("current_price", 0.0)
            if entry_price <= 0 or current_price <= 0:
                continue

            risk_manager.update_trailing_high(tic, current_price)

            reason = None
            if risk_manager.check_take_profit(tic, entry_price, current_price):
                reason = "take_profit"
            elif risk_manager.check_stop_loss(tic, entry_price, current_price):
                reason = "stop_loss"
            elif risk_manager.check_trailing_stop(tic, current_price):
                reason = "trailing_stop"

            if reason:
                exits.append({
                    "ticker": tic,
                    "action": "SELL",
                    "confidence": -1.0,
                    "position_size_usd": pos_info.get("market_value", 0),
                    "price": current_price,
                    "reasons": [f"risk_{reason}"],
                })
                risk_manager.reset_trailing(tic)

        if exits:
            alpaca_broker.execute_decisions(exits, dry_run=dry_run)

        return RiskCheckResponse(
            checked=len(positions),
            exits=exits,
            dry_run=dry_run,
            broker_connected=True,
        )

    except Exception as exc:
        logger.error("Risk check failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk check failed: {exc}") from exc


# -----------------------------------------------------------------------
# Pipeline backtest endpoint
# -----------------------------------------------------------------------


class PipelineBacktestRequest(BaseModel):
    start_date: str = Field(default="2024-06-01", description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(default="2026-02-01", description="Backtest end date (YYYY-MM-DD)")
    rebalance_days: int = Field(default=21, ge=5, le=63, description="Rebalance period in trading days")
    initial_cash: float = Field(default=100_000.0, gt=0)
    use_xgboost: bool = Field(default=True, description="Use XGBoost ranking (vs momentum proxy)")


@app.post("/backtest/pipeline")
async def pipeline_backtest(request: PipelineBacktestRequest) -> Dict[str, Any]:
    """Run full v3 pipeline backtest (scan → rank → synthesize → execute).

    This is a long-running operation.
    """
    try:
        from backtests.backtest_pipeline import run_pipeline_backtest

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_pipeline_backtest(
                start_date=request.start_date,
                end_date=request.end_date,
                rebalance_days=request.rebalance_days,
                initial_cash=request.initial_cash,
                use_xgboost=request.use_xgboost,
            ),
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Strip large arrays for API response (keep trade_log, drop daily_values)
        result.pop("daily_values", None)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pipeline backtest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline backtest failed: {exc}") from exc


# -----------------------------------------------------------------------
# Chart dashboard + TradingView integration
# -----------------------------------------------------------------------

from fastapi.responses import HTMLResponse


@app.get("/chart", response_class=HTMLResponse)
async def chart_page():
    """TradingView Lightweight Charts dashboard with live Binance data."""
    chart_html = Path(__file__).parent / "templates" / "chart.html"
    return HTMLResponse(chart_html.read_text(encoding="utf-8"))


@app.get("/api/signals")
async def get_signals(symbol: str = "btcusdt") -> List[Dict[str, Any]]:
    """Recent trading signals for chart BUY/SELL markers overlay."""
    pair = symbol.upper()
    if "USDT" in pair and "/" not in pair:
        pair = pair.replace("USDT", "/USDT")

    signals = []
    for ev in event_bus.get_recent_events(500):
        payload = ev.get("payload", ev.get("data", {}))
        if not payload:
            continue
        ticker = payload.get("ticker", "")
        if ticker.replace("/", "").lower() != symbol.lower():
            continue
        action = payload.get("action", "")
        if action in ("BUY", "SELL"):
            signals.append({
                "time": int(ev.get("timestamp", 0)),
                "side": action,
                "price": payload.get("price", 0),
                "reason": (payload.get("reasons") or [""])[0],
            })
    return signals


class TVWebhookPayload(BaseModel):
    action: str
    symbol: str
    price: float
    qty: Optional[float] = None
    strategy: Optional[str] = None
    timestamp: Optional[str] = None


@app.post("/webhook/tradingview")
async def tradingview_webhook(payload: TVWebhookPayload) -> Dict[str, str]:
    """Receive Pine Script alert webhooks from TradingView."""
    logger.info("[webhook] TradingView: %s %s @ %.2f", payload.action, payload.symbol, payload.price)

    pair = payload.symbol
    if "USDT" in pair and "/" not in pair:
        pair = pair.replace("USDT", "/USDT")

    decision = {
        "ticker": pair,
        "action": payload.action.upper(),
        "confidence": 0.8,
        "price": payload.price,
        "reasons": [f"tv_webhook_{payload.strategy or 'manual'}"],
    }

    await event_bus.publish("webhook.tradingview", {
        "decision": decision,
        "raw": payload.model_dump(),
    })

    dry_run = os.environ.get("LIVE_TRADING", "").lower() not in ("1", "true", "yes")
    if binance_broker.is_connected and "USDT" in pair:
        binance_broker.execute_decisions([decision], dry_run=dry_run)

    return {"status": "received", "action": payload.action}


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TRADING_SERVER_PORT", SERVER_CONFIG["port"]))
    host = os.environ.get("TRADING_SERVER_HOST", SERVER_CONFIG["host"])
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
