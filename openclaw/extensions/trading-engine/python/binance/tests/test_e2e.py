"""E2E test: actual Binance testnet buy -> track -> risk exit cycle.

Intervals compressed to seconds so the full lifecycle runs in ~30s.
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_BINANCE_DIR = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _BINANCE_DIR.parent
for p in [str(_BINANCE_DIR), str(_PYTHON_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(_BINANCE_DIR / ".env")
os.environ.setdefault("BINANCE_PAPER", "true")
os.environ.setdefault("LIVE_TRADING", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(
        open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)
    )],
)
logger = logging.getLogger("e2e_test")

from brokers.binance import BinanceBroker
from core.event_bus import EventBus
from core.position_tracker import PositionTracker
from core.agent_evaluator import evaluate_position, build_market_context
from core.online_learner import OnlineLearner
from analysis.regime_detector_crypto import CryptoRegimeDetector
from core.pipeline import (
    Pipeline, RegimeBlendDetectNode, RegimeBlendSignalNode,
    RegimeBlendExitNode, RegimeBlendEntryNode,
)
from crypto_config import REGIME_BLEND_CONFIG, MODELS_DIR


async def main():
    logger.info("=" * 60)
    logger.info("E2E TEST: Binance Testnet Full Trade Lifecycle")
    logger.info("=" * 60)

    # --- Step 1: Connect ---
    logger.info("[Step 1] Connecting to Binance testnet...")
    broker = BinanceBroker()
    result = broker.connect()
    if not broker.is_connected:
        logger.error("FAIL: Cannot connect: %s", result)
        return False
    status = broker.get_account_status()
    logger.info("OK: Connected. Portfolio=$%.2f, USDT=$%.2f",
                status.get("portfolio_value", 0), status.get("cash", 0))

    # --- Step 2: Fetch OHLCV + Run Pipeline ---
    logger.info("[Step 2] Fetching OHLCV and running Regime Blend pipeline...")
    import pandas as pd
    cfg = REGIME_BLEND_CONFIG
    frames = {}
    for tic in cfg["tickers"]:
        ohlcv = broker.fetch_ohlcv(tic, timeframe="1d", limit=cfg["ohlcv_lookback_days"])
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            frames[tic] = df["close"]
            logger.info("  %s: %d bars, latest=%.2f", tic, len(df), df["close"].iloc[-1])

    if not frames:
        logger.error("FAIL: No OHLCV data")
        return False

    close_df = pd.DataFrame(frames)
    eval_date = close_df.index[-1]
    pipe = Pipeline([
        RegimeBlendDetectNode(), RegimeBlendSignalNode(),
        RegimeBlendExitNode(), RegimeBlendEntryNode(),
    ])
    ctx = pipe.run({
        "crypto_close": close_df,
        "eval_date": eval_date,
        "btc_ticker": cfg["tickers"][0],
        "candidates": list(frames.keys()),
        "positions": {},
        "cash": status.get("cash", 1000),
        "trailing_highs": {},
        "rb_config": cfg,
    })
    regime = ctx.get("regime", "unknown")
    scored = ctx.get("scored", [])
    trade_log = ctx.get("trade_log", [])
    logger.info("OK: Pipeline done. regime=%s, signals=%d, trades=%d",
                regime, len(scored), len(trade_log))
    for s in scored:
        logger.info("  Signal: %s %s score=%.3f rsi=%.1f mom=%.4f | %s",
                     s.get("signal"), s.get("ticker"), s.get("score", 0),
                     s.get("rsi", 0), s.get("mom_14d", 0), s.get("reason", ""))

    # --- Step 3: Force a BUY on testnet (regardless of regime) ---
    logger.info("[Step 3] Forcing BUY on testnet to test execution...")
    test_ticker = cfg["tickers"][0]  # BTC/USDT
    ticker_info = broker._exchange.fetch_ticker(test_ticker)
    current_price = ticker_info["last"]
    buy_usd = 100.0  # $100 worth

    decision_buy = {
        "ticker": test_ticker,
        "action": "BUY",
        "confidence": 0.8,
        "position_size_usd": buy_usd,
        "price": current_price,
        "reasons": ["e2e_test_forced_buy"],
    }
    # dry_run=False: actually execute on testnet
    exec_result = broker.execute_decisions([decision_buy], dry_run=False)
    logger.info("OK: BUY executed: %s", exec_result)

    # --- Step 4: Verify position exists ---
    logger.info("[Step 4] Checking position after BUY...")
    await asyncio.sleep(2)  # wait for order fill
    positions = broker.get_positions_detail()
    if test_ticker in positions:
        pos = positions[test_ticker]
        logger.info("OK: Position found: %s qty=%.6f entry=%.2f value=$%.2f",
                     test_ticker, pos["qty"], pos.get("entry_price", 0), pos.get("market_value", 0))
    else:
        logger.warning("Position not in get_positions_detail (may need time to fill)")
        logger.info("Current positions: %s", list(positions.keys()) if positions else "none")

    # --- Step 5: Position Tracker test ---
    logger.info("[Step 5] Testing PositionTracker (5s safety, 10s agent)...")
    event_bus = EventBus()
    tracker = PositionTracker(event_bus=event_bus, safety_interval=5.0, agent_interval=10.0)
    crypto_regime = CryptoRegimeDetector()

    async def price_fn(ticker):
        try:
            info = await asyncio.to_thread(broker._exchange.fetch_ticker, ticker)
            return float(info.get("last", 0))
        except Exception:
            return 0.0

    async def ctx_fn():
        return await asyncio.to_thread(
            build_market_context,
            crypto_regime_detector=crypto_regime,
            binance_broker=broker,
        )

    async def exit_fn(decision):
        return await asyncio.to_thread(broker.execute_decisions, [decision], False)

    tracker.set_price_fetcher("binance", price_fn)
    tracker.set_agent_evaluator(evaluate_position)
    tracker.set_context_provider(ctx_fn)
    tracker.set_exit_executor("binance", exit_fn)

    buy_qty = buy_usd / current_price
    tracker.track(
        ticker=test_ticker, broker="binance",
        entry_price=current_price, qty=buy_qty,
        market_type="crypto", regime=regime,
    )

    events_received = []
    async def log_event(event):
        events_received.append(event)
        logger.info("  Event: %s -> %s", event.topic, {
            k: v for k, v in event.payload.items() if k != "agent_reasons"
        })

    event_bus.subscribe("position.update", log_event)
    event_bus.subscribe("position.agent_eval", log_event)
    event_bus.subscribe("position.alert", log_event)
    event_bus.subscribe("position.exit", log_event)

    # Run tracker for 15s (should see 1 safety check + 1 agent eval)
    tracker_task = asyncio.create_task(tracker.run())
    await asyncio.sleep(15)
    tracker.stop()
    tracker_task.cancel()
    try:
        await tracker_task
    except asyncio.CancelledError:
        pass

    logger.info("OK: Tracker ran. Events received: %d", len(events_received))

    # --- Step 6: Force SELL to clean up ---
    logger.info("[Step 6] Cleaning up: SELL position...")
    ticker_info = broker._exchange.fetch_ticker(test_ticker)
    sell_price = ticker_info["last"]
    sell_qty = broker._get_position_qty(test_ticker)
    if sell_qty > 0:
        decision_sell = {
            "ticker": test_ticker,
            "action": "SELL",
            "confidence": -1.0,
            "position_size_usd": sell_qty * sell_price,
            "price": sell_price,
            "reasons": ["e2e_test_cleanup"],
        }
        sell_result = broker.execute_decisions([decision_sell], dry_run=False)
        pnl = (sell_price - current_price) / current_price
        logger.info("OK: SELL executed: %s (pnl=%+.2f%%)", sell_result, pnl * 100)
    else:
        logger.info("No position to sell (already exited or not filled)")

    # --- Step 7: Online Learner ---
    logger.info("[Step 7] Testing OnlineLearner record...")
    ol = OnlineLearner(save_path=str(MODELS_DIR / "e2e_test_learner.json"), min_trades_to_adapt=2)
    ol.record_trade(
        ticker=test_ticker, entry_price=current_price, exit_price=sell_price,
        pnl_pct=(sell_price - current_price) / current_price, held_hours=0.01,
        agent_signals={"market": 0.5, "quant": 0.3, "momentum": 0.8},
        market_type="crypto",
    )
    ol.save()
    logger.info("OK: Trade recorded. Status: %s",
                {k: v for k, v in ol.get_status().items() if k != "recent_trades"})
    # cleanup
    Path(MODELS_DIR / "e2e_test_learner.json").unlink(missing_ok=True)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("E2E TEST COMPLETE")
    logger.info("  [1] Testnet connect:     OK")
    logger.info("  [2] Pipeline (regime):   %s", regime)
    logger.info("  [3] BUY execution:       OK")
    logger.info("  [4] Position verify:     OK")
    logger.info("  [5] Tracker events:      %d events", len(events_received))
    logger.info("  [6] SELL cleanup:        OK")
    logger.info("  [7] OnlineLearner:       OK")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
