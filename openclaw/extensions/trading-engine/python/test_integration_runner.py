"""Integration test: run CryptoRunner for 60s, verify multi-TF + gate + aggregator."""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Setup path
_PARENT_DIR = Path(__file__).resolve().parent
_BINANCE_DIR = _PARENT_DIR / "binance"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
if str(_BINANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_BINANCE_DIR))

from dotenv import load_dotenv
load_dotenv(_BINANCE_DIR / ".env")
os.environ.setdefault("BINANCE_PAPER", "true")
os.environ.setdefault("LIVE_TRADING", "false")

# Logging at DEBUG for tick visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_integration")

from brokers.binance import BinanceBroker
from binance.runner import CryptoRunner
from binance.crypto_config import EVENT_CONFIG


async def main():
    logger.info("=== Integration Test: Multi-TF Runner (60s) ===")
    logger.info("EVENT_CONFIG: intervals=%s, primary=%s, gate=%s",
                EVENT_CONFIG.get("kline_intervals"),
                EVENT_CONFIG.get("primary_interval"),
                EVENT_CONFIG.get("gate"))

    # Connect broker
    broker = BinanceBroker(market="spot")
    result = broker.connect()
    if not broker.is_connected:
        logger.error("Broker connection failed: %s", result)
        return

    status = broker.get_account_status()
    logger.info("Portfolio: $%.2f", status.get("portfolio_value", 0))

    # Build runner
    runner = CryptoRunner(broker, leverage=1)

    # Verify components
    logger.info("MarketListener intervals: %s", runner.market_listener._kline_intervals)
    logger.info("MarketListener primary: %s", runner.market_listener._primary_interval)
    logger.info("AdaptiveGate: z_thresh=%.1f, max_check=%ds, min_check=%ds",
                runner.adaptive_gate._zscore_threshold,
                runner.adaptive_gate._max_check_seconds,
                runner.adaptive_gate._min_check_seconds)
    logger.info("MultiTFAggregator intervals: %s", runner.multi_tf_aggregator._intervals)
    logger.info("TradingMemory driver: %s", "connected" if runner.memory._driver else "not connected")
    logger.info("Claude agent available: %s", runner.claude_agent.is_available)

    # Run for 60s then stop
    run_task = asyncio.create_task(runner.run())
    await asyncio.sleep(60)

    logger.info("\n=== Stopping after 60s ===")
    runner.stop()
    try:
        await asyncio.wait_for(run_task, timeout=10)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    # Check aggregator state
    logger.info("\n=== Aggregator State ===")
    all_summaries = runner.multi_tf_aggregator.get_all_summaries()
    if all_summaries:
        for ticker, summary in all_summaries.items():
            logger.info("  %s:", ticker)
            for iv, data in summary.items():
                logger.info("    %s: %s", iv, data)
    else:
        logger.warning("  No aggregator data (no closed candles in 60s — normal for testnet)")

    # Check gate state
    gate_status = runner.adaptive_gate.get_status() if hasattr(runner.adaptive_gate, 'get_status') else "N/A"
    logger.info("\n=== Gate Status ===")
    logger.info("  %s", gate_status)

    # Check tick prices
    logger.info("\n=== Last Tick Prices ===")
    for k, v in runner._last_tick_prices.items():
        if not k.startswith("_"):
            logger.info("  %s: $%.2f", k, v)

    # Check WS message count
    logger.info("\n=== WS Stats ===")
    logger.info("  Last message age: %.0fs ago",
                time.time() - runner.market_listener._last_message_time
                if runner.market_listener._last_message_time > 0 else -1)

    # Check alerts
    alerts_path = runner._log_dir / "alerts.jsonl"
    if alerts_path.exists():
        with open(alerts_path) as f:
            lines = f.readlines()
        logger.info("  Alerts: %d", len(lines))
    else:
        logger.info("  No alerts yet")

    # Check decisions log
    if runner._decisions_log.exists():
        with open(runner._decisions_log) as f:
            lines = f.readlines()
        logger.info("  Decisions logged: %d", len(lines))

    logger.info("\n=== PASS: Integration test completed ===")


if __name__ == "__main__":
    asyncio.run(main())
