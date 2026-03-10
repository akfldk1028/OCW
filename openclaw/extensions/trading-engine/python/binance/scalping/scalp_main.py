"""Scalping strategy CLI entrypoint.

Usage:
    # Standalone (testnet)
    python -m binance.scalping.scalp_main --testnet --futures --leverage 3

    # With shared DerivativesMonitor from swing strategy
    python -m binance.scalping.scalp_main --testnet --futures --leverage 3 --with-swing

    # Live trading
    python -m binance.scalping.scalp_main --live --futures --leverage 3
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

_SCALPING_DIR = Path(__file__).resolve().parent
_BINANCE_DIR = _SCALPING_DIR.parent
_ENGINE_DIR = _BINANCE_DIR.parent
for _p in (_SCALPING_DIR, _BINANCE_DIR, _ENGINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

from scalp_config import SCALP_LOG_LEVEL, SCALP_LOGS_DIR, SCALP_TICKERS
from brokers.binance import BinanceBroker
from scalp_runner import ScalpRunner


def _setup_logging(level: str) -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setStream(open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            stream_handler,
            logging.FileHandler(SCALP_LOGS_DIR / "scalping.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Scalping Strategy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    group.add_argument("--testnet", action="store_true", help="Use Binance testnet (default)")
    parser.add_argument("--futures", action="store_true", help="Use Futures market")
    parser.add_argument("--leverage", type=int, default=3, help="Futures leverage (default: 3x)")
    parser.add_argument("--with-swing", action="store_true",
                        help="Share DerivativesMonitor with swing strategy")
    args = parser.parse_args()

    # Load .env
    env_dir = _BINANCE_DIR
    if args.live:
        env_path = env_dir / ".env.live"
        if not env_path.exists():
            env_path = env_dir / ".env"
    else:
        env_path = env_dir / ".env"
    load_dotenv(env_path)

    if args.live:
        os.environ["LIVE_TRADING"] = "true"
        os.environ["BINANCE_PAPER"] = "false"
    else:
        os.environ.setdefault("LIVE_TRADING", "false")
        os.environ.setdefault("BINANCE_PAPER", "true")

    _setup_logging(SCALP_LOG_LEVEL)
    logger = logging.getLogger("scalp_main")

    market = "future" if args.futures else "spot"
    mode = "LIVE" if args.live else "TESTNET"
    logger.info("=" * 60)
    logger.info("Scalping Strategy - %s %s mode", mode, market.upper())
    logger.info("Tickers: %s", SCALP_TICKERS)
    logger.info("Leverage: %dx", args.leverage)
    logger.info("=" * 60)

    if args.live:
        logger.warning("LIVE TRADING ENABLED - real money at risk!")
    if args.leverage > 1 and not args.futures:
        logger.warning("Leverage > 1x requires --futures, ignoring leverage")
        args.leverage = 1

    # Connect broker
    broker = BinanceBroker(market=market)
    result = broker.connect()
    if not broker.is_connected:
        logger.error("Failed to connect to Binance: %s", result)
        sys.exit(1)

    usdt_bal = result.get("usdt_balance", 0)
    logger.info("Connected. USDT: $%.2f, Mode: %s", usdt_bal, result.get("mode", "?"))

    # Optional: shared DerivativesMonitor
    derivatives_monitor = None
    if args.with_swing:
        from core.derivatives_monitor import DerivativesMonitor
        from core.event_bus import EventBus as SwingBus
        deriv_bus = SwingBus()
        derivatives_monitor = DerivativesMonitor(
            event_bus=deriv_bus,
            broker=broker,
            tickers=SCALP_TICKERS,
            poll_base=120,
            poll_fast=30,
        )
        logger.info("DerivativesMonitor enabled for order flow signals")

    # Build runner
    runner = ScalpRunner(
        broker=broker,
        leverage=args.leverage,
        initial_balance=usdt_bal,
        derivatives_monitor=derivatives_monitor,
    )

    # Graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(runner.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down...")
        runner.stop()
    except Exception:
        logger.exception("FATAL: Unhandled exception in scalp runner")
    finally:
        loop.close()
        logger.info("Bye.")


if __name__ == "__main__":
    main()
