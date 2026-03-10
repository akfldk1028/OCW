"""Binance crypto standalone trader entry point.

Usage:
    python main.py                          # testnet spot, 4h swing (default)
    python main.py --daily                  # testnet spot, daily bars
    python main.py --futures --leverage 3   # demo futures 3x (auto-enables demo mode)
    python main.py --live                   # real trading (CAUTION)
    python main.py --live --futures --leverage 2  # real futures 2x

Requires Binance API keys in binance/.env:
    BINANCE_API_KEY=...
    BINANCE_SECRET_KEY=...
    BINANCE_PAPER=true

For Spot testnet: keys from https://testnet.binance.vision/
For Futures (demo): keys from https://www.binance.com/en/demo-trading
  (Binance deprecated futures testnet/sandbox — demo trading replaces it)
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Force unbuffered stdout/stderr (critical for background/pipe execution)
os.environ["PYTHONUNBUFFERED"] = "1"

# Add parent dir for shared module imports
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from dotenv import load_dotenv

from crypto_config import LOG_LEVEL, LOGS_DIR
from brokers.binance import BinanceBroker
from runner import CryptoRunner


def _setup_logging(level: str) -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setStream(open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            stream_handler,
            logging.FileHandler(LOGS_DIR / "trading.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Crypto Autonomous Trader")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    group.add_argument("--testnet", action="store_true", help="Use Binance testnet (default)")
    parser.add_argument("--futures", action="store_true", help="Use Futures market (default: Spot)")
    parser.add_argument("--leverage", type=int, default=1, help="Futures leverage (default: 1x)")
    parser.add_argument("--demo", action="store_true", help="Use Binance demo trading (required for futures testnet)")
    parser.add_argument("--daily", action="store_true", help="Use daily bars instead of 4h swing")
    args = parser.parse_args()

    # Load .env from binance/ dir (.env.live for --live, .env for testnet)
    env_dir = Path(__file__).resolve().parent
    if args.live:
        env_path = env_dir / ".env.live"
        if not env_path.exists():
            env_path = env_dir / ".env"
    else:
        env_path = env_dir / ".env"
    load_dotenv(env_path)

    # === API KEY GUARD ===
    # Max 구독 CLI 사용 — API key가 환경에 있으면 제거 (과금 방지)
    for _key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if _key in os.environ:
            del os.environ[_key]
            logging.getLogger("main").warning(
                "BLOCKED: %s removed from environment — using Max subscription CLI (no billing)", _key
            )

    # Override env based on args
    if args.live:
        os.environ["LIVE_TRADING"] = "true"
        os.environ["BINANCE_PAPER"] = "false"
    else:
        os.environ.setdefault("LIVE_TRADING", "false")
        os.environ.setdefault("BINANCE_PAPER", "true")

    if args.daily:
        os.environ["BLEND_MODE"] = "daily"

    # Demo trading: required for futures testnet (sandbox deprecated)
    if args.demo or (args.futures and not args.live):
        os.environ["BINANCE_USE_DEMO"] = "true"

    _setup_logging(LOG_LEVEL)
    logger = logging.getLogger("main")

    market = "future" if args.futures else "spot"
    is_demo = os.environ.get("BINANCE_USE_DEMO", "").lower() in ("1", "true")
    mode = "LIVE" if args.live else ("DEMO" if is_demo else "TESTNET")
    blend = "daily" if args.daily else "swing (4h)"
    logger.info("=" * 60)
    logger.info("Binance Crypto Trader - %s %s mode", mode, market.upper())
    logger.info("Config: %s, Leverage: %dx", blend, args.leverage)
    if is_demo and not args.live:
        logger.info("Demo trading: API keys from https://www.binance.com/en/demo-trading")
    logger.info("=" * 60)

    if args.live:
        logger.warning("LIVE TRADING ENABLED - real money at risk!")
    if args.leverage > 1 and not args.futures:
        logger.warning("Leverage > 1x requires --futures flag, ignoring leverage")
        args.leverage = 1

    # Connect broker
    broker = BinanceBroker(market=market)
    result = broker.connect()
    if not broker.is_connected:
        logger.error("Failed to connect to Binance: %s", result)
        sys.exit(1)

    usdt_bal = result.get("usdt_balance", 0)
    logger.info("Connected. USDT: $%.2f, Mode: %s, Market: %s",
                usdt_bal, result.get("mode", "?"), market)

    # Build runner (pass initial balance from connect() to avoid extra REST call)
    runner = CryptoRunner(broker, leverage=args.leverage, initial_balance=usdt_bal)

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
        logger.exception("FATAL: Unhandled exception in runner")
    finally:
        loop.close()
        logger.info("Bye.")


if __name__ == "__main__":
    main()
