"""Configuration constants for H-TS pre-training pipeline."""

from pathlib import Path

# Tickers
TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TICKER_MAP = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT"}

# Date range (5+ years for sufficient RL convergence)
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-03-01"

# Timeframes
TIMEFRAMES = ["1h"]

# Paths
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "output"

# Strategy weights for trade generation balance
STRATEGY_WEIGHTS = {
    "mean_reversion": 0.30,
    "momentum": 0.30,
    "breakout": 0.20,
    "trend_follow": 0.20,
}

# Fees
ROUND_TRIP_FEE = 0.001  # 0.1% round-trip (0.05% per side)

# Binance API
BINANCE_SPOT_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
