"""Crypto-only configuration. Imports validated params from parent config."""
import os
import sys
from pathlib import Path

# Add parent dir to sys.path so we can import from parent modules
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from config import REGIME_BLEND_CONFIG, SWING_BLEND_CONFIG, CRYPTO_RISK_CONFIG  # noqa: E402

# Select active config: SWING (4h bars) or DAILY (1d bars)
_MODE = os.environ.get("BLEND_MODE", "swing").lower()
ACTIVE_BLEND_CONFIG = SWING_BLEND_CONFIG if _MODE == "swing" else REGIME_BLEND_CONFIG

# Paths (TRADING_LOGS_DIR env var overrides default)
BINANCE_DIR = Path(__file__).resolve().parent
DATA_DIR = BINANCE_DIR / "data"
LOGS_DIR = Path(os.environ.get("TRADING_LOGS_DIR", str(BINANCE_DIR / "logs")))
MODELS_DIR = BINANCE_DIR / "models"

for d in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Event-driven config (replaces fixed cron intervals)
EVENT_CONFIG = {
    "significant_move_pct": 0.015,      # BTC >1.5% move triggers re-evaluation
    "kline_intervals": ["15m", "1h", "4h"],  # multi-TF WS subscription
    "primary_interval": "15m",          # gate trigger interval (most frequent)
    "derivatives_poll_base": 300,       # 5 min base polling for funding/OI
    "derivatives_poll_fast": 60,        # 1 min accelerated when extreme
    "funding_extreme_threshold": 0.0005,  # 0.05%/8h funding rate
    "oi_spike_threshold": 0.10,         # 10% OI change
    "min_decision_gap": 300,            # 5 min cooldown between decisions (was 2min, too frequent)
    # Adaptive gate (Phase 1: agent-autonomous scheduling)
    "gate": {
        "zscore_threshold": 2.5,        # restored (2.0 was too sensitive — 81 calls/90min)
        "zscore_window": 50,
        "max_check_seconds": 3600,      # 1h fallback ceiling (was 4h)
    },
}

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
