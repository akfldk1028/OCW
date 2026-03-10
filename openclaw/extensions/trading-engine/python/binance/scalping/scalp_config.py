"""Scalping strategy configuration.

Signal-driven (rule-based) with Claude confirmation for medium-confidence signals.
Targets 1% daily profit via 0.3-1% TP trades on 1m/3m/5m timeframes.
"""

import os
import sys
from pathlib import Path

# Ensure parent dirs are importable
_SCALPING_DIR = Path(__file__).resolve().parent
_BINANCE_DIR = _SCALPING_DIR.parent
_ENGINE_DIR = _BINANCE_DIR.parent
for _p in (_BINANCE_DIR, _ENGINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SCALP_TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# ---------------------------------------------------------------------------
# Event-driven config (WS kline intervals)
# ---------------------------------------------------------------------------
SCALP_EVENT_CONFIG = {
    "kline_intervals": ["1m", "3m", "5m"],
    "primary_interval": "1m",
    "significant_move_pct": 0.005,      # 0.5% move triggers re-evaluation
    "min_decision_gap": 10,             # 10s cooldown between decisions
}

# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
SCALP_RISK_CONFIG = {
    "take_profit_pct": 0.005,           # 0.5% base TP (ATR-adjusted: 0.3-1%)
    "stop_loss_pct": -0.007,            # -0.7% base SL (ATR-adjusted: 0.5-1%)
    "max_hold_minutes": 30,             # 30 min max hold
    "max_simultaneous": 3,              # 3 concurrent positions
    "leverage": 3,                      # 3x leverage
    "position_size_pct": 0.10,          # 10% of portfolio per trade
    "daily_loss_limit_pct": -0.02,      # -2% daily circuit breaker
    "daily_target_pct": 0.01,           # +1% daily target (stop trading)
    "trailing_activate_pct": 0.003,     # Activate trailing at +0.3%
    "trailing_width_pct": 0.002,        # Trail by 0.2%
}

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
SCALP_SIGNAL_CONFIG = {
    # RSI divergence
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "rsi_divergence_lookback": 10,      # bars to check for divergence

    # EMA cross
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_volume_confirm_ratio": 1.3,    # volume must be 1.3x avg

    # Bollinger Band squeeze
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze_threshold": 0.001,      # bandwidth < 0.1% = squeeze
    "bb_volume_spike_ratio": 2.0,       # 2x avg volume for breakout

    # Order flow (CVD / taker delta)
    "cvd_extreme_ratio": 1.15,          # taker buy/sell > 1.15 = extreme

    # Funding rate bias
    "funding_long_threshold": -0.0003,  # negative = long bias
    "funding_short_threshold": 0.0003,  # positive = short bias
}

# ---------------------------------------------------------------------------
# Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCALP_SIGNAL_WEIGHTS = {
    "rsi_divergence": 0.25,
    "ema_cross": 0.20,
    "bb_squeeze": 0.20,
    "order_flow": 0.20,
    "funding_bias": 0.15,
}

# ---------------------------------------------------------------------------
# Execution thresholds
# ---------------------------------------------------------------------------
SCALP_EXECUTION_CONFIG = {
    "instant_threshold": 0.85,          # |score| >= 0.85 -> instant exec
    "claude_threshold": 0.70,           # 0.70 <= |score| < 0.85 -> Claude confirm
    "min_agreeing_signals": 2,          # need 2+ signals agreeing for Claude confirm
    "ignore_threshold": 0.70,           # |score| < 0.70 -> ignore
}

# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------
SCALP_REGIME_FILTER = {
    "high_volatility": "skip",          # Don't scalp in high vol
    "extreme_volatility": "skip",       # Don't scalp in extreme vol
    "medium_volatility": 0.5,           # Half size in medium vol
    "low_volatility": 1.0,              # Full size in low vol
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCALP_DATA_DIR = _SCALPING_DIR / "data"
SCALP_LOGS_DIR = Path(os.environ.get("SCALP_LOGS_DIR", str(_SCALPING_DIR / "logs")))
SCALP_MODELS_DIR = _SCALPING_DIR / "models"

for _d in (SCALP_DATA_DIR, SCALP_LOGS_DIR, SCALP_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
SCALP_LOG_LEVEL = os.environ.get("SCALP_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO"))
