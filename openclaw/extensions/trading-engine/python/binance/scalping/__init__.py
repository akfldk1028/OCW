"""Scalping strategy package — signal-driven short-term trading.

Usage:
    python -m binance.scalping.scalp_main --testnet --futures --leverage 3
"""

from .scalp_config import (
    SCALP_TICKERS,
    SCALP_RISK_CONFIG,
    SCALP_SIGNAL_CONFIG,
    SCALP_SIGNAL_WEIGHTS,
)
from .scalp_signal import ScalpSignalEngine, EnsembleResult, SignalResult
from .scalp_runner import ScalpRunner

__all__ = [
    "SCALP_TICKERS",
    "SCALP_RISK_CONFIG",
    "SCALP_SIGNAL_CONFIG",
    "SCALP_SIGNAL_WEIGHTS",
    "ScalpSignalEngine",
    "EnsembleResult",
    "SignalResult",
    "ScalpRunner",
]
