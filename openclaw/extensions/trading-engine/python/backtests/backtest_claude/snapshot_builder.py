"""Build MarketSnapshot from historical data for Claude simulation.

Takes a point-in-time from cached OHLCV data and constructs the same
MarketSnapshot that the live runner would build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util
import pandas as pd

# Direct module import to avoid core/__init__.py pulling in fastapi/websockets
_spec = importlib.util.spec_from_file_location(
    "claude_agent_mod", _ROOT / "core" / "claude_agent.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["claude_agent_mod"] = _mod
_spec.loader.exec_module(_mod)
MarketSnapshot = _mod.MarketSnapshot


def build_snapshot_at(
    bar_time: pd.Timestamp,
    ticker_data: Dict[str, pd.DataFrame],
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    portfolio_value: float,
    trigger: str = "candle_close",
    wake_reasons: List[str] | None = None,
    agent_memory: str = "",
    ts_mean_weights: Dict[str, float] | None = None,
    ts_regime_info: str = "",
    ts_total_trades: int = 0,
    ts_cumulative_pnl_pct: float = 0.0,
    recent_trades: List[Dict] | None = None,
) -> MarketSnapshot:
    """Build a MarketSnapshot at a specific historical bar time."""

    candidates = list(ticker_data.keys())
    ticker_prices = {}
    ticker_returns_4h = {}
    ticker_returns_24h = {}
    ticker_volumes = {}

    for tic, df in ticker_data.items():
        # Get data up to bar_time
        mask = df.index <= bar_time
        available = df[mask]
        if available.empty:
            continue

        px = float(available["close"].iloc[-1])
        ticker_prices[tic] = px
        ticker_volumes[tic] = float(available["volume"].iloc[-1])

        if len(available) >= 2:
            ticker_returns_4h[tic] = px / float(available["close"].iloc[-2]) - 1

        if len(available) >= 7:
            ticker_returns_24h[tic] = px / float(available["close"].iloc[-7]) - 1

    # BTC reference
    btc_ticker = "BTC/USDT"
    btc_price = ticker_prices.get(btc_ticker, 0)
    btc_1h = 0.0  # not available in 4h data
    btc_24h = ticker_returns_24h.get(btc_ticker, 0.0)

    # Regime detection (simplified — same as backtest_swing)
    crypto_regime = _detect_regime(ticker_data.get(btc_ticker), bar_time)

    # Enrich positions with current PnL
    enriched = {}
    for tic, pos in positions.items():
        current_px = ticker_prices.get(tic, 0)
        entry_px = pos.get("entry_price", 0)
        pnl_pct = (current_px / entry_px - 1) if entry_px > 0 else 0
        enriched[tic] = {
            **pos,
            "current_price": current_px,
            "pnl_pct": pnl_pct,
        }

    return MarketSnapshot(
        ticker_prices=ticker_prices,
        ticker_returns_4h=ticker_returns_4h,
        ticker_returns_24h=ticker_returns_24h,
        ticker_volumes=ticker_volumes,
        btc_price=btc_price,
        btc_change_1h=btc_1h,
        btc_change_24h=btc_24h,
        crypto_regime=crypto_regime,
        macro_regime="unknown",
        combined_regime=f"{crypto_regime}_unknown",
        ts_mean_weights=ts_mean_weights or {},
        ts_regime_info=ts_regime_info,
        ts_total_trades=ts_total_trades,
        ts_cumulative_pnl_pct=ts_cumulative_pnl_pct,
        recent_trades=recent_trades or [],
        positions=enriched,
        portfolio_value=portfolio_value,
        cash=cash,
        candidates=candidates,
        trigger=trigger,
        gate_wake_reasons=wake_reasons or [],
        agent_memory=agent_memory,
    )


def _detect_regime(btc_df: pd.DataFrame | None, bar_time: pd.Timestamp) -> str:
    """Simple regime detection from 4h BTC data."""
    if btc_df is None:
        return "unknown"
    available = btc_df[btc_df.index <= bar_time]
    if len(available) < 72:
        return "unknown"

    close = available["close"]
    vol_w = 30
    trend_w = 72

    total_range = close.iloc[-vol_w:].max() - close.iloc[-vol_w:].min()
    net_move = abs(float(close.iloc[-1]) - float(close.iloc[-vol_w]))
    efficiency = net_move / total_range if total_range > 0 else 0
    sma = close.iloc[-trend_w:].mean()
    trend_str = abs(float(close.iloc[-1]) - sma) / sma if sma > 0 else 0

    if efficiency > 0.45 and trend_str > 0.025:
        return "trending"
    return "ranging"
