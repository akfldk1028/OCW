"""In-process MCP tools for Claude Agent — transforms Claude from LLM to Agent.

Claude can call these tools DURING decision-making to:
- Run technical analysis on any ticker/timeframe/period
- Inspect orderbook-like data (recent bars, volume profile)
- Query position details and trade history
- Access derivatives context on demand

Tools run in-process via claude_agent_sdk's @tool decorator + create_sdk_mcp_server().
No IPC overhead — direct access to runner state.

Usage:
    from core.agent_tools import create_trading_tools_server, set_runner
    set_runner(runner)  # inject live state once
    server = create_trading_tools_server()
    # pass to ClaudeAgentOptions(mcp_servers={"trading": server})
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("trading-engine.agent_tools")

# Module-level runner reference — set once at startup by runner.py
_runner: Any = None


def set_runner(runner: Any) -> None:
    """Inject the live CryptoRunner so tools can access state."""
    global _runner
    _runner = runner
    logger.info("[agent_tools] Runner injected — %d tools available", len(TOOL_NAMES))


def _text(content: str) -> dict:
    """Helper: wrap text in MCP tool response format."""
    return {"content": [{"type": "text", "text": content}]}


def _error(msg: str) -> dict:
    """Helper: error response."""
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "is_error": True}


# ------------------------------------------------------------------
# Technical Analysis helpers (pure Python, no external deps)
# ------------------------------------------------------------------

def _calc_rsi(closes: List[float], period: int = 14) -> float:
    """Wilder's RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_ema(values: List[float], period: int) -> float:
    """EMA — returns last value."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _calc_ema_series(values: List[float], period: int) -> List[float]:
    """Full EMA series."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _calc_stoch_rsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14) -> float:
    """Stochastic RSI."""
    if len(closes) < rsi_period + stoch_period + 1:
        return 50.0
    # Calculate RSI series
    rsi_series = []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(max(d, 0) for d in deltas[:rsi_period]) / rsi_period
    avg_loss = sum(max(-d, 0) for d in deltas[:rsi_period]) / rsi_period
    for i in range(rsi_period, len(deltas)):
        avg_gain = (avg_gain * (rsi_period - 1) + max(deltas[i], 0)) / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + max(-deltas[i], 0)) / rsi_period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi_series.append(100.0 - (100.0 / (1.0 + rs)))
    if len(rsi_series) < stoch_period:
        return 50.0
    window = rsi_series[-stoch_period:]
    low = min(window)
    high = max(window)
    if high == low:
        return 50.0
    return ((rsi_series[-1] - low) / (high - low)) * 100.0


def _calc_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """MACD line, signal line, histogram."""
    if len(closes) < slow + signal:
        return {"macd": 0, "signal": 0, "histogram": 0}
    ema_fast = _calc_ema_series(closes, fast)
    ema_slow = _calc_ema_series(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _calc_ema_series(macd_line[slow - 1:], signal)  # MACD valid from bar slow-1
    if not signal_line:
        return {"macd": macd_line[-1], "signal": 0, "histogram": macd_line[-1]}
    return {
        "macd": round(macd_line[-1], 4),
        "signal": round(signal_line[-1], 4),
        "histogram": round(macd_line[-1] - signal_line[-1], 4),
    }


def _calc_bollinger(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """Bollinger Bands."""
    if len(closes) < period:
        return {"upper": 0, "middle": 0, "lower": 0, "bandwidth": 0, "pct_b": 0.5}
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((x - mean) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = mean + std_dev * std
    lower = mean - std_dev * std
    bandwidth = (upper - lower) / mean if mean > 0 else 0
    pct_b = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return {
        "upper": round(upper, 2),
        "middle": round(mean, 2),
        "lower": round(lower, 2),
        "bandwidth": round(bandwidth, 6),
        "pct_b": round(pct_b, 4),
    }


def _calc_atr(bars: List[tuple], period: int = 14) -> float:
    """Average True Range from OHLCV bars."""
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i][2], bars[i][3], bars[i - 1][4]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _calc_vwap(bars: List[tuple], n: int = 0) -> Dict[str, float]:
    """Volume-Weighted Average Price with standard deviation bands."""
    if not bars:
        return {"vwap": 0, "upper_1sd": 0, "lower_1sd": 0, "deviation_pct": 0}
    if n > 0:
        bars = bars[-n:]
    cum_vol = 0.0
    cum_tp_vol = 0.0
    cum_tp2_vol = 0.0
    for bar in bars:
        tp = (bar[2] + bar[3] + bar[4]) / 3  # (H+L+C)/3
        vol = bar[5]
        cum_vol += vol
        cum_tp_vol += tp * vol
        cum_tp2_vol += tp * tp * vol
    if cum_vol == 0:
        return {"vwap": 0, "upper_1sd": 0, "lower_1sd": 0, "deviation_pct": 0}
    vwap = cum_tp_vol / cum_vol
    variance = max(0, cum_tp2_vol / cum_vol - vwap * vwap)
    std = math.sqrt(variance)
    current_close = bars[-1][4]
    dev_pct = (current_close - vwap) / vwap if vwap > 0 else 0
    return {
        "vwap": round(vwap, 2),
        "upper_1sd": round(vwap + std, 2),
        "lower_1sd": round(vwap - std, 2),
        "upper_2sd": round(vwap + 2 * std, 2),
        "lower_2sd": round(vwap - 2 * std, 2),
        "deviation_pct": round(dev_pct, 6),
        "current_close": round(current_close, 2),
    }


def _calc_supertrend(bars: List[tuple], period: int = 10, multiplier: float = 3.0) -> int:
    """Supertrend direction: +1 (bullish) or -1 (bearish).

    Uses ATR-based bands; price above upper band = bullish, below lower = bearish.
    bars: [(ts, open, high, low, close, volume), ...]
    """
    if len(bars) < period + 1:
        return 0
    # Calculate ATR series
    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i][2], bars[i][3], bars[i - 1][4]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    if len(trs) < period:
        return 0
    atr = sum(trs[:period]) / period
    # Track supertrend
    direction = 1  # start bullish
    prev_upper = 0.0
    prev_lower = 0.0
    for i in range(period, len(bars)):
        atr = (atr * (period - 1) + trs[i - 1]) / period
        hl2 = (bars[i][2] + bars[i][3]) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr
        # Carry forward bands
        upper = min(basic_upper, prev_upper) if prev_upper > 0 and bars[i - 1][4] <= prev_upper else basic_upper
        lower = max(basic_lower, prev_lower) if prev_lower > 0 and bars[i - 1][4] >= prev_lower else basic_lower
        # Direction flip
        if direction == 1:
            if bars[i][4] < lower:
                direction = -1
        else:
            if bars[i][4] > upper:
                direction = 1
        prev_upper = upper
        prev_lower = lower
    return direction


def _calc_mfi(bars: List[tuple], period: int = 14) -> float:
    """Money Flow Index (0-100). Uses typical price × volume.

    bars: [(ts, open, high, low, close, volume), ...]
    """
    if len(bars) < period + 1:
        return 50.0
    # Typical prices
    tps = [(b[2] + b[3] + b[4]) / 3 for b in bars]
    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(-period, 0):
        mf = tps[i] * bars[i][5]  # typical_price × volume
        if tps[i] > tps[i - 1]:
            pos_flow += mf
        elif tps[i] < tps[i - 1]:
            neg_flow += mf
    if neg_flow == 0:
        return 100.0
    ratio = pos_flow / neg_flow
    return 100.0 - (100.0 / (1.0 + ratio))


def _calc_obv_direction(closes: List[float], volumes: List[float], lookback: int = 10) -> float:
    """OBV vs price divergence over lookback bars.

    Returns: +1.0 = bullish divergence (price down, OBV up)
             -1.0 = bearish divergence (price up, OBV down)
              0.0 = aligned (no divergence)
    """
    if len(closes) < lookback + 1 or len(volumes) < lookback + 1:
        return 0.0
    # Price direction
    price_change = closes[-1] - closes[-lookback]
    # OBV direction over lookback
    obv_change = 0.0
    for i in range(-lookback, 0):
        if closes[i] > closes[i - 1]:
            obv_change += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_change -= volumes[i]
    # Divergence detection
    if price_change > 0 and obv_change < 0:
        return -1.0  # bearish divergence
    elif price_change < 0 and obv_change > 0:
        return 1.0   # bullish divergence
    return 0.0


def _get_bars(ticker: str, interval: str, n: int = 100) -> List[tuple]:
    """Get OHLCV bars from the runner's store."""
    if _runner is None:
        return []
    return _runner.ohlcv_store.get_bars(ticker, interval, n)


# ------------------------------------------------------------------
# MCP Tool definitions
# ------------------------------------------------------------------

def _define_tools() -> list:
    """Define all trading tools. Called once at server creation."""
    from claude_agent_sdk import tool

    @tool(
        "technical_analysis",
        "Run technical indicators (RSI, MACD, EMA, Bollinger, StochRSI, ATR, VWAP) on any ticker and timeframe. "
        "Use this to get precise numeric values before making trading decisions.",
        {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. 'BTC/USDT' or 'ETH/USDT'",
                },
                "interval": {
                    "type": "string",
                    "description": "Candle interval: '5m', '15m', '1h'",
                    "enum": ["5m", "15m", "1h"],
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators: 'rsi', 'macd', 'ema_cross', 'bollinger', 'stoch_rsi', 'atr', 'vwap', 'all'",
                },
                "rsi_period": {"type": "integer", "description": "RSI period (default 7 for scalping, 14 for swing)"},
                "ema_fast": {"type": "integer", "description": "Fast EMA period (default 9)"},
                "ema_slow": {"type": "integer", "description": "Slow EMA period (default 21)"},
                "bb_period": {"type": "integer", "description": "Bollinger period (default 20)"},
                "bb_std": {"type": "number", "description": "Bollinger std dev (default 2.0)"},
            },
            "required": ["ticker", "interval", "indicators"],
        },
    )
    async def technical_analysis(args: dict) -> dict:
        """Calculate technical indicators on live OHLCV data."""
        ticker = args["ticker"]
        interval = args["interval"]
        indicators = args.get("indicators", ["all"])

        bars = _get_bars(ticker, interval, 200)
        if not bars:
            return _error(f"No OHLCV data for {ticker}/{interval}")

        closes = [b[4] for b in bars if b[4] > 0]
        if len(closes) < 5:
            return _error(f"Insufficient data: {len(closes)} bars")

        result = {"ticker": ticker, "interval": interval, "bar_count": len(bars)}
        do_all = "all" in indicators

        if do_all or "rsi" in indicators:
            period = args.get("rsi_period", 7)
            result["rsi"] = round(_calc_rsi(closes, period), 2)
            result["rsi_period"] = period

        if do_all or "stoch_rsi" in indicators:
            result["stoch_rsi"] = round(_calc_stoch_rsi(closes), 2)

        if do_all or "macd" in indicators:
            result["macd"] = _calc_macd(closes)

        if do_all or "ema_cross" in indicators:
            fast_p = args.get("ema_fast", 9)
            slow_p = args.get("ema_slow", 21)
            ema_fast = _calc_ema(closes, fast_p)
            ema_slow = _calc_ema(closes, slow_p)
            cross = "bullish" if ema_fast > ema_slow else "bearish"
            gap_pct = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
            result["ema_cross"] = {
                "status": cross,
                f"ema_{fast_p}": round(ema_fast, 2),
                f"ema_{slow_p}": round(ema_slow, 2),
                "gap_pct": round(gap_pct, 6),
            }

        if do_all or "bollinger" in indicators:
            bb_period = args.get("bb_period", 20)
            bb_std = args.get("bb_std", 2.0)
            result["bollinger"] = _calc_bollinger(closes, bb_period, bb_std)

        if do_all or "atr" in indicators:
            atr = _calc_atr(bars)
            atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
            result["atr"] = round(atr, 2)
            result["atr_pct"] = round(atr_pct, 6)

        if do_all or "vwap" in indicators:
            result["vwap"] = _calc_vwap(bars, n=50)

        result["last_close"] = round(closes[-1], 2)
        result["last_5_closes"] = [round(c, 2) for c in closes[-5:]]

        return _text(json.dumps(result, indent=2))

    @tool(
        "get_ohlcv",
        "Get raw OHLCV candle data for a ticker. Returns recent bars with open/high/low/close/volume.",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "e.g. 'BTC/USDT'"},
                "interval": {"type": "string", "description": "'5m', '15m', '1h'", "enum": ["5m", "15m", "1h"]},
                "count": {"type": "integer", "description": "Number of bars (default 20, max 100)"},
            },
            "required": ["ticker", "interval"],
        },
    )
    async def get_ohlcv(args: dict) -> dict:
        """Return raw OHLCV bars."""
        ticker = args["ticker"]
        interval = args["interval"]
        count = min(args.get("count", 20), 100)

        bars = _get_bars(ticker, interval, count)
        if not bars:
            return _error(f"No data for {ticker}/{interval}")

        formatted = []
        for ts, o, h, l, c, v in bars:
            formatted.append({
                "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts / 1000)),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": round(v, 2),
                "change_pct": round((c / o - 1) * 100, 3) if o > 0 else 0,
            })

        return _text(json.dumps({
            "ticker": ticker,
            "interval": interval,
            "bars": formatted,
            "count": len(formatted),
        }, indent=2))

    @tool(
        "get_derivatives",
        "Get derivatives market data: funding rates, open interest, long/short ratio, CVD, basis spread, liquidations.",
        {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to query, or 'all' for all tracked tickers. e.g. 'BTC/USDT'",
                },
            },
            "required": ["ticker"],
        },
    )
    async def get_derivatives(args: dict) -> dict:
        """Get derivatives context from the monitor."""
        if _runner is None:
            return _error("Runner not initialized")

        deriv = _runner._derivatives_context or _runner.derivatives_monitor.get_context()
        ticker = args["ticker"]

        if ticker == "all":
            return _text(json.dumps(deriv, indent=2, default=str))

        # Filter for specific ticker
        result = {"ticker": ticker}
        for key in ("funding_rates", "open_interest", "taker_delta", "long_short_ratio", "basis_spread"):
            data = deriv.get(key, {})
            if ticker in data:
                result[key] = data[ticker]
        result["liquidations"] = deriv.get("liquidations", {})
        result["stablecoin_supply"] = deriv.get("stablecoin_supply", {})

        return _text(json.dumps(result, indent=2, default=str))

    @tool(
        "check_position",
        "Get detailed info about an open position: entry price, current PnL, held hours, trailing stop state.",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "e.g. 'BTC/USDT'. Use 'all' for all positions."},
            },
            "required": ["ticker"],
        },
    )
    async def check_position(args: dict) -> dict:
        """Get position details from tracker."""
        if _runner is None:
            return _error("Runner not initialized")

        ticker = args["ticker"]

        def _pos_info(tic: str) -> Optional[dict]:
            tracked = _runner.position_tracker.get_position(tic)
            if tracked is None or tracked.qty <= 0:
                return None
            entry_px = _runner._entry_prices.get(tic, tracked.entry_price)
            current_px = _runner._last_tick_prices.get(tic, entry_px)
            from crypto_config import ROUND_TRIP_FEE
            gross_pnl = (current_px / entry_px - 1) if entry_px > 0 else 0
            pnl_pct = gross_pnl - ROUND_TRIP_FEE  # net after fees
            peak_pnl = (tracked.trailing_high / entry_px - 1) - ROUND_TRIP_FEE if entry_px > 0 else 0
            return {
                "ticker": tic,
                "entry_price": round(entry_px, 2),
                "current_price": round(current_px, 2),
                "qty": tracked.qty,
                "value_usd": round(tracked.qty * current_px, 2),
                "pnl_pct": round(pnl_pct * 100, 3),
                "peak_pnl_pct": round(peak_pnl * 100, 3),
                "held_hours": round(tracked.held_hours, 2),
                "trailing_high": round(tracked.trailing_high, 2),
                "regime_at_entry": tracked.regime,
            }

        if ticker == "all":
            positions = []
            for tic in list(_runner._entry_prices.keys()):
                info = _pos_info(tic)
                if info:
                    positions.append(info)
            return _text(json.dumps({"positions": positions, "count": len(positions)}, indent=2))

        info = _pos_info(ticker)
        if info is None:
            return _text(json.dumps({"ticker": ticker, "status": "no position"}))
        return _text(json.dumps(info, indent=2))

    @tool(
        "search_trades",
        "Search past trade history. Filter by ticker, regime, or date range. Returns PnL, signals used, and outcomes.",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker (optional)"},
                "regime": {"type": "string", "description": "Filter by regime (optional)"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
                "winners_only": {"type": "boolean", "description": "Only show profitable trades"},
                "losers_only": {"type": "boolean", "description": "Only show losing trades"},
            },
            "required": [],
        },
    )
    async def search_trades(args: dict) -> dict:
        """Search trade history from online learner."""
        if _runner is None:
            return _error("Runner not initialized")

        status = _runner.online_learner.get_status()
        all_trades = status.get("recent_trades", [])

        # Apply filters
        filtered = all_trades
        if args.get("ticker"):
            tic = args["ticker"]
            filtered = [t for t in filtered if t.get("ticker", "").startswith(tic)]
        if args.get("regime"):
            reg = args["regime"]
            filtered = [t for t in filtered if reg in t.get("regime", "")]
        if args.get("winners_only"):
            filtered = [t for t in filtered if t.get("pnl_pct", 0) > 0]
        if args.get("losers_only"):
            filtered = [t for t in filtered if t.get("pnl_pct", 0) < 0]

        limit = min(args.get("limit", 20), 50)
        filtered = filtered[-limit:]

        # Summary stats
        pnls = [t.get("pnl_pct", 0) for t in filtered]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total_pnl = sum(pnls)

        return _text(json.dumps({
            "trades": filtered,
            "count": len(filtered),
            "summary": {
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / max(len(pnls), 1), 3),
                "total_pnl_pct": round(total_pnl, 4),
            },
        }, indent=2, default=str))

    @tool(
        "get_ts_weights",
        "Get current Thompson Sampling weights — which signal groups and individual signals are most reliable in the current regime.",
        {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "description": "Regime to query (default: current)"},
            },
            "required": [],
        },
    )
    async def get_ts_weights(args: dict) -> dict:
        """Get H-TS posteriors."""
        if _runner is None:
            return _error("Runner not initialized")

        regime = args.get("regime")
        if not regime:
            try:
                crypto = _runner.crypto_regime_detector.detect().get("regime_label", "unknown")
                macro = _runner.macro_detector.regime.value
                regime = f"{crypto}_{macro}"
            except Exception:
                regime = "unknown"

        group_w = _runner.online_learner.get_group_weights(regime=regime)
        signal_w = _runner.online_learner.get_mean_weights(regime=regime)
        meta_params = _runner.online_learner.get_meta_param_means(regime=regime)
        status = _runner.online_learner.get_status()

        # Sort by weight
        sorted_groups = sorted(group_w.items(), key=lambda x: -x[1])
        sorted_signals = sorted(signal_w.items(), key=lambda x: -x[1])

        return _text(json.dumps({
            "regime": regime,
            "total_trades": status.get("total_trades", 0),
            "group_weights": {k: round(v, 4) for k, v in sorted_groups},
            "top_signals": {k: round(v, 4) for k, v in sorted_signals[:15]},
            "bottom_signals": {k: round(v, 4) for k, v in sorted_signals[-5:]},
            "meta_params": meta_params,
        }, indent=2))

    @tool(
        "volume_analysis",
        "Analyze volume patterns: volume profile, relative volume, buy/sell pressure, volume-price divergence.",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "e.g. 'BTC/USDT'"},
                "interval": {"type": "string", "description": "'5m', '15m', '1h'", "enum": ["5m", "15m", "1h"]},
                "lookback": {"type": "integer", "description": "Number of bars to analyze (default 50)"},
            },
            "required": ["ticker", "interval"],
        },
    )
    async def volume_analysis(args: dict) -> dict:
        """Analyze volume patterns."""
        ticker = args["ticker"]
        interval = args["interval"]
        lookback = min(args.get("lookback", 50), 200)

        bars = _get_bars(ticker, interval, lookback)
        if len(bars) < 10:
            return _error(f"Insufficient data: {len(bars)} bars")

        volumes = [b[5] for b in bars]
        closes = [b[4] for b in bars]

        # Relative volume (current vs 20-bar average)
        avg_vol_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
        rel_vol = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # Buy/sell pressure estimate (green bar volume vs red bar volume)
        green_vol = sum(b[5] for b in bars[-20:] if b[4] >= b[1])  # close >= open
        red_vol = sum(b[5] for b in bars[-20:] if b[4] < b[1])
        total_vol = green_vol + red_vol
        buy_pressure = green_vol / total_vol if total_vol > 0 else 0.5

        # Volume trend (5-bar vs previous 5-bar)
        if len(volumes) >= 10:
            recent_avg = sum(volumes[-5:]) / 5
            prev_avg = sum(volumes[-10:-5]) / 5
            vol_trend_ratio = recent_avg / prev_avg if prev_avg > 0 else 1.0
        else:
            vol_trend_ratio = 1.0

        # Volume-price divergence
        # Rising price + falling volume = bearish divergence
        # Falling price + rising volume = bearish confirmation (or capitulation)
        price_change = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
        vol_change = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        if price_change > 0 and vol_change < -0.2:
            divergence = "bearish_divergence (price up, volume down)"
        elif price_change < 0 and vol_change > 0.2:
            divergence = "possible_capitulation (price down, volume up)"
        elif price_change > 0 and vol_change > 0.2:
            divergence = "bullish_confirmation (price up, volume up)"
        else:
            divergence = "neutral"

        # Volume profile (price levels with highest volume)
        price_vol = {}
        for bar in bars:
            # Bucket to nearest 0.1%
            mid = (bar[2] + bar[3]) / 2
            bucket = round(mid, -int(math.log10(mid)) + 2) if mid > 0 else 0
            price_vol[bucket] = price_vol.get(bucket, 0) + bar[5]
        top_levels = sorted(price_vol.items(), key=lambda x: -x[1])[:5]

        return _text(json.dumps({
            "ticker": ticker,
            "interval": interval,
            "current_volume": round(volumes[-1], 2),
            "avg_volume_20": round(avg_vol_20, 2),
            "relative_volume": round(rel_vol, 3),
            "buy_pressure": round(buy_pressure, 3),
            "volume_trend_ratio": round(vol_trend_ratio, 3),
            "divergence": divergence,
            "high_volume_levels": [{"price": round(p, 2), "volume": round(v, 2)} for p, v in top_levels],
        }, indent=2))

    return [
        technical_analysis,
        get_ohlcv,
        get_derivatives,
        check_position,
        search_trades,
        get_ts_weights,
        volume_analysis,
    ]


# Tool names for allowed_tools config
TOOL_NAMES = [
    "technical_analysis",
    "get_ohlcv",
    "get_derivatives",
    "check_position",
    "search_trades",
    "get_ts_weights",
    "volume_analysis",
]


def create_trading_tools_server():
    """Create the in-process MCP server with all trading tools.

    Returns:
        McpSdkServerConfig ready for ClaudeAgentOptions.mcp_servers
    """
    from claude_agent_sdk import create_sdk_mcp_server
    tools = _define_tools()
    server = create_sdk_mcp_server(
        name="trading_engine",
        version="1.0.0",
        tools=tools,
    )
    logger.info("[agent_tools] MCP server created with %d tools: %s", len(tools), TOOL_NAMES)
    return server
