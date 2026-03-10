"""Dashboard API for the crypto trading engine.

Runs as a 6th async task inside CryptoRunner.
Reads live state from runner + log files on disk.

Endpoints:
    GET  /api/status        — portfolio, positions, regime, gate state
    GET  /api/decisions      — Claude decision history (JSONL)
    GET  /api/trades         — trade history (CSV)
    GET  /api/ts-weights     — Thompson Sampling weights
    GET  /api/price-history  — OHLCV candle data from ohlcv_store
    GET  /api/equity-curve   — portfolio value over time
    GET  /api/health         — simple health check
    WS   /ws/live            — real-time state push (10s interval)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import time
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from runner import CryptoRunner

logger = logging.getLogger(__name__)

app = FastAPI(title="OCW Trading Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Set by start_api_server() before uvicorn starts
_runner: Optional["CryptoRunner"] = None


def _logs_dir() -> Path:
    from crypto_config import LOGS_DIR
    return LOGS_DIR


def _models_dir() -> Path:
    from crypto_config import MODELS_DIR
    return MODELS_DIR


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.post("/api/force-wake")
async def force_wake():
    """Force gate to wake immediately for a decision cycle."""
    r = _runner
    if r is None:
        return {"error": "runner not ready"}
    gate = getattr(r, 'adaptive_gate', None)
    if gate:
        gate._next_check_at = time.time()
        return {"status": "ok", "message": "gate timer reset to now"}
    return {"error": "gate not found"}


@app.get("/api/status")
async def status():
    """Portfolio value, positions, regime, gate state."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    # Positions
    positions = {}
    for tic, entry_px in r._entry_prices.items():
        tracked = r.position_tracker.get_position(tic)
        if tracked and tracked.qty > 0:
            current_px = r._last_tick_prices.get(tic, entry_px)
            side = getattr(tracked, "side", "long")
            if entry_px > 0:
                if side == "short":
                    pnl_pct = (entry_px - current_px) / entry_px
                else:
                    pnl_pct = (current_px - entry_px) / entry_px
            else:
                pnl_pct = 0
            positions[tic] = {
                "entry_price": entry_px,
                "current_price": current_px,
                "qty": tracked.qty,
                "side": side,
                "pnl_pct": round(pnl_pct, 6),
                "held_hours": round(tracked.held_hours, 1),
                "trailing_high": tracked.trailing_high,
                "trailing_low": getattr(tracked, "trailing_low", 0.0),
            }

    # Regime
    crypto_regime = "unknown"
    try:
        result = await asyncio.to_thread(r.crypto_regime_detector.detect)
        crypto_regime = result.get("regime_label", "unknown")
    except Exception:
        pass
    macro_regime = r.macro_detector.regime.value if hasattr(r, "macro_detector") else "unknown"

    # Gate
    gate = r.adaptive_gate
    gate_info = {
        "next_check_at": gate._next_check_at,
        "seconds_until_next": max(0, gate._next_check_at - time.time()),
        "wake_conditions": [str(w) for w in getattr(gate, "_wake_conditions", [])],
    }

    # Fear & Greed (cached)
    fg_idx, fg_label = r._fg_cache[1], r._fg_cache[2]

    # Today PnL from trades.csv (quick scan)
    today_pnl_pct = 0.0
    today_trades = 0
    try:
        from datetime import datetime, timezone
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        csv_path = _logs_dir() / "trades.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("action") not in ("SELL", "COVER"):
                        continue
                    ts = row.get("timestamp", "")
                    if ts[:10] == today_str:
                        try:
                            today_pnl_pct += float(row.get("pnl_pct", 0))
                            today_trades += 1
                        except (ValueError, TypeError):
                            pass
    except Exception:
        pass

    return {
        "portfolio_value": round(r._estimate_portfolio_value(), 2),
        "realized_pnl_usd": round(getattr(r, "_realized_pnl_usd", 0.0), 2),
        "today_pnl_pct": round(today_pnl_pct * 100, 3),
        "today_trades": today_trades,
        "positions": positions,
        "position_count": len(positions),
        "crypto_regime": crypto_regime,
        "macro_regime": macro_regime,
        "combined_regime": getattr(r, '_last_combined_regime', f"{crypto_regime}_{macro_regime}"),
        "fear_greed_index": fg_idx,
        "fear_greed_label": fg_label,
        "gate": gate_info,
        "agent_memory": r._agent_memory[:200],
        "claude_available": r.claude_agent.is_available if hasattr(r, "claude_agent") else False,
        "uptime_seconds": round(time.time() - getattr(r, "_start_time", time.time()), 0),
        "tick_prices": {k: v for k, v in r._last_tick_prices.items() if not k.startswith("_")},
        "timestamp": time.time(),
    }


@app.get("/api/decisions")
async def decisions(limit: int = Query(50, ge=1, le=500)):
    """Recent Claude decision history from decisions.jsonl."""
    log_path = _logs_dir() / "decisions.jsonl"
    if not log_path.exists():
        return {"decisions": [], "total": 0}

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]

    # Return most recent N
    recent = lines[-limit:]
    recent.reverse()

    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {"decisions": entries, "total": len(lines)}


@app.get("/api/trades")
async def trades(limit: int = Query(100, ge=1, le=1000)):
    """Trade history from trades.csv."""
    csv_path = _logs_dir() / "trades.csv"
    if not csv_path.exists():
        return {"trades": [], "total": 0}

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k in ("price", "qty", "value_usd", "pnl_pct", "held_hours", "confidence"):
                if k in row and row[k]:
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)

    # Pair BUY→SELL and SHORT→COVER: attach entry_price to exit rows
    last_buy_price: dict[str, float] = {}
    last_short_price: dict[str, float] = {}
    for row in rows:
        ticker = row.get("ticker", "")
        action = row.get("action", "")
        price = row.get("price", 0)
        if isinstance(price, str):
            try:
                price = float(price)
            except ValueError:
                price = 0
        if action == "BUY":
            last_buy_price[ticker] = price
        elif action == "SELL" and ticker in last_buy_price:
            row["entry_price"] = last_buy_price.pop(ticker)
        elif action == "SHORT":
            last_short_price[ticker] = price
        elif action == "COVER" and ticker in last_short_price:
            row["entry_price"] = last_short_price.pop(ticker)

    # Return most recent N
    recent = rows[-limit:]
    recent.reverse()

    # Summary stats — count SELL and COVER trades (completed round-trips)
    exits = [r for r in rows if r.get("action") in ("SELL", "COVER")]
    sell_pnls = [r.get("pnl_pct", 0) for r in exits if isinstance(r.get("pnl_pct"), (int, float))]
    total_pnl = sum(sell_pnls)
    win_count = sum(1 for p in sell_pnls if p > 0)
    loss_count = sum(1 for p in sell_pnls if p < 0)
    completed = win_count + loss_count

    return {
        "trades": recent,
        "total": len(rows),
        "summary": {
            "total_trades": len(exits),
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_count / max(completed, 1), 4),
            "cumulative_pnl_pct": round(total_pnl * 100, 3),
        },
    }


@app.get("/api/ts-weights")
async def ts_weights():
    """Thompson Sampling weights and agent status."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    status = r.online_learner.get_status()

    # Current regime
    crypto_regime = "unknown"
    try:
        result = await asyncio.to_thread(r.crypto_regime_detector.detect)
        crypto_regime = result.get("regime_label", "unknown")
    except Exception:
        pass
    macro_regime = r.macro_detector.regime.value if hasattr(r, "macro_detector") else "unknown"
    combined = f"{crypto_regime}_{macro_regime}"

    return {
        "version": status.get("version", 1),
        "current_regime": combined,
        "mean_weights": r.online_learner.get_mean_weights(regime=combined),
        "signal_reliability": r.online_learner.get_signal_reliability(regime=combined),
        "group_weights": r.online_learner.get_group_weights(regime=combined),
        "meta_params": r.online_learner.get_meta_param_means(regime=combined),
        "meta_params_global": status.get("meta_param_means_global", {}),
        "meta_param_betas": r.online_learner.get_meta_param_betas(regime=combined),
        "meta_param_betas_global": r.online_learner.get_meta_param_betas_global(),
        "group_summary": status.get("group_summary", {}),
        "regime_info": status.get("regime_info", {}),
        "num_groups": status.get("num_groups", 0),
        "num_signals": status.get("num_signals", 0),
        "total_trades": status.get("total_trades", 0),
        "has_enough_data": status.get("has_enough_data", False),
        "cumulative_pnl_pct": status.get("cumulative_pnl_pct", 0),
        "recent_trades": status.get("recent_trades", [])[:10],
    }


@app.get("/api/price-history")
async def price_history(
    ticker: str = Query("BTC/USDT"),
    timeframe: str = Query("1h", description="Candle interval: 5m, 15m, 1h"),
    limit: int = Query(200, ge=1, le=500),
):
    """OHLCV candle data from the in-memory ohlcv_store."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    if timeframe not in ("5m", "15m", "1h"):
        timeframe = "1h"

    bars = r.ohlcv_store.get_bars(ticker, timeframe, limit)
    if not bars:
        return {"ticker": ticker, "timeframe": timeframe, "candles": []}

    candles = []
    for ts_ms, o, h, l, c, v in bars:
        candles.append({
            "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts_ms / 1000)),
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
            "volume": round(float(v), 2),
        })

    return {"ticker": ticker, "timeframe": timeframe, "candles": candles}


@app.get("/api/equity-curve")
async def equity_curve():
    """Portfolio value over time extracted from decisions.jsonl."""
    log_path = _logs_dir() / "decisions.jsonl"
    points = []

    # Add initial balance point
    r = _runner
    if r is not None:
        start_time = getattr(r, "_start_time", time.time())
        initial_bal = getattr(r, "_initial_balance", 0)
        if initial_bal > 0:
            points.append({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start_time)),
                "value": round(initial_bal, 2),
            })

    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if "portfolio_value" in entry and "timestamp" in entry:
                    points.append({
                        "time": entry["timestamp"],
                        "value": round(float(entry["portfolio_value"]), 2),
                    })
            except (json.JSONDecodeError, ValueError):
                continue

    # Add current value
    if r is not None:
        points.append({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "value": round(r._estimate_portfolio_value(), 2),
        })

    return {"points": points}


@app.get("/api/daily-pnl")
async def daily_pnl():
    """Daily PnL breakdown — how much profit/loss per day."""
    csv_path = _logs_dir() / "trades.csv"
    if not csv_path.exists():
        return {"days": [], "summary": {}}

    from collections import defaultdict
    from datetime import datetime

    daily: dict = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0, "pnl_pct": 0.0,
        "pnl_usd": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    })

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    for row in rows:
        if row.get("action") not in ("SELL", "COVER"):
            continue
        try:
            pnl = float(row.get("pnl_pct", 0))
            value = float(row.get("value_usd", 0))
            ts = row.get("timestamp", "")
            date = ts[:10] if len(ts) >= 10 else "unknown"
            held = float(row.get("held_hours", 0))
        except (ValueError, TypeError):
            continue

        d = daily[date]
        d["trades"] += 1
        d["pnl_pct"] += pnl
        entry_value = value / (1.0 + pnl) if abs(1.0 + pnl) > 1e-8 else value
        d["pnl_usd"] += pnl * entry_value
        if pnl > 0:
            d["wins"] += 1
        elif pnl < 0:
            d["losses"] += 1
        d["best_trade"] = max(d["best_trade"], pnl)
        d["worst_trade"] = min(d["worst_trade"], pnl)

    # Fill in missing dates up to today so dashboard always shows continuous days
    from datetime import date as date_cls, timedelta
    today_str = date_cls.today().isoformat()
    if daily:
        first_date = min(daily.keys())
        cur = date_cls.fromisoformat(first_date)
        end = date_cls.today()
        while cur <= end:
            ds = cur.isoformat()
            if ds not in daily:
                daily[ds]  # creates default entry via defaultdict
            cur += timedelta(days=1)

    # Build sorted list
    days = []
    total_pnl_pct = 0.0
    total_pnl_usd = 0.0
    for date in sorted(daily.keys()):
        d = daily[date]
        total_pnl_pct += d["pnl_pct"]
        total_pnl_usd += d["pnl_usd"]
        days.append({
            "date": date,
            "trades": d["trades"],
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate": round(d["wins"] / max(d["trades"], 1), 3),
            "pnl_pct": round(d["pnl_pct"] * 100, 3),
            "pnl_usd": round(d["pnl_usd"], 2),
            "best_trade_pct": round(d["best_trade"] * 100, 3),
            "worst_trade_pct": round(d["worst_trade"] * 100, 3),
            "cumulative_pnl_pct": round(total_pnl_pct * 100, 3),
        })

    # Current portfolio value
    r = _runner
    current_value = round(r._estimate_portfolio_value(), 2) if r else 0

    return {
        "days": days,
        "summary": {
            "total_days": len(days),
            "total_trades": sum(d["trades"] for d in daily.values()),
            "total_pnl_pct": round(total_pnl_pct * 100, 3),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "current_portfolio_value": current_value,
            "initial_balance": round(getattr(r, "_initial_balance", 0), 2) if r else 0,
        },
    }


# ------------------------------------------------------------------
# Tool endpoints — for Claude MCP tool calls
# ------------------------------------------------------------------

def _ta_helpers():
    """Lazy-import TA helpers from agent_tools (pure Python, no SDK dependency)."""
    from core.agent_tools import (
        _calc_rsi, _calc_ema, _calc_ema_series, _calc_stoch_rsi,
        _calc_macd, _calc_bollinger, _calc_atr, _calc_vwap,
    )
    return _calc_rsi, _calc_ema, _calc_ema_series, _calc_stoch_rsi, _calc_macd, _calc_bollinger, _calc_atr, _calc_vwap


def _get_bars(ticker: str, interval: str, n: int = 200) -> List[tuple]:
    """Get OHLCV bars from runner's ohlcv_store."""
    if _runner is None:
        return []
    return _runner.ohlcv_store.get_bars(ticker, interval, n)


@app.get("/api/tools/ta")
async def tools_ta(
    ticker: str = Query("BTC/USDT:USDT"),
    interval: str = Query("5m", description="Candle interval: 5m, 15m, 1h"),
    indicators: str = Query("all", description="Comma-separated: rsi,macd,ema_cross,bollinger,stoch_rsi,atr,vwap,all"),
    rsi_period: int = Query(7, ge=2, le=50),
    ema_fast: int = Query(9, ge=2, le=100),
    ema_slow: int = Query(21, ge=2, le=200),
):
    """Technical analysis — run TA indicators with custom params on any ticker/timeframe."""
    if interval not in ("5m", "15m", "1h"):
        interval = "5m"

    bars = _get_bars(ticker, interval, 200)
    if not bars:
        return JSONResponse({"error": f"No OHLCV data for {ticker}/{interval}"}, status_code=404)

    closes = [b[4] for b in bars if b[4] > 0]
    if len(closes) < 5:
        return JSONResponse({"error": f"Insufficient data: {len(closes)} bars"}, status_code=404)

    calc_rsi, calc_ema, _, calc_stoch_rsi, calc_macd, calc_bollinger, calc_atr, calc_vwap = _ta_helpers()

    ind_list = [s.strip() for s in indicators.split(",")]
    do_all = "all" in ind_list

    result: Dict[str, Any] = {"ticker": ticker, "interval": interval, "bar_count": len(bars)}

    if do_all or "rsi" in ind_list:
        result["rsi"] = round(calc_rsi(closes, rsi_period), 2)
        result["rsi_period"] = rsi_period

    if do_all or "stoch_rsi" in ind_list:
        result["stoch_rsi"] = round(calc_stoch_rsi(closes), 2)

    if do_all or "macd" in ind_list:
        result["macd"] = calc_macd(closes)

    if do_all or "ema_cross" in ind_list:
        ema_f = calc_ema(closes, ema_fast)
        ema_s = calc_ema(closes, ema_slow)
        cross = "bullish" if ema_f > ema_s else "bearish"
        gap_pct = (ema_f - ema_s) / ema_s if ema_s > 0 else 0
        result["ema_cross"] = {
            "status": cross,
            f"ema_{ema_fast}": round(ema_f, 2),
            f"ema_{ema_slow}": round(ema_s, 2),
            "gap_pct": round(gap_pct, 6),
        }

    if do_all or "bollinger" in ind_list:
        result["bollinger"] = calc_bollinger(closes)

    if do_all or "atr" in ind_list:
        atr = calc_atr(bars)
        atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
        result["atr"] = round(atr, 2)
        result["atr_pct"] = round(atr_pct, 6)

    if do_all or "vwap" in ind_list:
        result["vwap"] = calc_vwap(bars, n=50)

    result["last_close"] = round(closes[-1], 2)
    result["last_5_closes"] = [round(c, 2) for c in closes[-5:]]
    return result


@app.get("/api/tools/trades")
async def tools_trades(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    regime: Optional[str] = Query(None, description="Filter by regime"),
    limit: int = Query(20, ge=1, le=50),
    winners_only: bool = Query(False),
    losers_only: bool = Query(False),
    side: Optional[str] = Query(None, description="Filter by side: long or short"),
):
    """Search past trades — filter by ticker, regime, side, winners/losers."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    status = r.online_learner.get_status()
    all_trades = status.get("recent_trades", [])

    filtered = all_trades
    if ticker:
        filtered = [t for t in filtered if ticker.lower() in t.get("ticker", "").lower()]
    if regime:
        filtered = [t for t in filtered if regime in t.get("regime", "")]
    if winners_only:
        filtered = [t for t in filtered if t.get("pnl_pct", 0) > 0]
    if losers_only:
        filtered = [t for t in filtered if t.get("pnl_pct", 0) < 0]
    if side:
        filtered = [t for t in filtered if t.get("position_side", "long") == side.lower()]

    filtered = filtered[-limit:]

    pnls = [t.get("pnl_pct", 0) for t in filtered]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)

    return {
        "trades": filtered,
        "count": len(filtered),
        "summary": {
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(len(pnls), 1), 3),
            "total_pnl_pct": round(sum(pnls), 4),
        },
    }


@app.get("/api/tools/volume")
async def tools_volume(
    ticker: str = Query("BTC/USDT:USDT"),
    interval: str = Query("5m", description="Candle interval: 5m, 15m, 1h"),
    lookback: int = Query(50, ge=10, le=200),
):
    """Volume analysis — relative volume, buy/sell pressure, divergence, volume profile."""
    if interval not in ("5m", "15m", "1h"):
        interval = "5m"

    bars = _get_bars(ticker, interval, lookback)
    if len(bars) < 10:
        return JSONResponse({"error": f"Insufficient data: {len(bars)} bars"}, status_code=404)

    volumes = [b[5] for b in bars]
    closes = [b[4] for b in bars]

    # Relative volume (current vs 20-bar average)
    avg_vol_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
    rel_vol = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Buy/sell pressure (green vs red bar volume)
    green_vol = sum(b[5] for b in bars[-20:] if b[4] >= b[1])
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
    price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 and closes[-5] > 0 else 0
    vol_change = (volumes[-1] - volumes[-5]) / volumes[-5] if len(volumes) >= 5 and volumes[-5] > 0 else 0
    if price_change > 0 and vol_change < -0.2:
        divergence = "bearish_divergence"
    elif price_change < 0 and vol_change > 0.2:
        divergence = "possible_capitulation"
    elif price_change > 0 and vol_change > 0.2:
        divergence = "bullish_confirmation"
    else:
        divergence = "neutral"

    # Volume profile (top price levels by volume)
    price_vol: Dict[float, float] = {}
    for bar in bars:
        mid = (bar[2] + bar[3]) / 2
        bucket = round(mid, -int(math.log10(mid)) + 2) if mid > 0 else 0
        price_vol[bucket] = price_vol.get(bucket, 0) + bar[5]
    top_levels = sorted(price_vol.items(), key=lambda x: -x[1])[:5]

    return {
        "ticker": ticker,
        "interval": interval,
        "current_volume": round(volumes[-1], 2),
        "avg_volume_20": round(avg_vol_20, 2),
        "relative_volume": round(rel_vol, 3),
        "buy_pressure": round(buy_pressure, 3),
        "volume_trend_ratio": round(vol_trend_ratio, 3),
        "divergence": divergence,
        "high_volume_levels": [{"price": round(p, 2), "volume": round(v, 2)} for p, v in top_levels],
    }


# ------------------------------------------------------------------
# WebSocket — real-time push
# ------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Push status every 10 seconds."""
    await websocket.accept()
    try:
        while True:
            if _runner is not None:
                data = await status()
                await websocket.send_json(data)
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("[ws] Error: %s", e)


# ------------------------------------------------------------------
# MCP-specific endpoints (for Claude.ai interaction)
# ------------------------------------------------------------------

@app.get("/api/mcp/snapshot")
async def mcp_snapshot():
    """Full MarketSnapshot for MCP clients (Claude.ai)."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    data = await status()
    ts_data = await ts_weights()
    return {
        **data,
        "ts_weights": ts_data,
    }


@app.get("/api/mcp/explain-trade/{trade_idx}")
async def mcp_explain_trade(trade_idx: int):
    """Explain a specific trade by index (0 = most recent)."""
    log_path = _logs_dir() / "decisions.jsonl"
    if not log_path.exists():
        return JSONResponse({"error": "no decisions log"}, status_code=404)

    lines = [l for l in log_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    if trade_idx >= len(lines):
        return JSONResponse({"error": f"index {trade_idx} out of range (max {len(lines)-1})"}, status_code=404)

    # Reverse index: 0 = most recent
    entry = json.loads(lines[-(trade_idx + 1)])
    return {"trade_index": trade_idx, "decision": entry}


# ------------------------------------------------------------------
# MCP mount (fastapi-mcp)
# ------------------------------------------------------------------

try:
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(app, name="ocw-trading-engine")
    mcp.mount()
    logger.info("[api] MCP server mounted — all endpoints exposed as MCP tools")
except ImportError:
    logger.info("[api] fastapi-mcp not installed, MCP disabled")


# ------------------------------------------------------------------
# Server startup
# ------------------------------------------------------------------

async def start_api_server(runner: "CryptoRunner", host: str = "0.0.0.0", port: int = 8080):
    """Start FastAPI as an async task inside the runner's event loop."""
    global _runner
    _runner = runner
    runner._start_time = time.time()

    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    logger.info("[api] Dashboard API starting on %s:%d", host, port)
    await server.serve()
