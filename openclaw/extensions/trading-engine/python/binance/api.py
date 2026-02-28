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
            pnl_pct = (current_px / entry_px - 1) if entry_px > 0 else 0
            positions[tic] = {
                "entry_price": entry_px,
                "current_price": current_px,
                "qty": tracked.qty,
                "pnl_pct": round(pnl_pct, 6),
                "held_hours": round(tracked.held_hours, 1),
                "trailing_high": tracked.trailing_high,
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
                    if row.get("action") != "SELL":
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
        "combined_regime": f"{crypto_regime}_{macro_regime}",
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

    # Pair BUY→SELL: attach entry_price to SELL rows
    last_buy_price: dict[str, float] = {}
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

    # Return most recent N
    recent = rows[-limit:]
    recent.reverse()

    # Summary stats — only count SELL trades (completed round-trips)
    sells = [r for r in rows if r.get("action") == "SELL"]
    sell_pnls = [r.get("pnl_pct", 0) for r in sells if isinstance(r.get("pnl_pct"), (int, float))]
    total_pnl = sum(sell_pnls)
    win_count = sum(1 for p in sell_pnls if p > 0)
    loss_count = sum(1 for p in sell_pnls if p < 0)
    completed = win_count + loss_count

    return {
        "trades": recent,
        "total": len(rows),
        "summary": {
            "total_trades": len(sells),
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
        "group_weights": r.online_learner.get_group_weights(regime=combined),
        "meta_params": r.online_learner.get_meta_param_means(regime=combined),
        "meta_params_global": status.get("meta_param_means_global", {}),
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
    limit: int = Query(200, ge=1, le=500),
):
    """OHLCV candle data from the in-memory ohlcv_store."""
    r = _runner
    if r is None:
        return JSONResponse({"error": "runner not initialized"}, status_code=503)

    store = r.ohlcv_store
    df = store.get(ticker, "1h")
    if df is None or df.empty:
        return {"ticker": ticker, "timeframe": "1h", "candles": []}

    df = df.tail(limit)
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time": str(ts),
            "open": round(float(row.get("open", 0)), 2),
            "high": round(float(row.get("high", 0)), 2),
            "low": round(float(row.get("low", 0)), 2),
            "close": round(float(row.get("close", 0)), 2),
            "volume": round(float(row.get("volume", 0)), 2),
        })

    return {"ticker": ticker, "timeframe": "1h", "candles": candles}


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
        if row.get("action") != "SELL":
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
