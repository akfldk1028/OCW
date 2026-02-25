"""Dashboard API for the crypto trading engine.

Runs as a 6th async task inside CryptoRunner.
Reads live state from runner + log files on disk.

Endpoints:
    GET  /api/status        — portfolio, positions, regime, gate state
    GET  /api/decisions      — Claude decision history (JSONL)
    GET  /api/trades         — trade history (CSV)
    GET  /api/ts-weights     — Thompson Sampling weights
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
        crypto_regime = r.crypto_regime_detector.detect().get("regime_label", "unknown")
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

    return {
        "portfolio_value": round(r._estimate_portfolio_value(), 2),
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

    # Return most recent N
    recent = rows[-limit:]
    recent.reverse()

    # Summary stats
    pnl_values = [r.get("pnl_pct", 0) for r in rows if isinstance(r.get("pnl_pct"), (int, float))]
    total_pnl = sum(pnl_values)
    win_count = sum(1 for p in pnl_values if p > 0)
    loss_count = sum(1 for p in pnl_values if p < 0)

    return {
        "trades": recent,
        "total": len(rows),
        "summary": {
            "total_trades": len(rows),
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_count / max(len(pnl_values), 1), 4),
            "cumulative_pnl_pct": round(total_pnl, 6),
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
        crypto_regime = r.crypto_regime_detector.detect().get("regime_label", "unknown")
    except Exception:
        pass
    macro_regime = r.macro_detector.regime.value if hasattr(r, "macro_detector") else "unknown"
    combined = f"{crypto_regime}_{macro_regime}"

    return {
        "current_regime": combined,
        "mean_weights": r.online_learner.get_mean_weights(regime=combined),
        "global_agents": status.get("global_agents", {}),
        "regime_agents": status.get("regime_agents", {}),
        "total_trades": status.get("total_trades", 0),
        "has_enough_data": status.get("has_enough_data", False),
        "cumulative_pnl_pct": status.get("cumulative_pnl_pct", 0),
        "recent_trades": status.get("recent_trades", [])[:10],
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
