"""Offline integration test: verify multi-TF flow without Binance API."""
import asyncio
import sys
import time
from pathlib import Path

_PARENT_DIR = Path(__file__).resolve().parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from core.multi_tf_aggregator import MultiTFAggregator
from core.market_listener import MarketListener
from core.event_bus import EventBus
from core.memory_store import TradingMemory
from core.claude_agent import MarketSnapshot

print("=" * 60)
print("Offline Integration Test")
print("=" * 60)

# --- Test 1: MarketListener init with multi-TF ---
print("\n[1] MarketListener init")
bus = EventBus()
ml = MarketListener(
    event_bus=bus,
    tickers=["BTC/USDT", "ETH/USDT"],
    kline_intervals=["15m", "1h", "4h"],
    primary_interval="15m",
    testnet=True,
    market="spot",
)
assert ml._kline_intervals == ["15m", "1h", "4h"], f"Expected 3 intervals, got {ml._kline_intervals}"
assert ml._primary_interval == "15m"
url = ml._build_stream_url()
assert "btcusdt@kline_15m" in url
assert "btcusdt@kline_1h" in url
assert "btcusdt@kline_4h" in url
assert "ethusdt@kline_15m" in url
stream_count = url.count("@kline_")
assert stream_count == 6, f"Expected 6 streams (2 tickers x 3 intervals), got {stream_count}"
print(f"  OK: {stream_count} streams in URL")

# --- Test 2: MarketListener backward compat ---
print("\n[2] MarketListener backward compat (single interval)")
ml_old = MarketListener(
    event_bus=bus,
    tickers=["BTC/USDT"],
    kline_interval="4h",  # old param
    testnet=True,
)
assert ml_old._kline_intervals == ["4h"]
print("  OK: legacy kline_interval param works")

# --- Test 3: MultiTFAggregator ---
print("\n[3] MultiTFAggregator")
agg = MultiTFAggregator(intervals=("15m", "1h", "4h"))

# Feed 20 bars into 15m
base = 96000
for i in range(20):
    agg.update("BTC/USDT", "15m", {
        "open": base + i * 10, "high": base + i * 10 + 50,
        "low": base + i * 10 - 30, "close": base + i * 15,
        "volume": 100 + i * 5, "timestamp": time.time() + i,
    })

# Feed 5 bars into 1h
for i in range(5):
    agg.update("BTC/USDT", "1h", {
        "open": base + i * 40, "high": base + i * 40 + 200,
        "low": base + i * 40 - 100, "close": base + i * 60,
        "volume": 500 + i * 20, "timestamp": time.time() + i * 100,
    })

summary = agg.get_summary("BTC/USDT")
assert "15m" in summary and summary["15m"].get("bar_count") == 20
assert "1h" in summary and summary["1h"].get("bar_count") == 5
assert "4h" in summary and summary["4h"].get("status") == "no data"

prompt_text = agg.format_for_prompt("BTC/USDT")
assert "15m:" in prompt_text
assert "1h:" in prompt_text
assert "RSI=" in prompt_text
print(f"  OK: 15m={summary['15m']['bar_count']} bars, 1h={summary['1h']['bar_count']} bars")
print(f"  Prompt preview: {prompt_text[:80]}...")

# RSI test
rsi_15m = summary["15m"]["rsi"]
print(f"  RSI(15m) = {rsi_15m:.1f}")
assert 50 <= rsi_15m <= 100, f"RSI should be bullish for ascending prices, got {rsi_15m}"

# --- Test 4: MarketSnapshot with multi-TF ---
print("\n[4] MarketSnapshot.to_prompt() with multi_tf_summary")
snap = MarketSnapshot(
    ticker_prices={"BTC/USDT": 96000},
    candidates=["BTC/USDT"],
    btc_price=96000,
    multi_tf_summary={"BTC/USDT": prompt_text},
    historical_insights="Regime(low_vol_goldilocks): momentum works best",
    gate_wake_reasons=["timer_expired", "z-score > 2.0"],
    agent_memory="Watching 95K support",
)
full_prompt = snap.to_prompt()
assert "## Multi-Timeframe Analysis" in full_prompt
assert "## Historical Insights" in full_prompt
assert "momentum works best" in full_prompt
assert "## Wake Reasons" in full_prompt
assert "1bar:" in full_prompt  # renamed from "4h:"
print("  OK: prompt includes Multi-TF + Historical Insights + Wake Reasons")
print(f"  Prompt length: {len(full_prompt)} chars")

# --- Test 5: TradingMemory graceful degradation ---
print("\n[5] TradingMemory (graceful degradation)")
mem = TradingMemory()
# Should not crash even without Neo4j running
mem.record_trade({"ticker": "BTC/USDT", "pnl_pct": 0.05, "regime": "test"})
insights = mem.build_historical_insights("test", ["BTC/USDT"])
# If Neo4j is running, we get data; if not, empty string
print(f"  Neo4j available: {mem._driver is not None}")
print(f"  Insights: '{insights}' (empty is OK if Neo4j not running)")
mem.close()

# --- Test 6: Event flow simulation ---
print("\n[6] Event flow: tick → gate filter (primary only)")
tick_count = {"total": 0, "primary": 0}

async def count_ticks(event):
    tick_count["total"] += 1
    if event.payload.get("interval") == "15m":
        tick_count["primary"] += 1

bus2 = EventBus()
bus2.subscribe("market.tick", count_ticks)

async def simulate():
    # Simulate ticks from all 3 intervals
    for iv in ["15m", "1h", "4h"]:
        await bus2.publish("market.tick", {
            "ticker": "BTC/USDT", "price": 96000, "interval": iv,
            "is_closed": False, "volume": 100, "timestamp": time.time(),
        })

asyncio.run(simulate())
assert tick_count["total"] == 3, f"Expected 3 total ticks, got {tick_count['total']}"
assert tick_count["primary"] == 1, f"Expected 1 primary tick, got {tick_count['primary']}"
print(f"  OK: {tick_count['total']} ticks received, {tick_count['primary']} primary (gate evaluates only primary)")

# --- Test 7: Config values ---
print("\n[7] Config verification")
from binance.crypto_config import EVENT_CONFIG, ACTIVE_BLEND_CONFIG

assert EVENT_CONFIG["kline_intervals"] == ["15m", "1h", "4h"]
assert EVENT_CONFIG["primary_interval"] == "15m"
assert EVENT_CONFIG["significant_move_pct"] == 0.015
assert EVENT_CONFIG["min_decision_gap"] == 120
assert EVENT_CONFIG["gate"]["zscore_threshold"] == 2.0
assert EVENT_CONFIG["gate"]["max_check_seconds"] == 3600
print("  EVENT_CONFIG: OK")

assert ACTIVE_BLEND_CONFIG["ohlcv_timeframe"] == "1h"
assert ACTIVE_BLEND_CONFIG["position_pct"] == 0.20
assert ACTIVE_BLEND_CONFIG["trail_pct"] == 0.06
assert ACTIVE_BLEND_CONFIG["trail_activation_pct"] == 0.04
print("  SWING_BLEND_CONFIG: OK")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED")
print("=" * 60)
