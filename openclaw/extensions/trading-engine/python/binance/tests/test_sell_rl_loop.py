"""Quick integration test: BUY → price drop → safety SL → position.exit → RL update.

Exercises the full SELL → RL feedback loop without needing real price movement.
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Setup paths
BINANCE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BINANCE_DIR.parent
sys.path.insert(0, str(BINANCE_DIR))
sys.path.insert(0, str(PYTHON_DIR))
os.chdir(str(BINANCE_DIR))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_sell_rl")


async def test_sell_rl_loop():
    from core.event_bus import EventBus
    from core.position_tracker import PositionTracker
    from core.online_learner import OnlineLearner

    # --- Setup ---
    event_bus = EventBus()
    learner = OnlineLearner()
    tracker = PositionTracker(event_bus=event_bus)

    # Track results
    exit_events = []
    rl_before = json.loads(json.dumps(learner.get_status()))

    # --- Wire up position.exit → RL ---
    async def on_exit(event):
        data = event.payload
        logger.info("EXIT EVENT: %s pnl=%.2f%% held=%.1fh source=%s",
                     data["ticker"], data["pnl_pct"] * 100, data["held_hours"], data["source"])
        exit_events.append(data)

        # Simulate runner's _on_exit: record trade in RL
        agent_signals = data.get("agent_signals", {"sentiment": 0.5, "market": 0.4})
        learner.record_trade(
            ticker=data["ticker"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            pnl_pct=data["pnl_pct"],
            held_hours=data["held_hours"],
            agent_signals=agent_signals,
            market_type="crypto",
            regime="state_3_unknown",
        )

    event_bus.subscribe("position.exit", on_exit)

    # --- Fake price fetcher ---
    fake_prices = {"BTC/USDT": 63000.0, "ETH/USDT": 1820.0}

    async def price_fetcher(ticker: str) -> float:
        return fake_prices.get(ticker, 0.0)

    tracker.set_price_fetcher("binance", price_fetcher)

    # --- Fake exit executor (dry run) ---
    async def exit_executor(decision: dict) -> dict:
        logger.info("EXIT EXECUTOR: %s %s @ $%.2f",
                     decision["action"], decision["ticker"], decision.get("price", 0))
        return {"status": "dry_run", "ticker": decision["ticker"]}

    tracker.set_exit_executor("binance", exit_executor)

    # === TEST 1: BUY → price drop → safety SL trigger ===
    logger.info("=" * 60)
    logger.info("TEST 1: BTC safety stop-loss trigger")
    logger.info("=" * 60)

    # Track BTC at $70,000 (fake high entry)
    tracker.track(
        ticker="BTC/USDT",
        broker="binance",
        entry_price=70000.0,
        qty=0.3,
        market_type="crypto",
    )
    logger.info("Tracked BTC/USDT: entry=$70,000, qty=0.3")

    # Current price $63,000 = -10% loss → should trigger SL
    pos = tracker.get_position("BTC/USDT")
    logger.info("Position PnL: %.2f%% (SL threshold: -10%%)",
                ((63000 / 70000) - 1) * 100)

    # Run one safety check cycle
    await tracker._safety_check()
    await asyncio.sleep(0.1)  # Let events propagate

    if exit_events:
        logger.info("PASS: Safety SL triggered! Exit event received.")
        ev = exit_events[-1]
        logger.info("  ticker=%s, entry=$%.0f, exit=$%.0f, pnl=%.2f%%",
                     ev["ticker"], ev["entry_price"], ev["exit_price"], ev["pnl_pct"] * 100)
    else:
        logger.error("FAIL: Safety SL did NOT trigger. Position still open.")
        # Check what the safety rules see
        pos = tracker.get_position("BTC/USDT")
        if pos:
            current = await price_fetcher("BTC/USDT")
            pnl = (current / pos.entry_price - 1)
            logger.error("  entry=%.0f, current=%.0f, pnl=%.2f%%", pos.entry_price, current, pnl * 100)

    # === TEST 2: Check RL updated ===
    logger.info("=" * 60)
    logger.info("TEST 2: Online Learner (RL) state after trade")
    logger.info("=" * 60)

    rl_after = learner.get_status()
    logger.info("Before: total_trades=%d, total_pnl=%.4f",
                rl_before.get("total_trades", 0), rl_before.get("total_pnl", 0))
    logger.info("After:  total_trades=%d, total_pnl=%.4f",
                rl_after.get("total_trades", 0), rl_after.get("total_pnl", 0))

    if rl_after.get("total_trades", 0) > rl_before.get("total_trades", 0):
        logger.info("PASS: RL recorded the trade!")
        # Check regime-specific data
        regime_data = rl_after.get("regime_agents", {}).get("state_3_unknown", {})
        if regime_data:
            logger.info("  Regime 'state_3_unknown': %d trades", regime_data.get("trade_count", 0))
        # Check agent weights changed
        for agent_name, agent_data in rl_after.get("agents", {}).items():
            if agent_data.get("total_trades", 0) > 0:
                logger.info("  Agent '%s': alpha=%.2f, beta=%.2f, mean=%.4f (trades=%d)",
                            agent_name, agent_data["alpha"], agent_data["beta"],
                            agent_data["mean"], agent_data["total_trades"])
        # Check recent trades
        recent = rl_after.get("recent_trades", [])
        if recent:
            logger.info("  Last trade: %s", recent[-1])
    else:
        logger.error("FAIL: RL did NOT record the trade.")

    # === TEST 3: Thompson Sampling weights changed ===
    logger.info("=" * 60)
    logger.info("TEST 3: Thompson Sampling weight sampling")
    logger.info("=" * 60)

    weights_global = learner.get_mean_weights()
    weights_regime = learner.get_mean_weights(regime="state_3_unknown")
    logger.info("Global weights: %s", {k: f"{v:.3f}" for k, v in weights_global.items()})
    logger.info("Regime weights: %s", {k: f"{v:.3f}" for k, v in weights_regime.items()})

    sampled = learner.sample_weights(regime="state_3_unknown")
    logger.info("Sampled weights: %s", {k: f"{v:.3f}" for k, v in sampled.items()})

    # Verify at least some weights changed from default 0.500
    changed = {k: v for k, v in weights_global.items() if abs(v - 0.5) > 0.001}
    if changed:
        logger.info("PASS: Weights updated from default! Changed: %s", list(changed.keys()))
    else:
        logger.warning("WARN: Weights still at default 0.500 (may need more trades)")

    # === TEST 4: Save and reload RL state ===
    logger.info("=" * 60)
    logger.info("TEST 4: RL state persistence")
    logger.info("=" * 60)

    test_path = BINANCE_DIR / "models" / "online_learner_test.json"
    learner._save_path = test_path
    learner.save()
    logger.info("Saved to %s", test_path)

    # Reload
    learner2 = OnlineLearner(save_path=str(test_path))
    learner2.load()
    status2 = learner2.get_status()
    logger.info("Reloaded: total_trades=%d, total_pnl=%.4f",
                status2.get("total_trades", 0), status2.get("total_pnl", 0))

    if status2.get("total_trades", 0) == rl_after.get("total_trades", 0):
        logger.info("PASS: State persisted and reloaded correctly!")
    else:
        logger.error("FAIL: State lost on reload.")

    # Cleanup
    test_path.unlink(missing_ok=True)

    # === SUMMARY ===
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    results = {
        "Safety SL trigger": len(exit_events) > 0,
        "RL trade recorded": rl_after["total_trades"] > 0,
        "TS weights updated": len(changed) > 0,
        "State persistence": status2["total_trades"] == rl_after["total_trades"],
    }
    for test_name, passed in results.items():
        logger.info("  [%s] %s", "PASS" if passed else "FAIL", test_name)

    all_passed = all(results.values())
    logger.info("Overall: %s", "ALL PASSED" if all_passed else "SOME FAILED")
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(test_sell_rl_loop())
    sys.exit(0 if result else 1)
