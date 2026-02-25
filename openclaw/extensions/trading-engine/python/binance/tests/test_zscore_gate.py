"""Tests for core.zscore_gate — AdaptiveGate, ZScoreTracker, WakeCondition."""

import sys
import time
from pathlib import Path

# Ensure imports work from tests/
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.zscore_gate import ZScoreTracker, WakeCondition, AdaptiveGate


# --- ZScoreTracker ---

def test_zscore_returns_none_with_few_samples():
    t = ZScoreTracker(window=50)
    assert t.update(100) is None
    assert t.update(101) is None  # still < 3

def test_zscore_normal_values():
    t = ZScoreTracker(window=50)
    for v in [100, 101, 99, 100, 101, 99, 100]:
        z = t.update(v)
    assert z is not None
    assert abs(z) < 1.5  # normal value, low z-score

def test_zscore_outlier():
    t = ZScoreTracker(window=50)
    for _ in range(30):
        t.update(100.0)
    z = t.update(110.0)  # big jump
    assert z is not None
    assert abs(z) > 2.0


# --- WakeCondition ---

def test_wake_gt():
    wc = WakeCondition(metric="btc_price", operator="gt", threshold=95000)
    assert wc.evaluate(96000, 94000) is True
    assert wc.evaluate(94000, 94000) is False

def test_wake_lt():
    wc = WakeCondition(metric="btc_price", operator="lt", threshold=95000)
    assert wc.evaluate(94000, 96000) is True
    assert wc.evaluate(96000, 96000) is False

def test_wake_crosses_above():
    wc = WakeCondition(metric="btc_price", operator="crosses_above", threshold=95000)
    assert wc.evaluate(95500, 94500) is True  # crossed above
    assert wc.evaluate(96000, 95500) is False  # already above
    assert wc.evaluate(95500, None) is False   # no previous

def test_wake_crosses_below():
    wc = WakeCondition(metric="btc_price", operator="crosses_below", threshold=95000)
    assert wc.evaluate(94500, 95500) is True
    assert wc.evaluate(94000, 94500) is False  # already below

def test_wake_abs_change_pct():
    wc = WakeCondition(metric="btc_price", operator="abs_change_pct_gt", threshold=0.02)
    assert wc.evaluate(102100, 100000) is True   # 2.1% change
    assert wc.evaluate(100500, 100000) is False   # 0.5% change

def test_wake_none_current():
    wc = WakeCondition(metric="btc_price", operator="gt", threshold=95000)
    assert wc.evaluate(None, 94000) is False


# --- AdaptiveGate ---

def test_gate_candle_close_always_passes():
    gate = AdaptiveGate(min_check_seconds=0)
    ok, reasons = gate.evaluate({"btc_price": 100000}, is_candle_close=True)
    assert ok is True
    assert "candle_close" in reasons

def test_gate_cooldown_blocks():
    gate = AdaptiveGate(min_check_seconds=300)
    # First call passes (candle close)
    gate.evaluate({"btc_price": 100000}, is_candle_close=True)
    # Immediate second call blocked by cooldown (not candle close)
    ok, reasons = gate.evaluate({"btc_price": 100000}, is_candle_close=False)
    assert ok is False
    assert reasons == []

def test_gate_timer_expired():
    gate = AdaptiveGate(min_check_seconds=0)
    gate.update_from_claude(next_check_seconds=0.01)  # expires almost immediately
    time.sleep(0.02)
    ok, reasons = gate.evaluate({"btc_price": 100000}, is_candle_close=False)
    assert ok is True
    assert any("timer_expired" in r for r in reasons)

def test_gate_zscore_trigger():
    gate = AdaptiveGate(zscore_threshold=2.0, zscore_window=50, min_check_seconds=0)
    # Build baseline
    for _ in range(30):
        gate.evaluate({"volume": 1000.0}, is_candle_close=False)
    # Spike
    ok, reasons = gate.evaluate({"volume": 5000.0}, is_candle_close=False)
    assert ok is True
    assert any("zscore:volume" in r for r in reasons)

def test_gate_wake_condition_trigger():
    gate = AdaptiveGate(min_check_seconds=0)
    gate.update_from_claude(
        next_check_seconds=99999,
        wake_conditions=[{"metric": "btc_price", "operator": "lt", "threshold": 95000, "reason": "support break"}],
    )
    # First tick sets prev_features
    gate.evaluate({"btc_price": 96000}, is_candle_close=False)
    # Second tick triggers condition
    ok, reasons = gate.evaluate({"btc_price": 94000}, is_candle_close=False)
    assert ok is True
    assert any("support break" in r for r in reasons)

def test_gate_update_from_claude():
    gate = AdaptiveGate(min_check_seconds=0)
    gate.update_from_claude(
        next_check_seconds=600,
        wake_conditions=[
            {"metric": "btc_price", "operator": "gt", "threshold": 100000, "reason": "breakout"},
        ],
    )
    status = gate.get_status()
    assert status["wake_conditions"] == 1
    assert status["next_check_in"] is not None
    assert status["next_check_in"] > 500

def test_gate_no_trigger_returns_false():
    gate = AdaptiveGate(min_check_seconds=0, zscore_threshold=10.0)
    # Normal tick, no conditions, no timer, high z threshold
    for _ in range(5):
        gate.evaluate({"btc_price": 100000}, is_candle_close=False)
    ok, reasons = gate.evaluate({"btc_price": 100001}, is_candle_close=False)
    assert ok is False


# --- ClaudeAgent circuit breaker ---

def test_circuit_breaker_half_open():
    """Circuit breaker should allow retry after cooldown (C2 fix)."""
    from unittest.mock import patch
    from core.claude_agent import ClaudeAgent

    agent = ClaudeAgent()
    agent._circuit_cooldown = 0.05  # 50ms for test
    agent._available = True
    agent._auth_configured = True

    # Simulate 3 consecutive failures → circuit open
    agent._consecutive_failures = 3
    assert agent.is_available is False  # breaker open

    # Still blocked before cooldown
    assert agent.is_available is False

    # Wait for cooldown
    time.sleep(0.06)

    # Half-open: should reset and re-check
    with patch("core.claude_agent._check_sdk", return_value=True):
        with patch.object(agent, "_ensure_auth", return_value=True):
            assert agent.is_available is True  # half-open → retry succeeds
    assert agent._consecutive_failures == 0
    assert agent._circuit_open_time == 0.0


def test_circuit_breaker_reset_on_success():
    """Successful call should reset circuit breaker fully."""
    from core.claude_agent import ClaudeAgent

    agent = ClaudeAgent()
    agent._consecutive_failures = 2
    agent._circuit_open_time = time.time()

    agent.reset_circuit_breaker()
    assert agent._consecutive_failures == 0
    assert agent._circuit_open_time == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
