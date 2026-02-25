"""Quick integration test — verify all imports and new fields."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.claude_agent import MarketSnapshot, SYSTEM_PROMPT
s = MarketSnapshot()
assert hasattr(s, "basis_spreads"), "Missing basis_spreads"
assert hasattr(s, "etf_daily_flow_usd"), "Missing etf_daily_flow_usd"
assert hasattr(s, "etf_flow_label"), "Missing etf_flow_label"
assert "signal_weights" in SYSTEM_PROMPT, "Missing signal_weights in prompt"
print("claude_agent.py: OK")

from core.zscore_gate import AdaptiveGate
gate = AdaptiveGate()
print("zscore_gate.py: OK")

from core.online_learner import OnlineLearner
ol = OnlineLearner()
print("online_learner.py: OK")

from config import REGIME_BLEND_CONFIG, SWING_BLEND_CONFIG
assert "PAXG/USDT" in REGIME_BLEND_CONFIG["tickers"], "PAXG missing from config"
assert "PAXG/USDT" in SWING_BLEND_CONFIG["tickers"], "PAXG missing from swing"
print("config.py: PAXG OK")

# Test TS signal weights flow
ol2 = OnlineLearner()
result = ol2.record_trade(
    ticker="BTC/USDT", entry_price=60000, exit_price=63000,
    pnl_pct=0.05, held_hours=12,
    agent_signals={"momentum": 0.7, "funding_rate": -0.3, "sentiment": 0.5},
    regime="trending",
)
assert result["profitable"] is True
assert result["regime"] == "trending"
print(f"online_learner TS: OK (trade #{result['trade_number']})")

# Test prompt generation
s2 = MarketSnapshot(
    candidates=["BTC/USDT"], ticker_prices={"BTC/USDT": 63000},
    basis_spreads={"BTC/USDT": 0.15}, etf_daily_flow_usd=150e6,
    etf_flow_label="inflow", fear_greed_index=25, fear_greed_label="Extreme Fear",
)
prompt = s2.to_prompt()
assert "basis=" in prompt, "basis not in prompt"
assert "ETF Daily Flow" in prompt, "ETF flow not in prompt"
print("to_prompt(): OK")

print("\nALL TESTS PASSED")
