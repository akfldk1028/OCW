"""Quick summary of decisions.jsonl."""
import json
from pathlib import Path

DECISIONS = Path(__file__).resolve().parent.parent / "logs" / "decisions.jsonl"

decisions = []
with open(DECISIONS, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line))
        except Exception:
            pass

# Count triggers
triggers = {}
has_positions = 0
for d in decisions:
    t = d.get("trigger", "?")
    triggers[t] = triggers.get(t, 0) + 1
    pos = d.get("positions", {})
    if pos:
        has_positions += 1

print(f"Total decisions: {len(decisions)}")
print(f"Decisions with positions: {has_positions}")
print(f"\nTriggers:")
for t, c in sorted(triggers.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

# Last 10 decisions
print("\n=== Last 10 decisions ===")
for d in decisions[-10:]:
    ts = d.get("timestamp", "?")
    if len(ts) > 19:
        ts = ts[:19]
    trigger = d.get("trigger", "?")
    btc = d.get("btc_price", 0)
    pv = d.get("portfolio_value", 0)
    positions = d.get("positions", {})
    if isinstance(positions, dict):
        pos_str = ", ".join(positions.keys()) if positions else "CASH"
    elif isinstance(positions, list):
        pos_str = str(len(positions)) + " positions" if positions else "CASH"
    else:
        pos_str = str(positions)[:30] if positions else "CASH"
    assessment = d.get("claude_assessment", "")
    if assessment:
        assessment = assessment[:120]
    next_s = d.get("next_check_s", "?")
    print(f"{ts} | {trigger:15} | BTC ${btc:,.0f} | PV ${pv:,.0f} | {pos_str}")
    print(f"  -> {assessment}")
    print(f"  next_check: {next_s}s")
    print()
