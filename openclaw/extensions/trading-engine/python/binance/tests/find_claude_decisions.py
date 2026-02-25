"""Find decisions where Claude actually gave an assessment."""
import io
import sys
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DECISIONS = Path(__file__).resolve().parent.parent / "logs" / "decisions.jsonl"

with open(DECISIONS, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

count = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    ca = d.get("claude_assessment", "")
    if ca and len(ca) > 20:
        count += 1
        ts = d.get("timestamp", "")[:19]
        pv = d.get("portfolio_value", 0)
        btc = d.get("btc_price", 0)
        trigger = d.get("trigger", "?")
        next_s = d.get("next_check_s", "?")
        print(f"[{count}] {ts} | {trigger} | BTC ${btc:,.0f} | PV ${pv:,.0f} | next:{next_s}s")
        print(f"    {ca[:250]}")
        print()

print(f"Total decisions with Claude assessment: {count} / {len(lines)}")
