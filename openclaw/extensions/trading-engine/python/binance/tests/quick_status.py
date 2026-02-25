"""Quick status check for the trading engine."""
import json
import os
import time
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
DECISIONS_LOG = LOGS_DIR / "decisions.jsonl"
STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "runner_state.json"

total = 0
trades_count = 0
holds = 0
last_assessment = ""
last_next_check = 0

if DECISIONS_LOG.exists():
    with open(DECISIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line.strip())
                total += 1
                if d.get("trades"):
                    trades_count += len(d["trades"])
                else:
                    holds += 1
                a = d.get("claude_assessment", "")
                if a:
                    last_assessment = a[:150]
                nc = d.get("next_check_s", 0)
                if nc:
                    last_next_check = nc
            except Exception:
                pass

positions = {}
memory = ""
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        positions = state.get("entry_prices", {})
        memory = state.get("agent_memory", "")[:200]
    except Exception:
        pass

logs = sorted(LOGS_DIR.glob("trading_*.log"), key=os.path.getmtime, reverse=True)
log_age = time.time() - os.path.getmtime(logs[0]) if logs else 999

print(f"=== Trading Engine Status ===")
print(f"Runner alive:      {'YES' if log_age < 300 else 'NO'} (log age: {log_age:.0f}s)")
print(f"Total decisions:   {total}")
print(f"Trade decisions:   {trades_count}")
print(f"HOLDs:             {holds}")
print(f"Positions:         {list(positions.keys()) or 'none'}")
for tic, px in positions.items():
    print(f"  {tic}: entry ${px:,.2f}")
print(f"Last next_check:   {last_next_check}s")
print(f"Last assessment:   {last_assessment}")
print(f"Agent memory:      {memory}")
