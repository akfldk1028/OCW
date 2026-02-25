"""Wait for IP ban to expire, then run main.py --testnet."""
import time
import sys
import os
import subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

BAN_UNTIL = 1772002079.320
MARGIN = 90  # seconds after ban expiry

# Check + wait
wait = BAN_UNTIL - time.time() + MARGIN
if wait > 0:
    print(f"Waiting {wait:.0f}s ({wait/60:.1f}min) for ban to expire + margin...", flush=True)
    time.sleep(wait)

# Quick test
print("Ban expired! Quick connection test...", flush=True)
parent = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("BINANCE_PAPER", "true")

from brokers.binance import BinanceBroker
b = BinanceBroker()
r = b.connect()
if not r.get("connected"):
    print(f"FAILED: {r.get('error', 'unknown')}", flush=True)
    sys.exit(1)

print(f"Connected! USDT: ${r.get('usdt_balance', 0):,.2f}", flush=True)
print("Starting main.py --testnet...", flush=True)

# Run main.py in same process (unbuffered)
main_py = Path(__file__).resolve().parent.parent / "main.py"
os.chdir(str(main_py.parent))
subprocess.run(
    [sys.executable, "-u", str(main_py), "--testnet"],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
