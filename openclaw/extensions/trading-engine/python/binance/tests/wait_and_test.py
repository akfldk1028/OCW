"""Wait for IP ban to expire, then test connection."""
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("BINANCE_PAPER", "true")

BAN_UNTIL = 1771998225.062
MARGIN = 90  # seconds after ban expiry

wait = BAN_UNTIL - time.time() + MARGIN
if wait > 0:
    print(f"Waiting {wait:.0f}s ({wait/60:.1f}min) for ban to expire + margin...")
    time.sleep(wait)

print("Ban expired! Testing connection...")
from brokers.binance import BinanceBroker

b = BinanceBroker()
r = b.connect()
if r.get("connected"):
    print(f"Connected! USDT: ${r.get('usdt_balance', 0):,.2f}")
    print(f"Mode: {r.get('mode')}, Market: {r.get('market')}")
    print("SUCCESS - ready to run main.py --testnet")
else:
    print(f"FAILED: {r.get('error', 'unknown')}")
    if "418" in str(r.get("error", "")):
        print(f"Still banned! IP ban until: {b._ip_ban_until}")
