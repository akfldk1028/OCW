"""Quick connection test."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("BINANCE_PAPER", "true")

from brokers.binance import BinanceBroker

b = BinanceBroker()
r = b.connect()
if r.get("connected"):
    print(f"Connected! USDT: ${r.get('usdt_balance', 0):,.2f}")
    print(f"Mode: {r.get('mode')}, Market: {r.get('market')}")
else:
    print(f"FAILED: {r.get('error', 'unknown')}")
    if "418" in str(r.get("error", "")):
        print(f"IP ban until: {b._ip_ban_until}")
