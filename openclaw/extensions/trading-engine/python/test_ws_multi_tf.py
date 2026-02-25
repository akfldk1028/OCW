"""Quick test: connect to Binance testnet WS, verify multi-TF ticks arrive."""
import asyncio
import json
import time
from collections import defaultdict

import websockets

WS_URL = "wss://stream.testnet.binance.vision/stream"
TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PAXG/USDT"]
INTERVALS = ["15m", "1h", "4h"]

streams = []
for tic in TICKERS:
    symbol = tic.replace("/", "").lower()
    for iv in INTERVALS:
        streams.append(f"{symbol}@kline_{iv}")

url = f"{WS_URL}?streams={'/'.join(streams)}"
print(f"Connecting to {len(streams)} streams...")
print(f"URL: {url[:120]}...")


async def main():
    counts = defaultdict(lambda: defaultdict(int))  # {ticker: {interval: count}}
    start = time.time()
    duration = 30  # seconds

    async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
        print(f"Connected! Listening for {duration}s...\n")
        while time.time() - start < duration:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                data = msg.get("data", {})
                if data.get("e") == "kline":
                    kline = data.get("k", {})
                    symbol = data.get("s", "")
                    interval = kline.get("i", "?")
                    close = float(kline.get("c", 0))
                    is_closed = kline.get("x", False)
                    counts[symbol][interval] += 1

                    # Print first few and closed candles
                    total = sum(sum(v.values()) for v in counts.values())
                    if total <= 12 or is_closed:
                        flag = " [CLOSED]" if is_closed else ""
                        print(f"  {symbol} {interval}: ${close:,.2f}{flag}")
            except asyncio.TimeoutError:
                print("  (no message for 5s)")

    print(f"\n=== Summary ({duration}s) ===")
    for symbol in sorted(counts.keys()):
        for iv in INTERVALS:
            c = counts[symbol].get(iv, 0)
            print(f"  {symbol} {iv}: {c} ticks")

    total = sum(sum(v.values()) for v in counts.values())
    print(f"\nTotal: {total} ticks from {len(counts)} symbols x {len(INTERVALS)} intervals")
    if total > 0:
        print("PASS: Multi-TF WS subscription working!")
    else:
        print("FAIL: No ticks received")


asyncio.run(main())
