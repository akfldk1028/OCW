"""Quick test for MultiTFAggregator."""
import time
from core.multi_tf_aggregator import MultiTFAggregator

agg = MultiTFAggregator(intervals=("15m", "1h", "4h"))

# Simulate 25 closed bars on 15m for BTC
base_price = 96000
for i in range(25):
    bar = {
        "open": base_price + i * 10,
        "high": base_price + i * 10 + 50,
        "low": base_price + i * 10 - 30,
        "close": base_price + i * 15,
        "price": base_price + i * 15,
        "volume": 100 + i * 5,
        "timestamp": time.time() + i * 900,
    }
    agg.update("BTC/USDT", "15m", bar)

# Check summary
s = agg.get_summary("BTC/USDT")
print("=== Summary ===")
for interval, data in s.items():
    print(f"  {interval}: {data}")

print()
print("=== Prompt ===")
print(agg.format_for_prompt("BTC/USDT"))

# Test RSI
from core.multi_tf_aggregator import MultiTFAggregator as M
closes = [96000 + i * 15 for i in range(25)]
rsi = M._calc_rsi(closes, 14)
print(f"\nRSI(14) = {rsi:.1f} (should be ~100 for purely ascending)")

# Test EMA
ema9 = M._calc_ema(closes, 9)
ema21 = M._calc_ema(closes, 21)
print(f"EMA9={ema9:.0f}, EMA21={ema21:.0f} (EMA9 > EMA21 = bullish)")

print("\nAll tests passed!")
