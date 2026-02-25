"""Fetch historical 4h OHLCV data from Binance for Claude backtest.

Downloads and caches data locally to avoid repeated API calls.

Usage:
    python backtests/backtest_claude/fetch_data.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import requests

TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TICKER_MAP = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT"}
START = "2025-03-01"
END = "2026-02-01"
DATA_DIR = Path(__file__).resolve().parent / "data"


def fetch_4h(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 4h klines from Binance REST API."""
    all_data = []
    current = start_ms
    while current < end_ms:
        resp = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": symbol, "interval": "4h",
            "startTime": current, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.3)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        "ot", "open", "high", "low", "close", "volume",
        "ct", "qv", "tr", "tbv", "tbqv", "ig"])
    df["timestamp"] = pd.to_datetime(df["ot"], unit="ms")
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_and_cache() -> dict[str, pd.DataFrame]:
    """Fetch all ticker data, cache as parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = int(datetime.strptime(START, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(END, "%Y-%m-%d").timestamp() * 1000)

    ticker_data = {}
    for ticker, symbol in TICKER_MAP.items():
        cache_path = DATA_DIR / f"{symbol}_4h.parquet"
        if cache_path.exists():
            print(f"  [cache] {ticker} loaded from {cache_path.name}")
            ticker_data[ticker] = pd.read_parquet(cache_path)
        else:
            print(f"  [fetch] {ticker} from Binance...")
            df = fetch_4h(symbol, start_ms, end_ms)
            if df.empty:
                print(f"  [warn] No data for {ticker}")
                continue
            df.to_parquet(cache_path)
            print(f"  [saved] {ticker}: {len(df)} bars -> {cache_path.name}")
            ticker_data[ticker] = df

    return ticker_data


if __name__ == "__main__":
    data = fetch_and_cache()
    for tic, df in data.items():
        print(f"{tic}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
