"""Fetch historical OHLCV + derivatives data from Binance.

Caches results as parquet files to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict

import pandas as pd
import requests

from pathlib import Path

from .config import (
    TICKERS, TICKER_MAP, DATA_DIR,
    BINANCE_SPOT_URL, BINANCE_FUTURES_URL,
)

logger = logging.getLogger("pretrain.fetch")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch klines from Binance spot /api/v3/klines with pagination."""
    all_data = []
    current = start_ms
    while current < end_ms:
        resp = requests.get(f"{BINANCE_SPOT_URL}/api/v3/klines", params={
            "symbol": symbol, "interval": interval,
            "startTime": current, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
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
        "ct", "qv", "tr", "tbv", "tbqv", "ig",
    ])
    df["timestamp"] = pd.to_datetime(df["ot"], unit="ms")
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch funding rate history from Binance futures API."""
    all_data = []
    current = start_ms
    while current < end_ms:
        try:
            resp = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate", params={
                "symbol": symbol, "startTime": current, "endTime": end_ms, "limit": 1000,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch funding rates for %s: %s", symbol, e)
            break
        if not data:
            break
        all_data.extend(data)
        current = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df = df.set_index("timestamp")
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["fundingRate"]]


def fetch_oi_history(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch open interest history from Binance futures data API.

    Binance /futures/data/ APIs reject startTime — paginate backwards via endTime.
    Data availability: ~30 days for 1h, more for 4h.
    """
    all_data = []
    cursor_end = end_ms
    for _ in range(50):  # safety limit
        try:
            params = {"symbol": symbol, "period": interval, "limit": 500}
            if cursor_end:
                params["endTime"] = cursor_end
            resp = requests.get(f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist",
                                params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch OI history for %s: %s", symbol, e)
            break
        if not data:
            break
        all_data.extend(data)
        earliest = data[0]["timestamp"]
        if earliest <= start_ms:
            break
        cursor_end = earliest - 1
        if len(data) < 500:
            break
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    # Deduplicate and sort
    seen = set()
    unique = []
    for d in all_data:
        ts = d["timestamp"]
        if ts not in seen and ts >= start_ms:
            seen.add(ts)
            unique.append(d)
    unique.sort(key=lambda x: x["timestamp"])

    df = pd.DataFrame(unique)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("ts")
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    return df[["sumOpenInterest", "sumOpenInterestValue"]]


def fetch_long_short_ratio(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch top trader long/short ratio from Binance futures data API.

    Same reverse-pagination as OI — Binance rejects startTime for these endpoints.
    """
    all_data = []
    cursor_end = end_ms
    for _ in range(50):
        try:
            params = {"symbol": symbol, "period": interval, "limit": 500}
            if cursor_end:
                params["endTime"] = cursor_end
            resp = requests.get(f"{BINANCE_FUTURES_URL}/futures/data/topLongShortAccountRatio",
                                params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch L/S ratio for %s: %s", symbol, e)
            break
        if not data:
            break
        all_data.extend(data)
        earliest = data[0]["timestamp"]
        if earliest <= start_ms:
            break
        cursor_end = earliest - 1
        if len(data) < 500:
            break
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    seen = set()
    unique = []
    for d in all_data:
        ts = d["timestamp"]
        if ts not in seen and ts >= start_ms:
            seen.add(ts)
            unique.append(d)
    unique.sort(key=lambda x: x["timestamp"])

    df = pd.DataFrame(unique)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("ts")
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    return df[["longShortRatio"]]


def _load_or_fetch(cache_path: Path, fetch_fn, label: str) -> pd.DataFrame:
    """Load from CSV cache or fetch and save."""
    if cache_path.exists():
        logger.info("  [cache] %s loaded from %s", label, cache_path.name)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    logger.info("  [fetch] %s from Binance...", label)
    df = fetch_fn()
    if not df.empty:
        df.to_csv(cache_path)
        logger.info("  [saved] %d records -> %s", len(df), cache_path.name)
    return df


def fetch_and_cache_all(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch and cache all data types for all tickers.

    Returns: {ticker: {"klines": df, "funding": df, "oi": df, "ls_ratio": df}}
    """
    from .config import DEFAULT_START, DEFAULT_END

    tickers = tickers or TICKERS
    start = start or DEFAULT_START
    end = end or DEFAULT_END

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)

    result: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ticker in tickers:
        symbol = TICKER_MAP.get(ticker, ticker.replace("/", ""))
        logger.info("Processing %s (%s)...", ticker, symbol)
        ticker_data: Dict[str, pd.DataFrame] = {}

        # 1h klines
        ticker_data["klines"] = _load_or_fetch(
            DATA_DIR / f"{symbol}_1h.csv",
            lambda: fetch_klines(symbol, "1h", start_ms, end_ms),
            "klines",
        )

        # Funding rates
        ticker_data["funding"] = _load_or_fetch(
            DATA_DIR / f"{symbol}_funding.csv",
            lambda: fetch_funding_rates(symbol, start_ms, end_ms),
            "funding",
        )

        # OI history (1h)
        ticker_data["oi"] = _load_or_fetch(
            DATA_DIR / f"{symbol}_oi_1h.csv",
            lambda: fetch_oi_history(symbol, "1h", start_ms, end_ms),
            "OI",
        )

        # Long/short ratio (1h)
        ticker_data["ls_ratio"] = _load_or_fetch(
            DATA_DIR / f"{symbol}_ls_ratio_1h.csv",
            lambda: fetch_long_short_ratio(symbol, "1h", start_ms, end_ms),
            "L/S ratio",
        )

        result[ticker] = ticker_data
        logger.info("  Done: klines=%d, funding=%d, oi=%d, ls=%d",
                     len(ticker_data.get("klines", [])),
                     len(ticker_data.get("funding", [])),
                     len(ticker_data.get("oi", [])),
                     len(ticker_data.get("ls_ratio", [])))

    return result
