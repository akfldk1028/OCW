"""WebSocket-fed OHLCV store — eliminates REST API polling after bootstrap.

On startup, bootstrap from a single REST call per ticker.
After that, WS candle close events append bars — zero REST OHLCV calls.

Usage:
    store = OHLCVStore(maxlen=1500)
    store.bootstrap("BTC/USDT", "1h", ohlcv_list)  # REST data once
    store.append("BTC/USDT", "1h", bar_dict)        # WS candle close
    df = store.get_close_df(["BTC/USDT", "ETH/USDT"], "1h")
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("trading-engine.ohlcv_store")


class OHLCVStore:
    """Thread-safe OHLCV bar store fed by WebSocket candle closes."""

    def __init__(self, maxlen: int = 1500) -> None:
        self._maxlen = maxlen
        self._lock = threading.Lock()
        # {(ticker, interval): deque of (timestamp_ms, o, h, l, c, v)}
        self._bars: Dict[Tuple[str, str], deque] = {}
        self._bootstrapped: Dict[Tuple[str, str], bool] = {}
        self._last_volumes: Dict[str, float] = {}

    def bootstrap(self, ticker: str, interval: str, ohlcv: List[List]) -> int:
        """Load historical OHLCV from REST (once per ticker/interval).

        Args:
            ohlcv: ccxt format [[timestamp_ms, o, h, l, c, v], ...]

        Returns:
            Number of bars loaded.
        """
        key = (ticker, interval)
        with self._lock:
            q = deque(maxlen=self._maxlen)
            for row in ohlcv:
                if len(row) >= 6:
                    q.append(tuple(row[:6]))
            self._bars[key] = q
            self._bootstrapped[key] = True
        count = len(q)
        logger.info("[ohlcv_store] Bootstrap %s/%s: %d bars", ticker, interval, count)
        return count

    def is_bootstrapped(self, ticker: str, interval: str) -> bool:
        return self._bootstrapped.get((ticker, interval), False)

    def append(self, ticker: str, interval: str, bar: Dict[str, Any]) -> None:
        """Append a closed candle bar from WS.

        Args:
            bar: dict with keys: timestamp (epoch s or ms), open, high, low, close, volume
        """
        key = (ticker, interval)
        ts = bar.get("timestamp", 0)
        # Normalize to milliseconds
        if ts < 1e12:
            ts = ts * 1000
        ts = int(ts)

        row = (
            ts,
            float(bar.get("open", 0)),
            float(bar.get("high", 0)),
            float(bar.get("low", 0)),
            float(bar.get("close", bar.get("price", 0))),
            float(bar.get("volume", 0)),
        )

        with self._lock:
            if key not in self._bars:
                self._bars[key] = deque(maxlen=self._maxlen)
            q = self._bars[key]
            # Deduplicate: if last bar has same timestamp, replace it
            if q and q[-1][0] == ts:
                q[-1] = row
            else:
                q.append(row)

    def get_close_df(
        self,
        tickers: List[str],
        interval: str,
    ) -> pd.DataFrame:
        """Build a close-price DataFrame matching runner._fetch_ohlcv_df_sync() output.

        Returns DataFrame with DatetimeIndex and one column per ticker (close prices).
        Also stores volume data accessible via get_latest_volumes().
        """
        frames = {}
        self._last_volumes: Dict[str, float] = {}

        with self._lock:
            for tic in tickers:
                key = (tic, interval)
                q = self._bars.get(key)
                if not q or len(q) == 0:
                    continue
                bars = list(q)
                df = pd.DataFrame(
                    bars,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
                frames[tic] = df["close"]
                self._last_volumes[tic] = float(df["volume"].iloc[-1])

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)

    def get_latest_volumes(self) -> Dict[str, float]:
        """Return latest volume per ticker from last get_close_df() call."""
        return getattr(self, "_last_volumes", {})

    def bar_count(self, ticker: str, interval: str) -> int:
        key = (ticker, interval)
        with self._lock:
            q = self._bars.get(key)
            return len(q) if q else 0
