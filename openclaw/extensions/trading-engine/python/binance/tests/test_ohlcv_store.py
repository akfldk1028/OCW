"""Tests for OHLCVStore — WS-fed OHLCV accumulator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.ohlcv_store import OHLCVStore


def test_bootstrap_and_read():
    """Bootstrap from REST data, then read DataFrame."""
    store = OHLCVStore(maxlen=100)

    # Simulate REST data: [[ts_ms, o, h, l, c, v], ...]
    ohlcv = [
        [1700000000000 + i * 3600000, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i * 10]
        for i in range(50)
    ]
    n = store.bootstrap("BTC/USDT", "1h", ohlcv)
    assert n == 50
    assert store.is_bootstrapped("BTC/USDT", "1h")
    assert store.bar_count("BTC/USDT", "1h") == 50

    df = store.get_close_df(["BTC/USDT"], "1h")
    assert not df.empty
    assert "BTC/USDT" in df.columns
    assert len(df) == 50
    # Last close should be 102 + 49 = 151
    assert df["BTC/USDT"].iloc[-1] == 151.0


def test_ws_append_updates_store():
    """WS candle closes append to the store."""
    store = OHLCVStore(maxlen=100)
    store.bootstrap("BTC/USDT", "1h", [
        [1700000000000, 100, 105, 95, 102, 1000],
        [1700003600000, 102, 108, 100, 106, 1200],
    ])
    assert store.bar_count("BTC/USDT", "1h") == 2

    # WS candle close (next hour)
    store.append("BTC/USDT", "1h", {
        "timestamp": 1700007200,  # epoch seconds
        "open": 106, "high": 112, "low": 104, "close": 110, "volume": 1500,
    })
    assert store.bar_count("BTC/USDT", "1h") == 3

    df = store.get_close_df(["BTC/USDT"], "1h")
    assert len(df) == 3
    assert df["BTC/USDT"].iloc[-1] == 110.0


def test_dedup_same_timestamp():
    """Same timestamp replaces, doesn't duplicate."""
    store = OHLCVStore(maxlen=100)
    store.bootstrap("ETH/USDT", "1h", [
        [1700000000000, 100, 105, 95, 102, 1000],
    ])

    # Same timestamp, different close
    store.append("ETH/USDT", "1h", {
        "timestamp": 1700000000000,  # ms, same as bootstrap
        "open": 100, "high": 105, "low": 95, "close": 108, "volume": 1100,
    })
    assert store.bar_count("ETH/USDT", "1h") == 1

    df = store.get_close_df(["ETH/USDT"], "1h")
    assert df["ETH/USDT"].iloc[0] == 108.0  # replaced


def test_multi_ticker():
    """Multiple tickers in one store."""
    store = OHLCVStore(maxlen=100)
    store.bootstrap("BTC/USDT", "1h", [
        [1700000000000, 100, 105, 95, 102, 1000],
    ])
    store.bootstrap("ETH/USDT", "1h", [
        [1700000000000, 3000, 3100, 2900, 3050, 5000],
    ])

    df = store.get_close_df(["BTC/USDT", "ETH/USDT"], "1h")
    assert "BTC/USDT" in df.columns
    assert "ETH/USDT" in df.columns
    assert df["BTC/USDT"].iloc[0] == 102.0
    assert df["ETH/USDT"].iloc[0] == 3050.0


def test_maxlen_enforced():
    """Store respects maxlen."""
    store = OHLCVStore(maxlen=10)
    ohlcv = [
        [1700000000000 + i * 3600000, 100, 105, 95, 100 + i, 1000]
        for i in range(20)
    ]
    store.bootstrap("BTC/USDT", "1h", ohlcv)
    assert store.bar_count("BTC/USDT", "1h") == 10

    df = store.get_close_df(["BTC/USDT"], "1h")
    # Should have last 10 bars (100+10 through 100+19)
    assert df["BTC/USDT"].iloc[0] == 110.0
    assert df["BTC/USDT"].iloc[-1] == 119.0


def test_volumes():
    """get_latest_volumes returns correct values."""
    store = OHLCVStore(maxlen=100)
    store.bootstrap("BTC/USDT", "1h", [
        [1700000000000, 100, 105, 95, 102, 1000],
        [1700003600000, 102, 108, 100, 106, 1500],
    ])
    store.get_close_df(["BTC/USDT"], "1h")
    vols = store.get_latest_volumes()
    assert vols["BTC/USDT"] == 1500.0


if __name__ == "__main__":
    test_bootstrap_and_read()
    test_ws_append_updates_store()
    test_dedup_same_timestamp()
    test_multi_ticker()
    test_maxlen_enforced()
    test_volumes()
    print("ALL PASSED")
