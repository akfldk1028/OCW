"""Regime Blend SWING backtest — 4h bars, daily rebalance.

Same Regime Blend logic but faster cycle:
- Regime check: daily (not 3-day)
- Entry/Exit signals: every 4h bar
- Trail stop / DD check: every 4h
- Holding period: 1-14 days (swing)

Tests multiple rebalance intervals: 4h, 8h, 12h, 24h

Usage:
    python backtests/backtest_swing.py
"""

import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config import REGIME_BLEND_CONFIG

TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TICKER_MAP = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT"}
START = "2025-03-01"
END = "2026-02-01"
INITIAL_CASH = 5_000.0
TX_COST = 0.001
REPORT_PATH = _ROOT / "backtests" / "backtest_swing_report.txt"


def fetch_4h(symbol, start_ms, end_ms):
    import requests
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
        _time.sleep(0.3)
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


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_bb(series, period=20, n_std=2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + n_std * std, sma - n_std * std


def run_strategy(ticker_data, rebal_bars, cfg, label):
    """Run regime blend with given rebalance interval (in 4h bars)."""
    cash = INITIAL_CASH
    positions = {}
    trailing_highs = {}
    trades = []
    daily_pvs = []
    daily_dates = []

    ref = list(ticker_data.values())[0]
    bt_start = pd.Timestamp(START)
    bt_mask = ref.index >= bt_start
    bt_idx = ref.index[bt_mask]

    last_day = None

    for bar_i, bar_time in enumerate(bt_idx):
        # --- Daily PV snapshot ---
        bar_date = bar_time.date()
        if bar_date != last_day:
            last_day = bar_date
            pv = cash
            for tic, pos in positions.items():
                if tic in ticker_data:
                    idx = ticker_data[tic].index.get_indexer([bar_time], method="pad")[0]
                    if idx >= 0:
                        pv += pos["qty"] * float(ticker_data[tic]["close"].iloc[idx])
            daily_pvs.append(pv)
            daily_dates.append(bar_date)

        # --- Risk check every bar ---
        for tic in list(positions.keys()):
            if tic not in ticker_data:
                continue
            idx = ticker_data[tic].index.get_indexer([bar_time], method="pad")[0]
            if idx < 0:
                continue
            px = float(ticker_data[tic]["close"].iloc[idx])
            pos = positions[tic]
            pnl = (px - pos["entry_price"]) / pos["entry_price"]

            prev_high = trailing_highs.get(tic, pos["entry_price"])
            if px > prev_high:
                trailing_highs[tic] = px
                prev_high = px

            reason = None
            if prev_high > pos["entry_price"] * (1 + cfg["trail_activation_pct"]):
                trail_stop = prev_high * (1 - cfg["trail_pct"])
                if px <= trail_stop:
                    reason = f"trail_stop"
            if reason is None and pnl < -cfg["dd_trigger"]:
                reason = "dd_exit"

            if reason:
                cash += pos["qty"] * px * (1 - TX_COST)
                trades.append({"time": str(bar_time), "tic": tic, "side": "SELL",
                               "price": px, "pnl": pnl, "reason": reason})
                del positions[tic]
                trailing_highs.pop(tic, None)

        # --- Signal every rebal_bars ---
        if bar_i % rebal_bars != 0:
            continue

        for tic in TICKERS:
            if tic not in ticker_data:
                continue
            df = ticker_data[tic]
            idx = df.index.get_indexer([bar_time], method="pad")[0]
            if idx < 30:
                continue
            close = df["close"].iloc[:idx + 1]

            # Regime detection (on 4h close, use vol_window=30 ~ 5 days)
            vol_w = 30  # 30 x 4h = 5 days
            trend_w = 72  # 72 x 4h ≈ 12 days
            if len(close) < trend_w + 5:
                continue

            total_range = close.iloc[-vol_w:].max() - close.iloc[-vol_w:].min()
            net_move = abs(float(close.iloc[-1]) - float(close.iloc[-vol_w]))
            efficiency = net_move / total_range if total_range > 0 else 0
            sma = close.iloc[-trend_w:].mean()
            trend_str = abs(float(close.iloc[-1]) - sma) / sma if sma > 0 else 0

            regime = "trending" if efficiency > 0.45 and trend_str > 0.025 else "ranging"

            px = float(close.iloc[-1])
            rsi = calc_rsi(close, 14)
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            mom = float(close.iloc[-1] / close.iloc[-18] - 1) if len(close) > 18 else 0  # 18 x 4h ≈ 3 days
            _, bb_up, bb_low = calc_bb(close, 20, 2.0)
            bb_l = float(bb_low.iloc[-1]) if not pd.isna(bb_low.iloc[-1]) else 0
            bb_u = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else float('inf')

            signal = "HOLD"
            if regime == "trending":
                if rsi_val > 50 and mom > 0.015:
                    signal = "BUY"
                elif mom < -0.04:
                    signal = "SELL"
            elif regime == "ranging":
                if px < bb_l * 1.02 and rsi_val < 40:
                    signal = "BUY"
                elif px > bb_u * 0.98:
                    signal = "SELL"

            # Execute
            if signal == "BUY" and tic not in positions:
                alloc = cash * cfg["position_pct"]
                if alloc < 10:
                    continue
                qty = alloc / (px * (1 + TX_COST))
                cash -= qty * px * (1 + TX_COST)
                positions[tic] = {"qty": qty, "entry_price": px}
                trailing_highs[tic] = px
                trades.append({"time": str(bar_time), "tic": tic, "side": "BUY",
                               "price": px, "pnl": 0, "reason": f"{regime}"})

            elif signal == "SELL" and tic in positions:
                pos = positions[tic]
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                cash += pos["qty"] * px * (1 - TX_COST)
                trades.append({"time": str(bar_time), "tic": tic, "side": "SELL",
                               "price": px, "pnl": pnl, "reason": f"{regime}"})
                del positions[tic]
                trailing_highs.pop(tic, None)

    return {"label": label, "daily_pvs": daily_pvs, "trades": trades, "dates": daily_dates}


def run():
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 70)
    log("  REGIME BLEND SWING BACKTEST (4h bars)")
    log(f"  Period: {START} ~ {END}")
    log(f"  Initial: ${INITIAL_CASH:,.0f}, Tickers: {TICKERS}")
    log("=" * 70)

    # Download 4h data
    start_dt = datetime.strptime(START, "%Y-%m-%d") - timedelta(days=60)
    end_dt = datetime.strptime(END, "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    ticker_data = {}
    for tic in TICKERS:
        sym = TICKER_MAP[tic]
        log(f"  Downloading {sym} 4h...")
        df = fetch_4h(sym, start_ms, end_ms)
        if not df.empty:
            ticker_data[tic] = df
            log(f"    {len(df)} bars")

    if not ticker_data:
        log("ERROR: no data")
        return

    # B&H benchmark
    bt_start = pd.Timestamp(START)
    bh_starts = {}
    for tic, df in ticker_data.items():
        mask = df.index >= bt_start
        if mask.any():
            bh_starts[tic] = float(df.loc[mask, "close"].iloc[0])

    # Run multiple rebalance intervals
    configs = [
        (1, "4h rebal"),     # every 4h bar
        (2, "8h rebal"),     # every 8h
        (3, "12h rebal"),    # every 12h
        (6, "24h rebal"),    # every 24h (daily)
        (18, "3-day rebal"), # every 3 days (original)
    ]

    # Swing-tuned params
    swing_cfg = {
        "position_pct": 0.25,           # smaller positions for faster trading
        "trail_pct": 0.08,              # tighter trail: 8% from high
        "trail_activation_pct": 0.05,   # activate trail at 5% profit
        "dd_trigger": 0.12,             # 12% drawdown exit
    }

    # Also test original params
    orig_cfg = {
        "position_pct": 0.30,
        "trail_pct": 0.12,
        "trail_activation_pct": 0.08,
        "dd_trigger": 0.15,
    }

    results = []
    for bars, label in configs:
        log(f"\n  Running: {label} (swing params)...")
        r = run_strategy(ticker_data, bars, swing_cfg, f"{label} [swing]")
        results.append(r)

    # Also run 24h with original params for comparison
    log(f"\n  Running: 24h rebal (original params)...")
    r = run_strategy(ticker_data, 6, orig_cfg, "24h rebal [original]")
    results.append(r)

    # Report
    log("\n" + "=" * 70)
    log("  RESULTS COMPARISON")
    log("=" * 70)
    log(f"  {'Strategy':<25} {'Return':>8} {'Sharpe':>7} {'MDD':>8} {'Trades':>7} {'WinR':>6} {'AvgW':>7} {'AvgL':>7}")
    log("-" * 80)

    # B&H
    ref_df = list(ticker_data.values())[0]
    bt_end_idx = ref_df.index.get_indexer([pd.Timestamp(END)], method="pad")[0]
    bh_pv = 0
    alloc_each = INITIAL_CASH / len(bh_starts)
    for tic, sp in bh_starts.items():
        ep = float(ticker_data[tic]["close"].iloc[bt_end_idx])
        bh_pv += alloc_each * ep / sp
    bh_ret = bh_pv / INITIAL_CASH - 1
    log(f"  {'Buy & Hold':<25} {bh_ret:>+7.1%} {'':>7} {'':>8} {'':>7}")

    for r in results:
        pvs = r["daily_pvs"]
        if not pvs:
            continue
        pv_s = pd.Series(pvs, dtype=float)
        ret = pvs[-1] / INITIAL_CASH - 1
        dr = pv_s.pct_change().dropna()
        sharpe = dr.mean() / dr.std() * (365 ** 0.5) if dr.std() > 0 else 0
        mdd = ((pv_s - pv_s.cummax()) / pv_s.cummax()).min()

        sells = [t for t in r["trades"] if t["side"] == "SELL"]
        wins = [t for t in sells if t["pnl"] > 0]
        losses = [t for t in sells if t["pnl"] <= 0]
        wr = len(wins) / len(sells) if sells else 0
        aw = np.mean([t["pnl"] for t in wins]) if wins else 0
        al = np.mean([t["pnl"] for t in losses]) if losses else 0
        n_trades = len(sells)

        log(f"  {r['label']:<25} {ret:>+7.1%} {sharpe:>7.2f} {mdd:>+7.1%} {n_trades:>7} {wr:>5.0%} {aw:>+6.1%} {al:>+6.1%}")

    log("\n" + "=" * 70)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    run()
