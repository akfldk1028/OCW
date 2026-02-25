"""Parameter optimization for Crypto Regime Blend.

Sweep key parameters to find the best risk-adjusted performance:
- max_exposure: 40-80%
- position_pct: 20-40%
- trail_pct: 8-15%
- dd_trigger: 8-15%
- rebalance_days: 2-7

Goal: Sharpe > 1.0, MDD < BTC MDD (~37%), alpha close to 0.
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.where(loss > 0, 1e-10)
    return 100 - 100 / (1 + rs)


def calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + num_std * std, sma - num_std * std


def detect_regime(close: pd.Series, date, vol_window: int = 20, trend_window: int = 50) -> str:
    hist = close.loc[:date].dropna()
    if len(hist) < max(vol_window, trend_window) + 5:
        return "unknown"
    total_range = hist.iloc[-vol_window:].max() - hist.iloc[-vol_window:].min()
    net_move = abs(hist.iloc[-1] - hist.iloc[-vol_window])
    efficiency = net_move / total_range if total_range > 0 else 0
    sma = hist.iloc[-trend_window:].mean()
    trend_strength = abs(hist.iloc[-1] - sma) / sma
    if efficiency > 0.5 and trend_strength > 0.03:
        return "trending"
    return "ranging"


def run_regime_blend(
    close: pd.DataFrame,
    btc: pd.Series,
    bt_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    tickers: List[str],
    initial_cash: float,
    max_exposure: float,
    position_pct: float,
    trail_pct: float,
    dd_trigger: float,
    rebalance_days: int,
    tx_cost: float = 0.001,
) -> Dict[str, Any]:
    """Run regime blend with given parameters. Returns metrics only (no printing)."""

    cash = initial_cash
    positions: Dict[str, Dict] = {}
    daily_values: List[float] = []
    n_trades = 0

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        pv = cash
        exits = []
        for tic, pos in list(positions.items()):
            if tic not in close.columns:
                continue
            px = float(close[tic].iloc[date_idx])
            if pd.isna(px) or px <= 0:
                continue
            pos["current_price"] = px
            if px > pos.get("high", px):
                pos["high"] = px
            pv += pos["qty"] * px

            trail_stop = pos["high"] * (1 - trail_pct)
            if pos["high"] > pos["entry_price"] * 1.03 and px <= trail_stop:
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                exits.append((tic, pnl))

        for tic, pnl in exits:
            pos = positions[tic]
            cash += pos["qty"] * pos["current_price"] * (1 - tx_cost)
            del positions[tic]
            n_trades += 1

        pv = cash + sum(
            p["qty"] * p.get("current_price", p["entry_price"])
            for p in positions.values()
        )
        daily_values.append(pv)

        if day_i % rebalance_days != 0:
            continue

        # DD risk management
        if len(daily_values) > 20:
            peak_pv = max(daily_values[-20:])
            current_dd = (peak_pv - pv) / peak_pv
            if current_dd > dd_trigger:
                for tic in list(positions.keys()):
                    pos = positions[tic]
                    px = pos.get("current_price", pos["entry_price"])
                    cash += pos["qty"] * px * (1 - tx_cost)
                    del positions[tic]
                    n_trades += 1
                continue

        regime = detect_regime(btc, date)

        for tic in tickers:
            if tic in positions or tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if len(hist) < 60:
                continue

            px = float(hist.iloc[-1])
            rsi = float(calc_rsi(hist).iloc[-1]) if len(hist) > 15 else 50
            mom_14d = float(hist.iloc[-1] / hist.iloc[-14] - 1) if len(hist) > 14 else 0
            _, _, bb_lower = calc_bollinger(hist, 20, 2.0)
            bb_low = float(bb_lower.iloc[-1]) if not bb_lower.isna().iloc[-1] else 0

            buy = False
            if regime == "trending" and rsi > 50 and mom_14d > 0.02:
                buy = True
            elif regime == "ranging" and px < bb_low * 1.02 and rsi < 40:
                buy = True
            elif regime == "unknown" and rsi > 55 and mom_14d > 0.05:
                buy = True

            if not buy:
                continue

            current_pv = cash + sum(
                p["qty"] * p.get("current_price", p["entry_price"])
                for p in positions.values()
            )
            exp = 1 - (cash / current_pv) if current_pv > 0 else 0
            if exp >= max_exposure:
                continue

            alloc = current_pv * position_pct
            alloc = min(alloc, cash * 0.90)
            if alloc < 50:
                continue

            qty = alloc / px
            cost = alloc * (1 + tx_cost)
            if cost > cash:
                continue

            cash -= cost
            positions[tic] = {
                "qty": qty, "entry_price": px,
                "high": px, "current_price": px,
            }
            n_trades += 1

        # Signal-based exits
        for tic in list(positions.keys()):
            if tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if len(hist) < 14:
                continue
            mom = float(hist.iloc[-1] / hist.iloc[-14] - 1)
            _, bb_up, _ = calc_bollinger(hist, 20, 2.0)
            bb_u = float(bb_up.iloc[-1]) if not bb_up.isna().iloc[-1] else float('inf')
            px = float(hist.iloc[-1])

            sell = False
            if regime == "trending" and mom < -0.05:
                sell = True
            elif regime == "ranging" and px > bb_u * 0.98:
                sell = True

            if sell:
                pos = positions[tic]
                cash += pos["qty"] * px * (1 - tx_cost)
                del positions[tic]
                n_trades += 1

    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": True}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 365**0.5) / max(np.std(returns), 1e-8)
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / np.maximum(peak, 1e-8)))

    btc_start = float(btc.loc[bt_dates[0]])
    btc_end = float(btc.loc[bt_dates[-1]])
    btc_return = btc_end / btc_start - 1
    alpha = total_return - btc_return

    btc_vals = btc.loc[bt_dates].values.astype(float)
    btc_peak = np.maximum.accumulate(btc_vals)
    btc_mdd = float(np.max((btc_peak - btc_vals) / np.maximum(btc_peak, 1e-8)))

    return {
        "total_return": total_return,
        "btc_return": btc_return,
        "alpha": alpha,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "btc_mdd": btc_mdd,
        "n_trades": n_trades,
        "final_value": values[-1],
    }


if __name__ == "__main__":
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
    start_date = "2023-01-01"
    end_date = "2026-02-01"
    initial_cash = 5_000.0

    print("Downloading data...")
    dl_start = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    btc = close["BTC-USD"]
    all_dates = close.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    # Parameter grid
    param_grid = {
        "max_exposure": [0.50, 0.60, 0.70],
        "position_pct": [0.20, 0.30, 0.35],
        "trail_pct": [0.10, 0.12, 0.15],
        "dd_trigger": [0.08, 0.10, 0.12],
        "rebalance_days": [3, 5],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"Testing {len(combos)} parameter combinations...\n")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        r = run_regime_blend(
            close=close, btc=btc, bt_dates=bt_dates, all_dates=all_dates,
            tickers=tickers, initial_cash=initial_cash, **params
        )

        if "error" in r:
            continue

        # Score: maximize Sharpe while keeping MDD < BTC MDD
        mdd_penalty = max(0, r["max_dd"] - r["btc_mdd"]) * 5  # heavy penalty for exceeding BTC MDD
        score = r["sharpe"] - mdd_penalty + min(r["alpha"], 0) * 2

        results.append({**params, **r, "score": score})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(combos)} done ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(combos)} combos in {elapsed:.0f}s\n")

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Print top 10
    print("=" * 100)
    print("  TOP 10 PARAMETER COMBINATIONS (by risk-adjusted score)")
    print("=" * 100)
    print(f"  {'#':>3} {'MaxExp':>7} {'PosPct':>7} {'Trail':>7} {'DDTrig':>7} {'Rebal':>6} | "
          f"{'Return':>9} {'Alpha':>9} {'Sharpe':>7} {'MDD':>7} {'BtcMDD':>7} {'Trades':>7} {'Score':>7}")
    print(f"  {'-'*95}")

    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>3} "
              f"{r['max_exposure']:>7.0%} "
              f"{r['position_pct']:>7.0%} "
              f"{r['trail_pct']:>7.0%} "
              f"{r['dd_trigger']:>7.0%} "
              f"{r['rebalance_days']:>6} | "
              f"{r['total_return']*100:>+8.1f}% "
              f"{r['alpha']*100:>+8.1f}% "
              f"{r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>6.1f}% "
              f"{r['btc_mdd']*100:>6.1f}% "
              f"{r['n_trades']:>7} "
              f"{r['score']:>7.2f}")

    # Print worst 5 for contrast
    print(f"\n  {'WORST 5':}")
    for r in results[-5:]:
        print(f"      "
              f"{r['max_exposure']:>7.0%} "
              f"{r['position_pct']:>7.0%} "
              f"{r['trail_pct']:>7.0%} "
              f"{r['dd_trigger']:>7.0%} "
              f"{r['rebalance_days']:>6} | "
              f"{r['total_return']*100:>+8.1f}% "
              f"{r['alpha']*100:>+8.1f}% "
              f"{r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>6.1f}% "
              f"{r['n_trades']:>7}")

    # Best params
    best = results[0]
    print(f"\n{'='*70}")
    print(f"  BEST PARAMETERS:")
    print(f"  max_exposure = {best['max_exposure']:.0%}")
    print(f"  position_pct = {best['position_pct']:.0%}")
    print(f"  trail_pct = {best['trail_pct']:.0%}")
    print(f"  dd_trigger = {best['dd_trigger']:.0%}")
    print(f"  rebalance_days = {best['rebalance_days']}")
    print(f"")
    print(f"  Return: {best['total_return']*100:+.1f}%")
    print(f"  BTC B&H: {best['btc_return']*100:+.1f}%")
    print(f"  Alpha: {best['alpha']*100:+.1f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MDD: {best['max_dd']*100:.1f}% (vs BTC {best['btc_mdd']*100:.1f}%)")
    print(f"  Trades: {best['n_trades']}")
    print(f"  $5K -> ${best['final_value']:,.0f}")

    # Among those with MDD < BTC MDD
    safe_results = [r for r in results if r["max_dd"] < r["btc_mdd"]]
    if safe_results:
        best_safe = max(safe_results, key=lambda x: x["sharpe"])
        print(f"\n  BEST WITH MDD < BTC ({best_safe['btc_mdd']*100:.0f}%):")
        print(f"  max_exposure = {best_safe['max_exposure']:.0%}")
        print(f"  position_pct = {best_safe['position_pct']:.0%}")
        print(f"  trail_pct = {best_safe['trail_pct']:.0%}")
        print(f"  dd_trigger = {best_safe['dd_trigger']:.0%}")
        print(f"  rebalance_days = {best_safe['rebalance_days']}")
        print(f"  Return: {best_safe['total_return']*100:+.1f}%  Alpha: {best_safe['alpha']*100:+.1f}%")
        print(f"  Sharpe: {best_safe['sharpe']:.2f}  MDD: {best_safe['max_dd']*100:.1f}%")
        print(f"  $5K -> ${best_safe['final_value']:,.0f}")

    print(f"\n{'='*70}")
