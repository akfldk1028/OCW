"""1-year backtest using the exact Regime Blend pipeline from runner.py.

Simulates the full trading cycle day-by-day:
- Decision every 3 days (rebalance_days) via Pipeline nodes
- Risk check every day (trail stop + drawdown exit)
- Logs every trade, risk event, regime change
- Outputs detailed report to backtest_report.txt

Usage:
    python backtest_1y.py
"""
import sys
from datetime import timedelta
from pathlib import Path

_BINANCE_DIR = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _BINANCE_DIR.parent
for p in [str(_BINANCE_DIR), str(_PYTHON_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import yfinance as yf

from core.pipeline import (
    Pipeline, RegimeBlendDetectNode, RegimeBlendSignalNode,
    RegimeBlendExitNode, RegimeBlendEntryNode,
)
from crypto_config import REGIME_BLEND_CONFIG

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
START = "2025-02-01"
END = "2026-02-01"
INITIAL_CASH = 5_000.0
CFG = REGIME_BLEND_CONFIG
TX_COST = CFG["tx_cost"]  # 0.1%
TICKERS_YF = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD"}
REPORT_PATH = Path(__file__).resolve().parent / "backtest_report.txt"  # tests/backtest_report.txt


def run():
    lines = []  # report lines

    def log(msg: str):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  1-YEAR BACKTEST: Regime Blend Pipeline (runner.py logic)")
    log(f"  Period: {START} ~ {END}")
    log(f"  Initial: ${INITIAL_CASH:,.0f}")
    log(f"  Tickers: {list(TICKERS_YF.keys())}")
    log(f"  Params: exposure={CFG['max_exposure']}, position={CFG['position_pct']}, "
        f"trail={CFG['trail_pct']}, dd_trigger={CFG['dd_trigger']}")
    log("=" * 70)

    # Download data
    dl_start = (pd.Timestamp(START) - timedelta(days=150)).strftime("%Y-%m-%d")
    yf_tickers = list(TICKERS_YF.values())
    data = yf.download(yf_tickers, start=dl_start, end=END, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        close_raw = data["Close"]
    else:
        close_raw = data

    # Rename columns to ccxt format
    close = close_raw.rename(columns={v: k for k, v in TICKERS_YF.items()})
    close = close.dropna()

    all_dates = close.index
    bt_start_idx = all_dates.get_indexer([pd.Timestamp(START)], method="pad")[0]
    bt_dates = all_dates[bt_start_idx:]

    log(f"\nData: {len(close)} total bars, backtesting {len(bt_dates)} days")
    log(f"BTC range: ${close['BTC/USDT'].min():,.0f} ~ ${close['BTC/USDT'].max():,.0f}")
    log("")

    # State
    cash = INITIAL_CASH
    positions = {}       # {ticker: {qty, entry_price, entry_date}}
    trailing_highs = {}  # {ticker: highest_price}
    trade_log = []
    daily_pv = []
    daily_dates = []
    regime_history = []

    # BTC buy & hold benchmark
    btc_start_price = float(close["BTC/USDT"].iloc[bt_start_idx])

    log("-" * 70)
    log(f"{'DATE':<12} {'EVENT':<15} {'TICKER':<10} {'SIDE':<5} "
        f"{'PRICE':>10} {'QTY':>10} {'PnL':>8} {'REASON'}")
    log("-" * 70)

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)
        date_str = str(date)[:10]

        # ---- Daily: update prices, check risk ----
        pv = cash
        risk_exits = []

        for tic in list(positions.keys()):
            if tic not in close.columns:
                continue
            px = float(close[tic].iloc[date_idx])
            if pd.isna(px) or px <= 0:
                continue

            pos = positions[tic]
            entry_price = pos["entry_price"]

            # Update trailing high
            prev_high = trailing_highs.get(tic, entry_price)
            if px > prev_high:
                trailing_highs[tic] = px
                prev_high = px

            pnl_pct = (px - entry_price) / entry_price
            reason = None

            # Trailing stop (same logic as runner._run_risk)
            if prev_high > entry_price * (1 + CFG["trail_activation_pct"]):
                trail_stop = prev_high * (1 - CFG["trail_pct"])
                if px <= trail_stop:
                    reason = f"trail_stop (high={prev_high:.0f})"

            # Drawdown exit
            if reason is None and pnl_pct < -CFG["dd_trigger"]:
                reason = f"dd_exit"

            if reason:
                risk_exits.append((tic, px, pnl_pct, reason))

            pv += pos["qty"] * px

        # Execute risk exits
        for tic, px, pnl_pct, reason in risk_exits:
            pos = positions[tic]
            proceeds = pos["qty"] * px * (1 - TX_COST)
            cash += proceeds
            trade_log.append({
                "date": date_str, "ticker": tic, "side": "SELL",
                "price": px, "qty": pos["qty"], "pnl_pct": pnl_pct,
                "reason": f"RISK: {reason}", "entry_price": pos["entry_price"],
            })
            log(f"{date_str:<12} {'RISK EXIT':<15} {tic:<10} {'SELL':<5} "
                f"${px:>9,.0f} {pos['qty']:>10.4f} {pnl_pct:>+7.1%} {reason}")
            del positions[tic]
            trailing_highs.pop(tic, None)

        # Portfolio drawdown check (Talyxion-style)
        if len(daily_pv) > 20:
            peak_pv = max(daily_pv[-20:])
            current_dd = (peak_pv - pv) / peak_pv
            if current_dd > CFG.get("portfolio_dd_trigger", CFG["dd_trigger"]) and positions:
                for tic in list(positions.keys()):
                    pos = positions[tic]
                    px = float(close[tic].iloc[date_idx])
                    pnl_pct = (px - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["qty"] * px * (1 - TX_COST)
                    trade_log.append({
                        "date": date_str, "ticker": tic, "side": "SELL",
                        "price": px, "qty": pos["qty"], "pnl_pct": pnl_pct,
                        "reason": f"PORTFOLIO DD {current_dd:.1%}",
                        "entry_price": pos["entry_price"],
                    })
                    log(f"{date_str:<12} {'PORTFOLIO DD':<15} {tic:<10} {'SELL':<5} "
                        f"${px:>9,.0f} {pos['qty']:>10.4f} {pnl_pct:>+7.1%} "
                        f"portfolio_dd={current_dd:.1%}")
                    del positions[tic]
                    trailing_highs.pop(tic, None)

        # Recalc PV after exits
        pv = cash + sum(
            positions[t]["qty"] * float(close[t].iloc[date_idx])
            for t in positions if t in close.columns
        )
        daily_pv.append(pv)
        daily_dates.append(date)

        # ---- Decision: every rebalance_days, run pipeline ----
        if day_i % CFG["rebalance_days"] != 0:
            continue

        # Build close_df for pipeline (lookback window)
        lookback = min(date_idx + 1, CFG["ohlcv_lookback_days"])
        close_window = close.iloc[date_idx - lookback + 1: date_idx + 1]

        pipe = Pipeline([
            RegimeBlendDetectNode(),
            RegimeBlendSignalNode(),
            RegimeBlendExitNode(),
            RegimeBlendEntryNode(),
        ])

        # Build positions for pipeline
        pipe_positions = {}
        for tic, pos in positions.items():
            pipe_positions[tic] = {
                "qty": pos["qty"],
                "entry_price": pos["entry_price"],
                "entry_date": pos["entry_date"],
            }

        ctx = pipe.run({
            "crypto_close": close_window,
            "eval_date": date,
            "btc_ticker": CFG["tickers"][0],
            "candidates": [t for t in CFG["tickers"] if t in close_window.columns],
            "positions": pipe_positions,
            "cash": cash,
            "trailing_highs": dict(trailing_highs),
            "rb_config": CFG,
        })

        regime = ctx.get("regime", "unknown")
        regime_history.append((date_str, regime))

        # Sync trailing highs from pipeline
        trailing_highs.clear()
        trailing_highs.update(ctx.get("trailing_highs", {}))

        # Execute trades from pipeline
        for trade in ctx.get("trade_log", []):
            tic = trade["ticker"]
            side = trade["side"]
            px = trade["price"]
            qty = trade["qty"]

            if side == "BUY" and tic not in positions:
                cost = qty * px * (1 + TX_COST)
                if cost > cash:
                    qty = cash / (px * (1 + TX_COST))
                if qty <= 0:
                    continue
                cash -= qty * px * (1 + TX_COST)
                positions[tic] = {
                    "qty": qty, "entry_price": px, "entry_date": date_str,
                }
                trailing_highs[tic] = px
                reason = trade.get("reason", regime)
                trade_log.append({
                    "date": date_str, "ticker": tic, "side": "BUY",
                    "price": px, "qty": qty, "pnl_pct": 0,
                    "reason": f"{regime}: {reason}",
                    "entry_price": px,
                })
                log(f"{date_str:<12} {'PIPELINE BUY':<15} {tic:<10} {'BUY':<5} "
                    f"${px:>9,.0f} {qty:>10.4f} {'':>8} {regime}: {reason}")

            elif side == "SELL" and tic in positions:
                pos = positions[tic]
                pnl_pct = (px - pos["entry_price"]) / pos["entry_price"]
                cash += pos["qty"] * px * (1 - TX_COST)
                reason = trade.get("reason", regime)
                trade_log.append({
                    "date": date_str, "ticker": tic, "side": "SELL",
                    "price": px, "qty": pos["qty"], "pnl_pct": pnl_pct,
                    "reason": f"{regime}: {reason}",
                    "entry_price": pos["entry_price"],
                })
                log(f"{date_str:<12} {'PIPELINE SELL':<15} {tic:<10} {'SELL':<5} "
                    f"${px:>9,.0f} {pos['qty']:>10.4f} {pnl_pct:>+7.1%} {regime}: {reason}")
                del positions[tic]
                trailing_highs.pop(tic, None)

    log("-" * 70)

    # ---------------------------------------------------------------- #
    # Performance Report
    # ---------------------------------------------------------------- #
    pv_series = pd.Series(daily_pv, index=daily_dates)
    btc_series = close["BTC/USDT"].loc[daily_dates]
    btc_bh_value = INITIAL_CASH * btc_series / btc_start_price

    total_return = (pv_series.iloc[-1] / INITIAL_CASH - 1)
    btc_return = (btc_series.iloc[-1] / btc_start_price - 1)
    alpha = total_return - btc_return

    daily_ret = pv_series.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * (365 ** 0.5) if daily_ret.std() > 0 else 0

    running_max = pv_series.cummax()
    drawdowns = (pv_series - running_max) / running_max
    max_dd = drawdowns.min()
    max_dd_date = str(drawdowns.idxmin())[:10]

    btc_running_max = btc_bh_value.cummax()
    btc_dd = ((btc_bh_value - btc_running_max) / btc_running_max).min()

    wins = [t for t in trade_log if t["side"] == "SELL" and t["pnl_pct"] > 0]
    losses = [t for t in trade_log if t["side"] == "SELL" and t["pnl_pct"] <= 0]
    total_sells = len(wins) + len(losses)
    win_rate = len(wins) / total_sells if total_sells > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    # Regime breakdown
    regime_counts = {}
    for _, r in regime_history:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    # Risk events
    risk_trades = [t for t in trade_log if "RISK" in t.get("reason", "") or "DD" in t.get("reason", "")]

    log("")
    log("=" * 70)
    log("  PERFORMANCE REPORT")
    log("=" * 70)
    log("")
    log(f"  Period:           {START} ~ {END} ({len(bt_dates)} days)")
    log(f"  Initial Capital:  ${INITIAL_CASH:,.0f}")
    log(f"  Final Value:      ${pv_series.iloc[-1]:,.0f}")
    log(f"  Total Return:     {total_return:+.1%}")
    log(f"  BTC Buy & Hold:   {btc_return:+.1%}")
    log(f"  Alpha vs BTC:     {alpha:+.1%}")
    log(f"  Sharpe Ratio:     {sharpe:.2f}")
    log(f"  Max Drawdown:     {max_dd:.1%} (on {max_dd_date})")
    log(f"  BTC Max Drawdown: {btc_dd:.1%}")
    log("")
    log("  --- Trade Stats ---")
    log(f"  Total Trades:     {len(trade_log)} ({total_sells} round-trips)")
    log(f"  Win Rate:         {win_rate:.0%} ({len(wins)}W / {len(losses)}L)")
    log(f"  Avg Win:          {avg_win:+.1%}")
    log(f"  Avg Loss:         {avg_loss:+.1%}")
    log(f"  Risk Exits:       {len(risk_trades)} (trail stops + drawdown)")
    log("")
    log("  --- Regime Breakdown ---")
    for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
        log(f"  {r:<12} {cnt:>4} days ({cnt/len(regime_history)*100:.0f}%)")
    log("")

    # Top 5 best/worst trades
    sells = [t for t in trade_log if t["side"] == "SELL"]
    if sells:
        sells_sorted = sorted(sells, key=lambda t: t["pnl_pct"], reverse=True)
        log("  --- Top 5 Best Trades ---")
        for t in sells_sorted[:5]:
            log(f"  {t['date']} {t['ticker']:<10} {t['pnl_pct']:>+7.1%} "
                f"(${t.get('entry_price',0):,.0f} -> ${t['price']:,.0f}) {t['reason']}")
        log("")
        log("  --- Top 5 Worst Trades ---")
        for t in sells_sorted[-5:]:
            log(f"  {t['date']} {t['ticker']:<10} {t['pnl_pct']:>+7.1%} "
                f"(${t.get('entry_price',0):,.0f} -> ${t['price']:,.0f}) {t['reason']}")

    # Monthly returns
    log("")
    log("  --- Monthly Returns ---")
    monthly = pv_series.resample("ME").last().pct_change().dropna()
    btc_monthly = btc_bh_value.resample("ME").last().pct_change().dropna()
    for m_date in monthly.index:
        m_str = str(m_date)[:7]
        strat_r = monthly.loc[m_date]
        btc_r = btc_monthly.loc[m_date] if m_date in btc_monthly.index else 0
        bar = "+" * int(max(strat_r * 100, 0)) + "-" * int(max(-strat_r * 100, 0))
        log(f"  {m_str}  Strategy: {strat_r:>+6.1%}  BTC: {btc_r:>+6.1%}  {bar}")

    log("")
    log("=" * 70)

    # Save report
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run()
