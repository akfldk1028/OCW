"""Analyze trades.csv and produce a performance report.

Reads BUY/SELL pairs from the trade log, computes round-trip stats,
and prints a formatted text report.

Usage: python analyze_trades.py [path/to/trades.csv]
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Strip whitespace from string columns
    for col in ["action", "ticker", "reason", "regime", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def build_roundtrips(df: pd.DataFrame) -> pd.DataFrame:
    """Match BUY -> SELL pairs per ticker (FIFO)."""
    buys = {ticker: [] for ticker in df["ticker"].unique()}
    trips = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        if row["action"] == "BUY":
            buys[ticker].append(row)
        elif row["action"] == "SELL" and buys.get(ticker):
            buy = buys[ticker].pop(0)
            buy_val = buy["price"] * buy["qty"]
            sell_val = row["price"] * row["qty"]
            pnl_pct = (row["price"] / buy["price"] - 1) * 100
            trips.append({
                "buy_time": buy["timestamp"],
                "sell_time": row["timestamp"],
                "ticker": ticker,
                "buy_price": buy["price"],
                "sell_price": row["price"],
                "qty": min(buy["qty"], row["qty"]),
                "buy_val": buy_val,
                "sell_val": sell_val,
                "pnl_pct": pnl_pct,
                "pnl_usd": sell_val - buy_val,
                "held_hours": row.get("held_hours", 0),
                "reason": row.get("reason", ""),
                "regime": buy.get("regime", ""),
                "sell_regime": row.get("regime", ""),
            })

    if not trips:
        return pd.DataFrame()
    rt = pd.DataFrame(trips)
    rt["sell_month"] = rt["sell_time"].dt.to_period("M")
    return rt

def calc_sharpe(rt: pd.DataFrame) -> float:
    """Annualized Sharpe from daily PnL aggregation."""
    if rt.empty or len(rt) < 2:
        return 0.0
    daily = rt.set_index("sell_time").resample("D")["pnl_pct"].sum().dropna()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(365))

def calc_max_drawdown(rt: pd.DataFrame) -> tuple[float, str]:
    """Max drawdown from cumulative PnL curve."""
    if rt.empty:
        return 0.0, ""
    cum = rt.sort_values("sell_time")["pnl_pct"].cumsum()
    peak = cum.cummax()
    dd = cum - peak
    idx = dd.idxmin()
    return float(dd.min()), str(rt.loc[idx, "sell_time"].date()) if idx is not None else ""

def print_report(df: pd.DataFrame, rt: pd.DataFrame, csv_path: str):
    n_trades = len(df)
    n_rt = len(rt)
    open_buys = len(df[df["action"] == "BUY"]) - len(df[df["action"] == "SELL"])

    print("=" * 70)
    print("  TRADE LOG ANALYSIS")
    print("=" * 70)
    print(f"  Source:  {csv_path}")
    if not df.empty:
        print(f"  Period:  {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")
    print(f"  Records: {n_trades} rows ({n_rt} round-trips, {max(0, open_buys)} open)")
    print()

    if rt.empty:
        print("  No completed round-trip trades to analyze.")
        print("=" * 70)
        return

    wins = rt[rt["pnl_pct"] > 0]
    losses = rt[rt["pnl_pct"] <= 0]
    total_return = rt["pnl_pct"].sum()
    total_pnl_usd = rt["pnl_usd"].sum()
    win_rate = len(wins) / n_rt * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0
    gross_win = wins["pnl_usd"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl_usd"].sum()) if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    sharpe = calc_sharpe(rt)
    mdd, mdd_date = calc_max_drawdown(rt)

    # --- Summary ---
    print("  --- Summary ---")
    print(f"  Total Return:    {total_return:+.1f}%")
    print(f"  Total PnL:       ${total_pnl_usd:+,.2f}")
    print(f"  Sharpe Ratio:    {sharpe:.2f}")
    print(f"  Max Drawdown:    {mdd:.1f}% (on {mdd_date})")
    print()

    # --- Trade Stats ---
    print("  --- Trade Stats ---")
    print(f"  Round-trips:     {n_rt}")
    print(f"  Win Rate:        {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win:         {avg_win:+.1f}%")
    print(f"  Avg Loss:        {avg_loss:+.1f}%")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    if open_buys > 0:
        print(f"  Open Positions:  {open_buys}")
    print()

    # --- Per-Ticker ---
    print("  --- Per-Ticker Breakdown ---")
    for ticker, g in rt.groupby("ticker"):
        tw = g[g["pnl_pct"] > 0]
        print(f"  {ticker:<12s}  {len(g):>2d} trades  "
              f"PnL {g['pnl_pct'].sum():+6.1f}%  "
              f"WR {len(tw)/len(g)*100:.0f}%  "
              f"Avg {g['pnl_pct'].mean():+.1f}%")
    print()

    # --- Per-Regime ---
    regimes = rt[rt["regime"] != "nan"]
    if not regimes.empty:
        print("  --- Per-Regime Breakdown (entry regime) ---")
        for regime, g in regimes.groupby("regime"):
            tw = g[g["pnl_pct"] > 0]
            print(f"  {regime:<25s}  {len(g):>2d} trades  "
                  f"PnL {g['pnl_pct'].sum():+6.1f}%  "
                  f"WR {len(tw)/len(g)*100:.0f}%")
        print()

    # --- Monthly Returns ---
    print("  --- Monthly Returns ---")
    for month, g in rt.groupby("sell_month"):
        pnl = g["pnl_pct"].sum()
        bar_len = int(abs(pnl))
        bar = ("+" if pnl >= 0 else "-") * bar_len
        print(f"  {month}  {pnl:+6.1f}%  {bar}")
    print()

    # --- Best / Worst ---
    sorted_rt = rt.sort_values("pnl_pct", ascending=False)
    n_show = min(5, n_rt)
    print(f"  --- Top {n_show} Best Trades ---")
    for _, r in sorted_rt.head(n_show).iterrows():
        print(f"  {str(r['sell_time'].date())}  {r['ticker']:<12s}  "
              f"{r['pnl_pct']:+5.1f}%  "
              f"(${r['buy_price']:,.2f} -> ${r['sell_price']:,.2f})  "
              f"{str(r['reason'])[:50]}")
    print()
    print(f"  --- Top {n_show} Worst Trades ---")
    for _, r in sorted_rt.tail(n_show).iterrows():
        print(f"  {str(r['sell_time'].date())}  {r['ticker']:<12s}  "
              f"{r['pnl_pct']:+5.1f}%  "
              f"(${r['buy_price']:,.2f} -> ${r['sell_price']:,.2f})  "
              f"{str(r['reason'])[:50]}")

    print()
    print("=" * 70)

def main():
    default = Path(__file__).resolve().parent.parent / "data" / "logs" / "trades.csv"
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(default)

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = load_trades(csv_path)
    if df.empty:
        print("trades.csv is empty -- nothing to analyze.")
        sys.exit(0)

    rt = build_roundtrips(df)
    print_report(df, rt, csv_path)

if __name__ == "__main__":
    main()
