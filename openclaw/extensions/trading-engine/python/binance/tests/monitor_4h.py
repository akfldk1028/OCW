"""4-hour testnet monitoring script.

Runs alongside the persistent runner. Every 5 minutes:
- Checks if the runner is alive (log growing)
- Counts decisions, trades, safety exits
- Tracks portfolio value trajectory
- At the end: generates a summary report

Usage:
    python3 binance/tests/monitor_4h.py
"""
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows cp949 encoding
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Force unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

# Add parent dirs
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent))

LOGS_DIR = _HERE.parent / "logs"
DECISIONS_LOG = LOGS_DIR / "decisions.jsonl"
TRADES_CSV = LOGS_DIR / "trades.csv"
STATE_FILE = _HERE.parent / "data" / "runner_state.json"
REPORT_FILE = LOGS_DIR / "monitor_report_4h.txt"

POLL_INTERVAL = 300  # 5 min
TOTAL_DURATION = 4 * 3600  # 4 hours


def count_decisions_since(since_ts: float) -> dict:
    """Count decisions since timestamp."""
    total = 0
    trades = []
    holds = 0
    errors = 0
    next_checks = []
    assessments = []

    if not DECISIONS_LOG.exists():
        return {"total": 0, "trades": [], "holds": 0}

    with open(DECISIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts_str = d.get("timestamp", "")
                # Parse ISO timestamp
                from datetime import datetime as dt
                entry_dt = dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                entry_ts = entry_dt.timestamp()

                if entry_ts < since_ts:
                    continue

                total += 1

                if d.get("trades"):
                    for t in d["trades"]:
                        trades.append({
                            "time": ts_str[:19],
                            "action": t["action"],
                            "ticker": t["ticker"],
                            "price": t.get("price", 0),
                            "confidence": t.get("confidence", 0),
                        })
                else:
                    holds += 1

                nc = d.get("next_check_s", 0)
                if nc:
                    next_checks.append(nc)

                assess = d.get("claude_assessment", "")
                if assess:
                    assessments.append(assess[:100])

            except (json.JSONDecodeError, ValueError):
                errors += 1
                continue

    return {
        "total": total,
        "trades": trades,
        "holds": holds,
        "errors": errors,
        "avg_next_check": sum(next_checks) / len(next_checks) if next_checks else 0,
        "assessments": assessments[-3:],  # last 3
    }


def count_trades_csv_since(since_ts: float) -> list:
    """Count completed trades (sells) from CSV."""
    import csv
    results = []
    if not TRADES_CSV.exists():
        return results

    with open(TRADES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                continue
            try:
                from datetime import datetime as dt
                entry_dt = dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                if entry_dt.timestamp() < since_ts:
                    continue
                results.append({
                    "time": ts_str,
                    "action": row.get("action", ""),
                    "ticker": row.get("ticker", ""),
                    "price": row.get("price", ""),
                    "pnl_pct": row.get("pnl_pct", ""),
                    "reason": row.get("reason", ""),
                    "source": row.get("source", ""),
                })
            except (ValueError, TypeError):
                continue
    return results


def get_latest_log_activity() -> tuple:
    """Check the latest log file activity."""
    logs = sorted(LOGS_DIR.glob("trading_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        return 0, "NO_LOG"
    latest = logs[0]
    mtime = os.path.getmtime(latest)
    return mtime, latest.name


def get_positions() -> dict:
    """Read current positions from state file."""
    if not STATE_FILE.exists():
        return {}
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return state.get("entry_prices", {})
    except Exception:
        return {}


def main():
    start_time = time.time()
    start_dt = datetime.now(timezone.utc)
    snapshots = []

    print(f"[monitor] Starting 4h monitoring at {start_dt.strftime('%H:%M:%S UTC')}")
    print(f"[monitor] Will check every {POLL_INTERVAL}s for {TOTAL_DURATION // 3600}h")
    print(f"[monitor] Report: {REPORT_FILE}")
    print()

    initial_decisions = count_decisions_since(0)
    initial_decision_count = initial_decisions["total"]

    while time.time() - start_time < TOTAL_DURATION:
        now = time.time()
        elapsed = now - start_time
        elapsed_min = elapsed / 60

        # Check runner alive
        log_mtime, log_name = get_latest_log_activity()
        log_age = now - log_mtime if log_mtime > 0 else 999
        alive = log_age < 300  # log updated in last 5 min

        # Count new decisions/trades
        stats = count_decisions_since(start_time)
        csv_trades = count_trades_csv_since(start_time)
        positions = get_positions()

        sells = [t for t in csv_trades if t["action"] == "SELL"]
        buys = [t for t in csv_trades if t["action"] == "BUY"]
        safety_exits = [t for t in csv_trades if "safety" in t.get("source", "")]

        snapshot = {
            "elapsed_min": round(elapsed_min, 1),
            "alive": alive,
            "log_age_s": round(log_age, 0),
            "decisions": stats["total"],
            "holds": stats["holds"],
            "trade_decisions": len(stats["trades"]),
            "csv_buys": len(buys),
            "csv_sells": len(sells),
            "safety_exits": len(safety_exits),
            "positions": list(positions.keys()),
            "avg_next_check_s": round(stats["avg_next_check"], 0),
        }
        snapshots.append(snapshot)

        status = "ALIVE" if alive else "DEAD?"
        pos_str = ", ".join(positions.keys()) or "none"
        print(
            f"[{elapsed_min:6.1f}m] {status} | "
            f"decisions={stats['total']} (holds={stats['holds']}, trades={len(stats['trades'])}) | "
            f"CSV: {len(buys)}B/{len(sells)}S/{len(safety_exits)}safety | "
            f"positions: {pos_str} | "
            f"avg_next_check={stats['avg_next_check']:.0f}s"
        )

        # Print latest assessment if any
        if stats.get("assessments"):
            print(f"         Latest: {stats['assessments'][-1][:120]}")

        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n[monitor] Interrupted by user")
            break

    # === Generate Report ===
    end_time = time.time()
    end_dt = datetime.now(timezone.utc)
    total_min = (end_time - start_time) / 60

    final_stats = count_decisions_since(start_time)
    final_trades = count_trades_csv_since(start_time)
    final_positions = get_positions()

    buys = [t for t in final_trades if t["action"] == "BUY"]
    sells = [t for t in final_trades if t["action"] == "SELL"]
    safety_exits = [t for t in final_trades if "safety" in t.get("source", "")]

    # Calculate decisions per hour
    hours = total_min / 60
    decisions_per_hour = final_stats["total"] / hours if hours > 0 else 0

    # Check if runner stayed alive
    alive_checks = sum(1 for s in snapshots if s["alive"])
    total_checks = len(snapshots)
    uptime_pct = (alive_checks / total_checks * 100) if total_checks > 0 else 0

    report = []
    report.append("=" * 70)
    report.append("  4-HOUR TESTNET MONITORING REPORT")
    report.append(f"  Period: {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%H:%M')} UTC")
    report.append(f"  Duration: {total_min:.0f} minutes ({hours:.1f} hours)")
    report.append("=" * 70)
    report.append("")

    report.append("--- System Health ---")
    report.append(f"  Uptime:                {uptime_pct:.0f}% ({alive_checks}/{total_checks} checks alive)")
    report.append(f"  Total decisions:       {final_stats['total']}")
    report.append(f"  Decisions/hour:        {decisions_per_hour:.1f}")
    report.append(f"  HOLD decisions:        {final_stats['holds']}")
    report.append(f"  Trade decisions:       {len(final_stats['trades'])}")
    report.append(f"  Avg next_check_s:      {final_stats['avg_next_check']:.0f}s")
    report.append("")

    report.append("--- Trading Activity ---")
    report.append(f"  BUYs:                  {len(buys)}")
    report.append(f"  SELLs (agent):         {len([s for s in sells if 'safety' not in s.get('source', '')])}")
    report.append(f"  SELLs (safety):        {len(safety_exits)}")
    report.append(f"  Total round-trips:     {len(sells)}")
    report.append("")

    if buys:
        report.append("  --- BUY Details ---")
        for b in buys:
            report.append(f"    {b['time']} {b['action']} {b['ticker']} @ ${b['price']}")
    if sells:
        report.append("  --- SELL Details ---")
        for s in sells:
            report.append(
                f"    {s['time']} {s['action']} {s['ticker']} @ ${s['price']} "
                f"PnL={s['pnl_pct']} reason={s['reason'][:50]} source={s['source']}"
            )
    report.append("")

    report.append("--- Risk Management ---")
    report.append(f"  Safety exits:          {len(safety_exits)}")
    if safety_exits:
        for se in safety_exits:
            report.append(f"    {se['time']} {se['ticker']} reason={se['reason']}")
    else:
        report.append("    (no safety exits triggered)")
    report.append(f"  Open positions:        {list(final_positions.keys()) or 'none'}")
    report.append("")

    report.append("--- Agent Quality ---")
    if final_stats['total'] > 0:
        trade_rate = len(final_stats['trades']) / final_stats['total'] * 100
        report.append(f"  Trade rate:            {trade_rate:.1f}% ({len(final_stats['trades'])}/{final_stats['total']})")
    report.append(f"  Gate efficiency:       {decisions_per_hour:.1f} calls/hr (target: 2-6)")
    efficiency = "GOOD" if 1 <= decisions_per_hour <= 8 else "TOO_FREQUENT" if decisions_per_hour > 8 else "TOO_RARE"
    report.append(f"  Gate verdict:          {efficiency}")
    report.append("")

    if final_stats.get("assessments"):
        report.append("--- Last 3 Assessments ---")
        for a in final_stats["assessments"]:
            report.append(f"    {a}")
    report.append("")

    # PnL summary from sells
    pnls = []
    for s in sells:
        try:
            pnls.append(float(s["pnl_pct"]))
        except (ValueError, TypeError):
            pass
    if pnls:
        report.append("--- PnL Summary ---")
        report.append(f"  Completed trades:      {len(pnls)}")
        report.append(f"  Win rate:              {sum(1 for p in pnls if p > 0)}/{len(pnls)}")
        report.append(f"  Avg PnL:               {sum(pnls)/len(pnls):+.2%}")
        report.append(f"  Total PnL:             {sum(pnls):+.2%}")
        report.append("")

    report.append("--- Timeline ---")
    for s in snapshots:
        pos = ",".join(s["positions"]) or "-"
        report.append(
            f"  {s['elapsed_min']:6.1f}m | "
            f"{'OK' if s['alive'] else 'XX'} | "
            f"d={s['decisions']} t={s['trade_decisions']} | "
            f"B={s['csv_buys']} S={s['csv_sells']} safe={s['safety_exits']} | "
            f"{pos}"
        )
    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\n{'='*60}")
    print(report_text)
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
