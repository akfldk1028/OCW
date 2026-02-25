"""Claude Agent simulation backtest engine.

Feeds historical MarketSnapshots to Claude via claude-agent-sdk,
records decisions, simulates execution, and calculates performance metrics.

Usage:
    python backtests/backtest_claude/simulator.py [--bars N] [--start-bar N]

    --bars N        : Number of bars to simulate (default: all)
    --start-bar N   : Skip first N bars (for warm-up, default: 72)
    --rebal N       : Rebalance every N bars (default: 2 = every 8h)
    --dry           : Don't call Claude — just build snapshots and print
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

import importlib.util

from fetch_data import fetch_and_cache, TICKERS, START
from snapshot_builder import build_snapshot_at

# Direct module import to avoid core/__init__.py pulling in fastapi/websockets
_spec = importlib.util.spec_from_file_location(
    "claude_agent_mod", _ROOT / "core" / "claude_agent.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["claude_agent_mod"] = _mod
_spec.loader.exec_module(_mod)
ClaudeAgent = _mod.ClaudeAgent
MarketSnapshot = _mod.MarketSnapshot
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT

logger = logging.getLogger("backtest_claude")

# Defaults
INITIAL_CASH = 5_000.0
TX_COST = 0.001  # 0.1% per trade
MAX_POSITION_PCT = 0.30
REPORT_DIR = Path(__file__).resolve().parent / "reports"


class SimulationEngine:
    """Replays historical data through Claude Agent and tracks PnL."""

    def __init__(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        rebal_bars: int = 2,
        initial_cash: float = INITIAL_CASH,
    ) -> None:
        self.ticker_data = ticker_data
        self.rebal_bars = rebal_bars
        self.initial_cash = initial_cash

        # State
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trailing_highs: Dict[str, float] = {}

        # Claude Agent
        self.claude_agent = ClaudeAgent(timeout=120.0)
        self.agent_memory = ""

        # Tracking
        self.trades: List[Dict] = []
        self.decisions_log: List[Dict] = []
        self.daily_pvs: List[float] = []
        self.daily_dates: List = []

        # Risk params (same as live)
        self.dd_trigger = 0.15
        self.trail_activation_pct = 0.08
        self.trail_pct = 0.12

        # Stats
        self.claude_calls = 0
        self.claude_failures = 0

    def portfolio_value(self, bar_time: pd.Timestamp) -> float:
        pv = self.cash
        for tic, pos in self.positions.items():
            df = self.ticker_data.get(tic)
            if df is None:
                continue
            available = df[df.index <= bar_time]
            if available.empty:
                continue
            pv += pos["qty"] * float(available["close"].iloc[-1])
        return pv

    def _get_price(self, tic: str, bar_time: pd.Timestamp) -> float:
        df = self.ticker_data.get(tic)
        if df is None:
            return 0.0
        available = df[df.index <= bar_time]
        if available.empty:
            return 0.0
        return float(available["close"].iloc[-1])

    def _risk_check(self, bar_time: pd.Timestamp) -> None:
        """Check trail stop and drawdown exits every bar."""
        for tic in list(self.positions.keys()):
            px = self._get_price(tic, bar_time)
            if px <= 0:
                continue
            pos = self.positions[tic]
            pnl = (px - pos["entry_price"]) / pos["entry_price"]

            prev_high = self.trailing_highs.get(tic, pos["entry_price"])
            if px > prev_high:
                self.trailing_highs[tic] = px
                prev_high = px

            reason = None
            if prev_high > pos["entry_price"] * (1 + self.trail_activation_pct):
                trail_stop = prev_high * (1 - self.trail_pct)
                if px <= trail_stop:
                    reason = "trail_stop"
            if reason is None and pnl < -self.dd_trigger:
                reason = "dd_exit"

            if reason:
                proceeds = pos["qty"] * px * (1 - TX_COST)
                self.cash += proceeds
                self.trades.append({
                    "time": str(bar_time), "ticker": tic, "side": "SELL",
                    "price": px, "pnl_pct": pnl, "reason": reason,
                })
                del self.positions[tic]
                self.trailing_highs.pop(tic, None)

    async def _call_claude(self, snapshot: MarketSnapshot) -> Dict | None:
        """Call Claude and parse response."""
        self.claude_calls += 1
        result = await self.claude_agent.decide(snapshot)
        if result is None:
            self.claude_failures += 1
        return result

    def _execute_decisions(
        self,
        decisions: List[Dict],
        bar_time: pd.Timestamp,
        snapshot: MarketSnapshot,
    ) -> None:
        """Simulate order execution from Claude's decisions."""
        for d in decisions:
            action = d.get("action", "HOLD").upper()
            ticker = d.get("ticker", "")
            # Normalize futures ticker (BTC/USDT:USDT → BTC/USDT)
            if ":" in ticker:
                ticker = ticker.split(":")[0]
            if action == "HOLD":
                continue

            px = self._get_price(ticker, bar_time)
            if px <= 0:
                continue

            if action == "BUY" and ticker not in self.positions:
                position_pct = min(d.get("position_pct", 0.10), MAX_POSITION_PCT)
                alloc = self.cash * position_pct
                if alloc < 10:
                    continue
                qty = alloc / (px * (1 + TX_COST))
                self.cash -= qty * px * (1 + TX_COST)
                self.positions[ticker] = {"qty": qty, "entry_price": px}
                self.trailing_highs[ticker] = px
                self.trades.append({
                    "time": str(bar_time), "ticker": ticker, "side": "BUY",
                    "price": px, "pnl_pct": 0, "reason": d.get("reasoning", "")[:100],
                    "confidence": d.get("confidence", 0),
                })

            elif action == "SELL" and ticker in self.positions:
                pos = self.positions[ticker]
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                proceeds = pos["qty"] * px * (1 - TX_COST)
                self.cash += proceeds
                self.trades.append({
                    "time": str(bar_time), "ticker": ticker, "side": "SELL",
                    "price": px, "pnl_pct": pnl, "reason": d.get("reasoning", "")[:100],
                    "confidence": d.get("confidence", 0),
                })
                del self.positions[ticker]
                self.trailing_highs.pop(ticker, None)

    async def run(
        self,
        start_bar: int = 72,
        max_bars: int | None = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run the simulation."""
        ref = list(self.ticker_data.values())[0]
        bt_start = pd.Timestamp(START)
        bt_idx = ref.index[ref.index >= bt_start]

        if max_bars:
            bt_idx = bt_idx[:start_bar + max_bars]

        total_bars = len(bt_idx) - start_bar
        logger.info("Simulating %d bars (%d rebal points, rebal_bars=%d)",
                     total_bars, total_bars // self.rebal_bars, self.rebal_bars)

        last_day = None
        recent_trades_window: List[Dict] = []

        for bar_i, bar_time in enumerate(bt_idx):
            if bar_i < start_bar:
                continue

            # Daily PV snapshot
            bar_date = bar_time.date()
            if bar_date != last_day:
                last_day = bar_date
                pv = self.portfolio_value(bar_time)
                self.daily_pvs.append(pv)
                self.daily_dates.append(bar_date)

            # Risk check every bar
            self._risk_check(bar_time)

            # Rebalance interval
            if (bar_i - start_bar) % self.rebal_bars != 0:
                continue

            pv = self.portfolio_value(bar_time)

            snapshot = build_snapshot_at(
                bar_time=bar_time,
                ticker_data=self.ticker_data,
                positions=dict(self.positions),
                cash=self.cash,
                portfolio_value=pv,
                trigger="candle_close",
                agent_memory=self.agent_memory,
                recent_trades=recent_trades_window[-10:],
            )

            progress = (bar_i - start_bar) / total_bars * 100
            logger.info("[%5.1f%%] %s | PV=$%.0f | pos=%d | claude_calls=%d",
                        progress, bar_time.strftime("%Y-%m-%d %H:%M"),
                        pv, len(self.positions), self.claude_calls)

            if dry_run:
                continue

            # Call Claude
            result = await self._call_claude(snapshot)

            if result and "decisions" in result:
                self.decisions_log.append({
                    "time": str(bar_time),
                    "assessment": result.get("market_assessment", ""),
                    "decisions": result.get("decisions", []),
                    "next_check": result.get("next_check_seconds"),
                    "memory": result.get("memory_update", ""),
                })

                self._execute_decisions(result["decisions"], bar_time, snapshot)

                # Update agent memory
                mem = result.get("memory_update", "")
                if mem:
                    self.agent_memory = mem[:500]

                # Track recent trades for context
                for t in self.trades[-5:]:
                    if t not in recent_trades_window:
                        recent_trades_window.append(t)
                recent_trades_window = recent_trades_window[-20:]

                # Rate limit (Pro/Max subscription still has limits)
                await asyncio.sleep(5.0)
            else:
                logger.warning("[%s] Claude unavailable, skipping", bar_time)
                await asyncio.sleep(1.0)

        # Final PV
        final_pv = self.portfolio_value(bt_idx[-1]) if len(bt_idx) > 0 else self.cash
        return self._compute_metrics(final_pv)

    def _compute_metrics(self, final_pv: float) -> Dict[str, Any]:
        """Compute backtest performance metrics."""
        if not self.daily_pvs:
            return {"error": "No data"}

        pvs = np.array(self.daily_pvs)
        returns = np.diff(pvs) / pvs[:-1]

        total_return = (final_pv / self.initial_cash) - 1
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if np.std(returns) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(pvs)
        dd = (pvs - peak) / peak
        mdd = float(np.min(dd))

        # Trade stats
        sell_trades = [t for t in self.trades if t["side"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl_pct", 0) <= 0]
        risk_exits = [t for t in sell_trades if t.get("reason", "") in ("trail_stop", "dd_exit")]

        avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0
        avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0

        # BTC B&H comparison
        btc_df = self.ticker_data.get("BTC/USDT")
        btc_bh = 0.0
        if btc_df is not None and not btc_df.empty:
            btc_start = float(btc_df[btc_df.index >= pd.Timestamp(START)]["close"].iloc[0])
            btc_end = float(btc_df["close"].iloc[-1])
            btc_bh = btc_end / btc_start - 1

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "mdd": mdd,
            "alpha_vs_btc": total_return - btc_bh,
            "btc_bh_return": btc_bh,
            "total_trades": len(self.trades),
            "sell_trades": len(sell_trades),
            "win_rate": len(wins) / len(sell_trades) if sell_trades else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "risk_exits": len(risk_exits),
            "risk_exit_pct": len(risk_exits) / len(sell_trades) if sell_trades else 0,
            "claude_calls": self.claude_calls,
            "claude_failures": self.claude_failures,
            "final_pv": final_pv,
            "initial_cash": self.initial_cash,
            "days": len(self.daily_pvs),
        }


def format_report(metrics: Dict, trades: List[Dict], decisions: List[Dict]) -> str:
    """Format backtest results as text report."""
    lines = [
        "=" * 60,
        "  CLAUDE AGENT SIMULATION BACKTEST",
        "=" * 60,
        f"  Period: {START} ~ (days={metrics['days']})",
        f"  Initial: ${metrics['initial_cash']:,.0f} -> Final: ${metrics['final_pv']:,.0f}",
        "",
        "  --- Performance ---",
        f"  Return:     {metrics['total_return']:+.1%}",
        f"  Sharpe:     {metrics['sharpe']:.2f}",
        f"  MDD:        {metrics['mdd']:.1%}",
        f"  Alpha(BTC): {metrics['alpha_vs_btc']:+.1%} (BTC B&H: {metrics['btc_bh_return']:+.1%})",
        "",
        "  --- Trades ---",
        f"  Total:      {metrics['total_trades']}",
        f"  Win rate:   {metrics['win_rate']:.0%} ({metrics['sell_trades']} sells)",
        f"  Avg Win:    {metrics['avg_win']:+.1%}",
        f"  Avg Loss:   {metrics['avg_loss']:+.1%}",
        f"  Risk exits: {metrics['risk_exits']} ({metrics['risk_exit_pct']:.0%})",
        "",
        "  --- Claude ---",
        f"  Calls:      {metrics['claude_calls']}",
        f"  Failures:   {metrics['claude_failures']}",
        "=" * 60,
        "",
        "TRADE LOG:",
    ]
    for t in trades:
        lines.append(f"  {t['time'][:16]} {t['side']:4s} {t['ticker']:15s} "
                      f"@ ${t['price']:,.2f}  PnL={t.get('pnl_pct',0):+.1%}  {t.get('reason','')[:50]}")

    lines.append("")
    lines.append("CLAUDE DECISIONS (sample):")
    for d in decisions[:20]:
        lines.append(f"  {d['time'][:16]}: {d.get('assessment', '')[:80]}")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Claude Agent Simulation Backtest")
    parser.add_argument("--bars", type=int, default=None, help="Max bars to simulate")
    parser.add_argument("--start-bar", type=int, default=72, help="Warm-up bars to skip")
    parser.add_argument("--rebal", type=int, default=2, help="Rebalance every N bars (2=8h)")
    parser.add_argument("--dry", action="store_true", help="Dry run (no Claude calls)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Fetching historical data...")
    ticker_data = fetch_and_cache()
    if not ticker_data:
        logger.error("No data fetched")
        return

    engine = SimulationEngine(
        ticker_data=ticker_data,
        rebal_bars=args.rebal,
    )

    if not args.dry and not engine.claude_agent.is_available:
        logger.error("Claude Agent not available. Run with --dry to test without Claude.")
        logger.error("Or ensure claude-agent-sdk is installed and OAuth token is configured.")
        return

    logger.info("Starting simulation (dry=%s)...", args.dry)
    metrics = await engine.run(
        start_bar=args.start_bar,
        max_bars=args.bars,
        dry_run=args.dry,
    )

    # Save reports first (before print to avoid encoding crash losing data)
    report = format_report(metrics, engine.trades, engine.decisions_log)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = REPORT_DIR / f"claude_backtest_{timestamp}.json"
    json_path.write_text(json.dumps({
        "metrics": metrics,
        "trades": engine.trades,
        "decisions": engine.decisions_log,
    }, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    logger.info("Raw data saved: %s", json_path)

    report_path = REPORT_DIR / f"claude_backtest_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved: %s", report_path)

    # Print to console
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + report)


if __name__ == "__main__":
    asyncio.run(main())
