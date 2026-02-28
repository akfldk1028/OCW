"use client";

import { useStatus, useEquityCurve, useTrades } from "@/lib/hooks";
import { clsx } from "clsx";

function formatUSD(n: number) {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function StatBox({
  label,
  value,
  sub,
  color,
  glow,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-xl bg-[var(--card)] border border-[var(--border)] p-4 transition-all",
        glow && color === "green" && "glow-green",
        glow && color === "red" && "glow-red"
      )}
    >
      <p className="text-xs text-[var(--muted)] mb-1 uppercase tracking-wider">
        {label}
      </p>
      <p
        className={clsx(
          "text-2xl md:text-3xl font-bold font-mono tracking-tight",
          color === "green" && "text-[var(--green)]",
          color === "red" && "text-[var(--red)]",
          !color && "text-[var(--text)]"
        )}
      >
        {value}
      </p>
      {sub && (
        <p className="text-xs text-[var(--muted)] mt-1 font-mono">{sub}</p>
      )}
    </div>
  );
}

export default function HeroStats() {
  const { data: status } = useStatus();
  const { data: equity } = useEquityCurve();
  const { data: trades } = useTrades();

  const portfolioValue = status?.portfolio_value ?? 0;
  const realizedPnlUsd = status?.realized_pnl_usd ?? 0;
  const todayPnlPct = status?.today_pnl_pct ?? 0;
  const todayTrades = status?.today_trades ?? 0;

  // Total PnL from actual trade results (not equity curve which breaks in dry-run)
  const summary = trades?.summary;
  const totalPnl = summary?.cumulative_pnl_pct ?? 0; // API already returns as % (e.g. -1.9)
  const isUp = totalPnl >= 0;

  // Unrealized PnL from open positions
  const positions = status?.positions ?? {};
  const unrealizedPnl = Object.values(positions).reduce(
    (sum, p) => sum + (p.pnl_pct ?? 0),
    0
  );
  const posCount = Object.keys(positions).length;

  // Win rate from trades (summary already declared above)
  const winRate = summary?.win_rate ?? 0;
  const wins = summary?.wins ?? 0;
  const losses = summary?.losses ?? 0;
  const totalTrades = summary?.total_trades ?? 0;

  // Max drawdown from equity curve
  const points = equity?.points ?? [];
  const values = points.map((p: { value: number }) => p.value);
  const peak = values.length > 0 ? Math.max(...values) : 0;
  const trough = values.length > 0 ? Math.min(...values) : 0;
  const maxDD = peak > 0 ? ((peak - trough) / peak) * 100 : 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <StatBox
        label="Portfolio Value"
        value={formatUSD(portfolioValue)}
        sub={`${posCount} position${posCount !== 1 ? "s" : ""} open`}
      />
      <StatBox
        label="Total PnL"
        value={`${isUp ? "+" : ""}${totalPnl.toFixed(2)}%`}
        sub={`${realizedPnlUsd >= 0 ? "+" : ""}$${Math.abs(realizedPnlUsd).toLocaleString(undefined, { maximumFractionDigits: 0 })} realized`}
        color={isUp ? "green" : "red"}
        glow
      />
      <StatBox
        label="Today PnL"
        value={`${todayPnlPct >= 0 ? "+" : ""}${todayPnlPct.toFixed(2)}%`}
        sub={`${todayTrades} trade${todayTrades !== 1 ? "s" : ""} · ${posCount} open`}
        color={todayPnlPct > 0 ? "green" : todayPnlPct < 0 ? "red" : undefined}
        glow={Math.abs(todayPnlPct) > 0.5}
      />
      <StatBox
        label="Win Rate"
        value={`${(winRate * 100).toFixed(0)}%`}
        sub={`${wins}W / ${losses}L (${totalTrades} trades) · MDD: -${maxDD.toFixed(1)}%`}
        color={winRate >= 0.5 ? "green" : winRate > 0 ? "red" : undefined}
      />
    </div>
  );
}
