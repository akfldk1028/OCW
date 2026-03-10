"use client";

import { useDailyPnl } from "@/lib/hooks";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Line,
  ComposedChart,
} from "recharts";

export default function DailyPerf() {
  const { data, isLoading } = useDailyPnl();

  if (isLoading) return <div className="card animate-pulse h-96" />;

  const days = data?.days ?? [];
  if (days.length === 0) {
    return (
      <div className="card flex flex-col items-center justify-center h-96">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Daily Performance
        </h2>
        <p className="text-xs text-[var(--muted)] mt-1">No data yet</p>
      </div>
    );
  }

  const chartData = days.map((d) => ({
    date: d.date.slice(5), // "03-04"
    pnl: d.pnl_pct,
    winRate: Math.round(d.win_rate * 100),
    trades: d.trades,
    wins: d.wins,
    losses: d.losses,
    cumPnl: d.cumulative_pnl_pct,
    best: d.best_trade_pct,
    worst: d.worst_trade_pct,
  }));

  const summary = data?.summary;
  const latestDay = days[days.length - 1];
  const prevDay = days.length > 1 ? days[days.length - 2] : null;
  const wrImproved =
    prevDay && latestDay.win_rate > prevDay.win_rate;
  const pnlImproved =
    prevDay && latestDay.pnl_pct > prevDay.pnl_pct;

  return (
    <div className="card">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-sm font-medium text-[var(--muted)] mb-3">
          Daily Performance
        </h2>

        {/* Day-by-day table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-[var(--muted)] border-b border-[var(--border)]">
                <th className="text-left py-1.5 pr-3">Date</th>
                <th className="text-right py-1.5 px-2">Trades</th>
                <th className="text-right py-1.5 px-2">W/L</th>
                <th className="text-right py-1.5 px-2">WR%</th>
                <th className="text-right py-1.5 px-2">PnL</th>
                <th className="text-right py-1.5 pl-2">Cum.</th>
              </tr>
            </thead>
            <tbody>
              {days.map((d, i) => {
                const wr = Math.round(d.win_rate * 100);
                const prevWr = i > 0 ? Math.round(days[i - 1].win_rate * 100) : null;
                const wrUp = prevWr !== null && wr > prevWr;
                const wrDown = prevWr !== null && wr < prevWr;
                return (
                  <tr
                    key={d.date}
                    className="border-b border-[var(--border)]/50"
                  >
                    <td className="py-1.5 pr-3 text-[var(--foreground)]">
                      {d.date.slice(5)}
                    </td>
                    <td className="text-right py-1.5 px-2">{d.trades}</td>
                    <td className="text-right py-1.5 px-2">
                      <span className="text-[var(--green)]">{d.wins}</span>
                      /
                      <span className="text-[var(--red)]">{d.losses}</span>
                    </td>
                    <td className="text-right py-1.5 px-2">
                      <span
                        className={
                          wrUp
                            ? "text-[var(--green)]"
                            : wrDown
                            ? "text-[var(--red)]"
                            : ""
                        }
                      >
                        {wr}%
                        {wrUp ? " +" : wrDown ? " -" : ""}
                      </span>
                    </td>
                    <td
                      className={`text-right py-1.5 px-2 font-medium ${
                        d.pnl_pct >= 0
                          ? "text-[var(--green)]"
                          : "text-[var(--red)]"
                      }`}
                    >
                      {d.pnl_pct >= 0 ? "+" : ""}
                      {d.pnl_pct.toFixed(2)}%
                    </td>
                    <td
                      className={`text-right py-1.5 pl-2 ${
                        d.cumulative_pnl_pct >= 0
                          ? "text-[var(--green)]"
                          : "text-[var(--red)]"
                      }`}
                    >
                      {d.cumulative_pnl_pct >= 0 ? "+" : ""}
                      {d.cumulative_pnl_pct.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Trend indicators */}
        {prevDay && (
          <div className="flex gap-3 mt-3 text-xs">
            <span
              className={
                pnlImproved
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              PnL {pnlImproved ? "improving" : "declining"}
            </span>
            <span
              className={
                wrImproved
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              WR {wrImproved ? "improving" : "declining"}
            </span>
          </div>
        )}
      </div>

      {/* Bar chart: daily PnL + win rate line */}
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chartData}>
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#71717a" }}
          />
          <YAxis
            yAxisId="pnl"
            tick={{ fontSize: 10, fill: "#71717a" }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            width={45}
          />
          <YAxis
            yAxisId="wr"
            orientation="right"
            tick={{ fontSize: 10, fill: "#71717a" }}
            tickFormatter={(v: number) => `${v}%`}
            width={40}
            domain={[0, 100]}
          />
          <ReferenceLine yAxisId="pnl" y={0} stroke="#71717a" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{
              background: "#12121a",
              border: "1px solid #1e1e2e",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number, name: string) => {
              if (name === "pnl") return [`${value.toFixed(2)}%`, "Daily PnL"];
              if (name === "winRate") return [`${value}%`, "Win Rate"];
              return [value, name];
            }}
          />
          <Bar yAxisId="pnl" dataKey="pnl" radius={[3, 3, 0, 0]}>
            {chartData.map((d, i) => (
              <Cell
                key={i}
                fill={d.pnl >= 0 ? "#22c55e" : "#ef4444"}
                fillOpacity={0.7}
              />
            ))}
          </Bar>
          <Line
            yAxisId="wr"
            type="monotone"
            dataKey="winRate"
            stroke="#a78bfa"
            strokeWidth={2}
            dot={{ r: 4, fill: "#a78bfa" }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex justify-center gap-4 mt-2 text-xs text-[var(--muted)]">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-[#22c55e]/70" />
          Daily PnL
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 rounded-sm bg-[#a78bfa]" />
          Win Rate
        </span>
      </div>
    </div>
  );
}
