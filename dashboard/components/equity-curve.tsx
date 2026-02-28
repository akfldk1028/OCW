"use client";

import { useEquityCurve } from "@/lib/hooks";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

export default function EquityCurve() {
  const { data, isLoading } = useEquityCurve();

  if (isLoading) return <div className="card animate-pulse h-80" />;

  const points = data?.points ?? [];
  if (points.length === 0) {
    return (
      <div className="card flex flex-col items-center justify-center h-80">
        <div className="text-4xl mb-2">📈</div>
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Equity Curve
        </h2>
        <p className="text-xs text-[var(--muted)] mt-1">
          Waiting for first decision...
        </p>
      </div>
    );
  }

  const chartData = points.map((p) => ({
    time: p.time.slice(5, 16),
    value: p.value,
  }));

  const values = chartData.map((d) => d.value);
  const initial = values[0] ?? 0;
  const current = values[values.length - 1] ?? initial;
  const pnl = initial > 0 ? ((current - initial) / initial) * 100 : 0;
  const isUp = pnl >= 0;
  const peak = Math.max(...values);
  const trough = Math.min(...values);
  const drawdown = peak > 0 ? ((peak - trough) / peak) * 100 : 0;

  // Y-axis domain: amplify small changes so they're visually obvious
  // Use max deviation from initial, then pad 50% extra so the line isn't at the edge
  const maxDeviation = Math.max(
    Math.abs(peak - initial),
    Math.abs(trough - initial),
    initial * 0.005, // minimum 0.5% range even if flat
  );
  const yPad = maxDeviation * 1.5;
  const yDomain: [number, number] = [
    Math.floor(initial - yPad),
    Math.ceil(initial + yPad),
  ];

  return (
    <div className="card">
      {/* Header with large PnL */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium text-[var(--muted)]">
            Equity Curve
          </h2>
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-mono ${
              isUp
                ? "bg-[var(--green)]/10 text-[var(--green)]"
                : "bg-[var(--red)]/10 text-[var(--red)]"
            }`}
          >
            {isUp ? "▲" : "▼"} {Math.abs(pnl).toFixed(2)}%
          </span>
        </div>
        <p className="text-2xl font-bold font-mono tracking-tight">
          ${current.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </p>
        <div className="flex gap-4 mt-1 text-xs text-[var(--muted)]">
          <span>
            Initial: ${initial.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
          <span className="text-[var(--red)]">
            MDD: -{drawdown.toFixed(1)}%
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor={isUp ? "#22c55e" : "#ef4444"}
                stopOpacity={0.25}
              />
              <stop
                offset="100%"
                stopColor={isUp ? "#22c55e" : "#ef4444"}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "#71717a" }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#71717a" }}
            tickFormatter={(v: number) =>
              `$${(v / 1000).toFixed(1)}k`
            }
            width={55}
            domain={yDomain}
          />
          <ReferenceLine
            y={initial}
            stroke="#71717a"
            strokeDasharray="3 3"
            label={{
              value: "Start",
              position: "right",
              fill: "#71717a",
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              background: "#12121a",
              border: "1px solid #1e1e2e",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => [
              `$${value.toLocaleString()}`,
              "Portfolio",
            ]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={isUp ? "#22c55e" : "#ef4444"}
            strokeWidth={2}
            fill="url(#equityGrad)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
