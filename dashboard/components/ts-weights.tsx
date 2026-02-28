"use client";

import { useTSWeights } from "@/lib/hooks";
import { clsx } from "clsx";

const GROUP_ICONS: Record<string, string> = {
  market: "📊",
  regime: "🌐",
  momentum: "⚡",
  sentiment: "💭",
  quant: "🔢",
  funding_rate: "💰",
  oi_signal: "📈",
  macro: "🏦",
  technical_trend: "📉",
  technical_reversion: "🔄",
  technical_volume: "📦",
  derivatives: "🔮",
};

function getBarColor(weight: number, rank: number): string {
  if (rank === 0) return "bg-[var(--green)]";
  if (rank === 1) return "bg-emerald-600";
  if (weight > 0.17) return "bg-[var(--blue)]";
  return "bg-[var(--border)]";
}

export default function TSWeightsCard() {
  const { data, isLoading } = useTSWeights();

  if (isLoading) return <div className="card animate-pulse h-48" />;
  if (!data) return null;

  const weights = Object.entries(data.mean_weights).sort(
    ([, a], [, b]) => b - a
  );
  const maxWeight = Math.max(...weights.map(([, v]) => v), 0.01);
  const cumPnl = data.cumulative_pnl_pct ?? 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          🧠 RL Signal Weights
        </h2>
        <span
          className={clsx(
            "text-xs font-mono px-2 py-0.5 rounded-full",
            cumPnl >= 0
              ? "bg-[var(--green)]/10 text-[var(--green)]"
              : "bg-[var(--red)]/10 text-[var(--red)]"
          )}
        >
          {cumPnl >= 0 ? "+" : ""}
          {cumPnl.toFixed(2)}%
        </span>
      </div>
      <p className="text-xs text-[var(--muted)] mb-3">
        {data.total_trades} trades · {data.has_enough_data ? "Exploiting" : "Exploring"} · {data.current_regime}
      </p>

      <div className="space-y-1.5">
        {weights.map(([name, weight], i) => (
          <div key={name} className="group flex items-center gap-2">
            <span className="text-sm w-5 text-center">
              {GROUP_ICONS[name] ?? "•"}
            </span>
            <span className="text-[11px] font-mono w-28 text-[var(--muted)] group-hover:text-[var(--text)] transition-colors truncate">
              {name}
            </span>
            <div className="flex-1 h-5 rounded-md bg-[var(--border)]/50 overflow-hidden relative">
              <div
                className={clsx(
                  "h-full rounded-md transition-all duration-700 ease-out",
                  getBarColor(weight, i)
                )}
                style={{ width: `${(weight / maxWeight) * 100}%` }}
              />
              {i === 0 && (
                <span className="absolute right-2 top-0.5 text-[10px] text-white/60 font-mono">
                  BEST
                </span>
              )}
            </div>
            <span
              className={clsx(
                "text-xs font-mono w-12 text-right font-bold",
                i === 0 && "text-[var(--green)]",
                i === 1 && "text-emerald-400",
                i > 1 && "text-[var(--muted)]"
              )}
            >
              {(weight * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
