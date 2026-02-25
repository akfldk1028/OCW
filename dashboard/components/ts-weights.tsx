"use client";

import { useTSWeights } from "@/lib/hooks";
import { clsx } from "clsx";

export default function TSWeightsCard() {
  const { data, isLoading } = useTSWeights();

  if (isLoading) return <div className="card animate-pulse h-48" />;
  if (!data) return null;

  const weights = Object.entries(data.mean_weights).sort(
    ([, a], [, b]) => b - a
  );
  const maxWeight = Math.max(...weights.map(([, v]) => v), 0.01);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Thompson Sampling Weights
        </h2>
        <span className="text-xs text-[var(--muted)]">
          {data.total_trades} trades
          {data.has_enough_data ? "" : " (using prior)"}
        </span>
      </div>
      <div className="space-y-2">
        {weights.map(([name, weight]) => (
          <div key={name} className="flex items-center gap-3">
            <span className="text-xs font-mono w-24 text-[var(--muted)]">
              {name}
            </span>
            <div className="flex-1 h-4 rounded bg-[var(--border)] overflow-hidden">
              <div
                className={clsx(
                  "h-full rounded transition-all duration-500",
                  weight > 0.2 ? "bg-[var(--green)]" : "bg-[var(--blue)]"
                )}
                style={{ width: `${(weight / maxWeight) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono w-12 text-right">
              {(weight * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
      <p className="text-xs text-[var(--muted)] mt-3 font-mono">
        Regime: {data.current_regime}
      </p>
    </div>
  );
}
