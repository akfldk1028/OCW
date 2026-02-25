"use client";

import { useStatus } from "@/lib/hooks";
import { clsx } from "clsx";

function formatUSD(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

function formatDuration(seconds: number) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function PortfolioCard() {
  const { data, error, isLoading } = useStatus();

  if (isLoading) return <div className="card animate-pulse h-40" />;
  if (error || !data)
    return <div className="card text-red-400">Failed to load status</div>;

  const nextCheck = data.gate.seconds_until_next;

  return (
    <div className="card glow-blue">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-[var(--muted)]">Portfolio</h2>
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              "w-2 h-2 rounded-full pulse-dot",
              data.claude_available ? "bg-[var(--green)]" : "bg-[var(--red)]"
            )}
          />
          <span className="text-xs text-[var(--muted)]">
            {data.claude_available ? "Claude Active" : "Claude Offline"}
          </span>
        </div>
      </div>

      <p className="text-3xl font-bold tracking-tight mb-1">
        {formatUSD(data.portfolio_value)}
      </p>
      <p className="text-sm text-[var(--muted)] mb-4">
        {data.position_count} position{data.position_count !== 1 ? "s" : ""} open
      </p>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-[var(--muted)] text-xs">Regime</p>
          <p className="font-mono">{data.crypto_regime}</p>
        </div>
        <div>
          <p className="text-[var(--muted)] text-xs">Fear/Greed</p>
          <p className="font-mono">
            {data.fear_greed_index}{" "}
            <span className="text-[var(--muted)]">{data.fear_greed_label}</span>
          </p>
        </div>
        <div>
          <p className="text-[var(--muted)] text-xs">Next Check</p>
          <p className="font-mono">{formatDuration(nextCheck)}</p>
        </div>
      </div>
    </div>
  );
}
