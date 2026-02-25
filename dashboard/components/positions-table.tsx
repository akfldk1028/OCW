"use client";

import { useStatus } from "@/lib/hooks";
import { clsx } from "clsx";

export default function PositionsTable() {
  const { data, isLoading } = useStatus();

  if (isLoading) return <div className="card animate-pulse h-32" />;
  if (!data) return null;

  const entries = Object.entries(data.positions);

  return (
    <div className="card">
      <h2 className="text-sm font-medium text-[var(--muted)] mb-3">
        Open Positions
      </h2>
      {entries.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">No open positions — cash only</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[var(--muted)] text-xs border-b border-[var(--border)]">
                <th className="text-left pb-2">Ticker</th>
                <th className="text-right pb-2">Entry</th>
                <th className="text-right pb-2">Current</th>
                <th className="text-right pb-2">PnL</th>
                <th className="text-right pb-2">Held</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([ticker, pos]) => (
                <tr key={ticker} className="border-b border-[var(--border)] last:border-0">
                  <td className="py-2 font-mono font-medium">{ticker}</td>
                  <td className="py-2 text-right font-mono">
                    ${pos.entry_price.toLocaleString()}
                  </td>
                  <td className="py-2 text-right font-mono">
                    ${pos.current_price.toLocaleString()}
                  </td>
                  <td
                    className={clsx(
                      "py-2 text-right font-mono font-medium",
                      pos.pnl_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"
                    )}
                  >
                    {(pos.pnl_pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 text-right font-mono text-[var(--muted)]">
                    {pos.held_hours.toFixed(1)}h
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
