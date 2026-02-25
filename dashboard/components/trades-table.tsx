"use client";

import { useTrades } from "@/lib/hooks";
import { clsx } from "clsx";

export default function TradesTable() {
  const { data, isLoading } = useTrades(50);

  if (isLoading) return <div className="card animate-pulse h-48" />;
  if (!data) return null;

  const { summary } = data;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Trade History
        </h2>
        <div className="flex items-center gap-3 text-xs text-[var(--muted)]">
          <span>
            W/L: {summary.wins}/{summary.losses}
          </span>
          <span>WR: {(summary.win_rate * 100).toFixed(0)}%</span>
          <span
            className={clsx(
              "font-mono",
              summary.cumulative_pnl_pct >= 0
                ? "text-[var(--green)]"
                : "text-[var(--red)]"
            )}
          >
            PnL: {(summary.cumulative_pnl_pct * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {data.trades.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">No trades executed yet</p>
      ) : (
        <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-[var(--card)]">
              <tr className="text-[var(--muted)] text-xs border-b border-[var(--border)]">
                <th className="text-left pb-2">Time</th>
                <th className="text-left pb-2">Action</th>
                <th className="text-left pb-2">Ticker</th>
                <th className="text-right pb-2">Price</th>
                <th className="text-right pb-2">PnL</th>
                <th className="text-right pb-2">Source</th>
              </tr>
            </thead>
            <tbody>
              {data.trades.map((t, i) => (
                <tr
                  key={`${t.timestamp}-${i}`}
                  className="border-b border-[var(--border)] last:border-0"
                >
                  <td className="py-1.5 text-xs text-[var(--muted)] font-mono">
                    {t.timestamp?.slice(5, 16)}
                  </td>
                  <td className="py-1.5">
                    <span
                      className={clsx(
                        "text-xs font-mono px-1 py-0.5 rounded",
                        t.action === "BUY"
                          ? "bg-green-900/30 text-[var(--green)]"
                          : "bg-red-900/30 text-[var(--red)]"
                      )}
                    >
                      {t.action}
                    </span>
                  </td>
                  <td className="py-1.5 font-mono text-xs">{t.ticker}</td>
                  <td className="py-1.5 text-right font-mono text-xs">
                    ${Number(t.price).toLocaleString()}
                  </td>
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono text-xs",
                      Number(t.pnl_pct) > 0
                        ? "text-[var(--green)]"
                        : Number(t.pnl_pct) < 0
                        ? "text-[var(--red)]"
                        : "text-[var(--muted)]"
                    )}
                  >
                    {t.action === "SELL"
                      ? `${(Number(t.pnl_pct) * 100).toFixed(2)}%`
                      : "-"}
                  </td>
                  <td className="py-1.5 text-right text-xs text-[var(--muted)]">
                    {t.source}
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
