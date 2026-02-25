"use client";

import { useDecisions } from "@/lib/hooks";
import { clsx } from "clsx";

function timeAgo(isoStr: string): string {
  const diff = Date.now() - new Date(isoStr).getTime();
  const min = Math.floor(diff / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function DecisionFeed() {
  const { data, isLoading } = useDecisions(20);

  if (isLoading) return <div className="card animate-pulse h-64" />;
  if (!data) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Claude Decisions
        </h2>
        <span className="text-xs text-[var(--muted)]">
          {data.total} total
        </span>
      </div>
      <div className="space-y-3 max-h-[400px] overflow-y-auto">
        {data.decisions.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">No decisions yet</p>
        ) : (
          data.decisions.map((d, i) => (
            <div
              key={`${d.timestamp}-${i}`}
              className="border-b border-[var(--border)] pb-3 last:border-0"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-[var(--border)] text-[var(--muted)]">
                    {d.trigger}
                  </span>
                  <span className="text-xs text-[var(--muted)]">
                    {timeAgo(d.timestamp)}
                  </span>
                </div>
                <span className="text-xs font-mono text-[var(--muted)]">
                  ${d.btc_price?.toLocaleString()}
                </span>
              </div>
              {d.claude_assessment && (
                <p className="text-sm leading-relaxed line-clamp-2">
                  {d.claude_assessment}
                </p>
              )}
              {d.trades && d.trades.length > 0 && (
                <div className="mt-1 flex gap-2">
                  {d.trades.map((t, j) => (
                    <span
                      key={j}
                      className={clsx(
                        "text-xs font-mono px-1.5 py-0.5 rounded",
                        t.action === "BUY"
                          ? "bg-green-900/30 text-[var(--green)]"
                          : t.action === "SELL"
                          ? "bg-red-900/30 text-[var(--red)]"
                          : "bg-blue-900/30 text-[var(--blue)]"
                      )}
                    >
                      {t.action} {t.ticker}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
