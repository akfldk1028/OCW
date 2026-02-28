"use client";

import { useState } from "react";
import { useDecisions } from "@/lib/hooks";
import { clsx } from "clsx";

interface WakeCondition {
  metric: string;
  operator: string;
  threshold: number;
  reason: string;
}

function formatWakeCondition(w: WakeCondition | string): string {
  if (typeof w === "string") return w;
  if (w && typeof w === "object") {
    return `${w.metric} ${w.operator} ${w.threshold}${w.reason ? ` (${w.reason})` : ""}`;
  }
  return String(w);
}

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
  const [expanded, setExpanded] = useState<number | null>(null);

  if (isLoading) return <div className="card animate-pulse h-64" />;
  if (!data) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          Agent Decisions
        </h2>
        <span className="text-xs text-[var(--muted)]">{data.total} total</span>
      </div>
      <div className="space-y-2 max-h-[500px] overflow-y-auto">
        {data.decisions.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">No decisions yet</p>
        ) : (
          data.decisions.map((d, i) => {
            const isOpen = expanded === i;
            return (
              <div
                key={`${d.timestamp}-${i}`}
                className={clsx(
                  "border border-[var(--border)] rounded-lg p-3 cursor-pointer transition-all",
                  isOpen && "glow-blue border-[var(--blue)]/30"
                )}
                onClick={() => setExpanded(isOpen ? null : i)}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-[var(--border)] text-[var(--muted)]">
                      {d.trigger}
                    </span>
                    <span className="text-xs text-[var(--muted)]">
                      {timeAgo(d.timestamp)}
                    </span>
                    {d.regime && (
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-purple-900/20 text-[var(--purple)]">
                        {d.regime}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-[var(--muted)]">
                      BTC ${d.btc_price?.toLocaleString()}
                    </span>
                    <span className="text-xs text-[var(--muted)]">
                      {isOpen ? "\u25B2" : "\u25BC"}
                    </span>
                  </div>
                </div>

                {d.claude_assessment && (
                  <p
                    className={clsx(
                      "text-sm leading-relaxed mt-2",
                      !isOpen && "line-clamp-2"
                    )}
                  >
                    {d.claude_assessment}
                  </p>
                )}

                {d.trades && d.trades.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {d.trades.map((t, j) => (
                      <span
                        key={j}
                        className={clsx(
                          "text-xs font-mono px-2 py-1 rounded",
                          t.action === "BUY"
                            ? "bg-green-900/30 text-[var(--green)]"
                            : t.action === "SELL"
                            ? "bg-red-900/30 text-[var(--red)]"
                            : "bg-blue-900/30 text-[var(--blue)]"
                        )}
                      >
                        {t.action} {t.ticker}
                        {t.confidence
                          ? ` (${(t.confidence * 100).toFixed(0)}%)`
                          : ""}
                      </span>
                    ))}
                  </div>
                )}

                {isOpen && (
                  <div className="mt-3 pt-3 border-t border-[var(--border)] space-y-3">
                    {d.trades?.map((t, j) => (
                      <div key={j} className="text-xs">
                        <span
                          className={clsx(
                            "font-mono font-bold",
                            t.action === "BUY"
                              ? "text-[var(--green)]"
                              : t.action === "SELL"
                              ? "text-[var(--red)]"
                              : "text-[var(--blue)]"
                          )}
                        >
                          {t.action} {t.ticker}
                        </span>
                        <span className="text-[var(--muted)]">
                          {" "}@ ${t.price?.toLocaleString()}
                        </span>
                        {t.reasons && t.reasons.length > 0 && (
                          <ul className="mt-1 ml-4 space-y-0.5 list-disc text-[var(--muted)]">
                            {t.reasons.map((r, k) => (
                              <li key={k}>{r}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    ))}

                    {d.wake_conditions && d.wake_conditions.length > 0 && (
                      <div>
                        <span className="text-xs font-medium text-[var(--muted)]">
                          Wake Conditions:
                        </span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {d.wake_conditions.map((w, k) => (
                            <span
                              key={k}
                              className="text-xs font-mono px-1.5 py-0.5 rounded bg-[var(--border)] text-[var(--muted)]"
                            >
                              {formatWakeCondition(w)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex gap-4 text-xs text-[var(--muted)]">
                      {d.next_check_s && (
                        <span>Next: {Math.round(d.next_check_s / 60)}m</span>
                      )}
                      {d.portfolio_value && (
                        <span>Portfolio: ${d.portfolio_value.toLocaleString()}</span>
                      )}
                      {d.fear_greed !== undefined && (
                        <span>F&G: {d.fear_greed}</span>
                      )}
                    </div>

                    {d.memory && (
                      <div>
                        <span className="text-xs font-medium text-[var(--muted)]">
                          Agent Memory:
                        </span>
                        <p className="text-xs text-[var(--muted)] mt-1 font-mono bg-[var(--bg)] rounded p-2 max-h-20 overflow-y-auto">
                          {d.memory}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
