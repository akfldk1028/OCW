"use client";

import { useState } from "react";
import { useTrades } from "@/lib/hooks";
import { clsx } from "clsx";

function fmt$(n: number) {
  return n >= 1000
    ? `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    : `$${n.toFixed(4)}`;
}

export default function TradesTable() {
  const { data, isLoading } = useTrades(50);
  const [isOpen, setIsOpen] = useState(false);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  if (isLoading) return <div className="card animate-pulse h-20" />;
  if (!data) return null;

  const { summary } = data;
  const cumPnl = summary.cumulative_pnl_pct; // API already returns as %

  return (
    <div className="card">
      {/* Collapsible header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between hover:opacity-80 transition-opacity"
      >
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-medium text-[var(--muted)]">
            Trade History
          </h2>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-green-900/20 text-[var(--green)]">
              {summary.wins}W
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-red-900/20 text-[var(--red)]">
              {summary.losses}L
            </span>
            <span className="text-xs font-mono text-[var(--muted)]">
              WR {(summary.win_rate * 100).toFixed(0)}%
            </span>
            <span
              className={clsx(
                "text-xs font-mono font-bold",
                cumPnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"
              )}
            >
              {cumPnl >= 0 ? "+" : ""}
              {cumPnl.toFixed(2)}%
            </span>
          </div>
        </div>
        <span
          className={clsx(
            "text-[var(--muted)] transition-transform duration-200 text-sm",
            isOpen && "rotate-180"
          )}
        >
          ▼
        </span>
      </button>

      {/* Expandable body */}
      {isOpen && (
        <div className="mt-3 border-t border-[var(--border)] pt-3">
          {data.trades.length === 0 ? (
            <p className="text-sm text-[var(--muted)]">
              No trades executed yet
            </p>
          ) : (
            <div className="space-y-1">
              {data.trades.map((t, i) => {
                const pnl = Number(t.pnl_pct) * 100;
                const isExpanded = expandedRow === i;
                const isExit = t.action === "SELL" || t.action === "COVER";
                const isShort = t.action === "SHORT" || t.action === "COVER";
                const entryPx = t.entry_price ? Number(t.entry_price) : null;
                const exitPx = Number(t.price);

                return (
                  <div key={`${t.timestamp}-${i}`}>
                    <button
                      onClick={() =>
                        setExpandedRow(isExpanded ? null : i)
                      }
                      className="w-full flex items-center gap-3 py-2 px-2 rounded-lg hover:bg-[var(--border)]/30 transition-colors text-left"
                    >
                      {/* Action badge */}
                      <span
                        className={clsx(
                          "text-[10px] font-mono font-bold px-1.5 py-0.5 rounded w-14 text-center shrink-0",
                          t.action === "BUY" &&
                            "bg-green-900/30 text-[var(--green)]",
                          t.action === "SELL" &&
                            "bg-red-900/30 text-[var(--red)]",
                          t.action === "SHORT" &&
                            "bg-orange-900/30 text-orange-400",
                          t.action === "COVER" &&
                            "bg-purple-900/30 text-purple-400",
                          t.action === "ADD" &&
                            "bg-blue-900/30 text-[var(--blue)]"
                        )}
                      >
                        {t.action}
                      </span>

                      {/* Ticker */}
                      <span className="text-xs font-mono w-16 shrink-0">
                        {t.ticker?.replace("/USDT", "").replace(":USDT", "")}
                      </span>

                      {/* Price — exit trades show entry→exit, entry trades show price */}
                      {isExit && entryPx ? (
                        <span className="text-xs font-mono text-[var(--muted)] w-44 shrink-0 flex items-center gap-1">
                          <span className="text-[var(--green)]">{fmt$(entryPx)}</span>
                          <span className="text-[var(--muted)]">→</span>
                          <span className="text-[var(--red)]">{fmt$(exitPx)}</span>
                        </span>
                      ) : (
                        <span className="text-xs font-mono text-[var(--muted)] w-44 shrink-0">
                          {fmt$(exitPx)}
                        </span>
                      )}

                      {/* PnL */}
                      <span
                        className={clsx(
                          "text-xs font-mono font-bold w-16 shrink-0",
                          isExit && pnl > 0 && "text-[var(--green)]",
                          isExit && pnl < 0 && "text-[var(--red)]",
                          !isExit && "text-[var(--muted)]"
                        )}
                      >
                        {isExit
                          ? `${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}%`
                          : "—"}
                      </span>

                      {/* Time */}
                      <span className="text-[10px] font-mono text-[var(--muted)] flex-1 text-right">
                        {t.timestamp?.slice(5, 16)}
                      </span>

                      {/* Expand indicator */}
                      <span
                        className={clsx(
                          "text-[10px] text-[var(--muted)] transition-transform duration-200 shrink-0",
                          isExpanded && "rotate-180"
                        )}
                      >
                        ▾
                      </span>
                    </button>

                    {/* Expanded details */}
                    {isExpanded && (
                      <div className="ml-12 mr-4 mb-2 px-3 py-2 rounded-lg bg-[var(--border)]/20 border border-[var(--border)]/50 text-xs space-y-2">
                        {/* Price detail for exit trades (SELL/COVER) */}
                        {isExit && entryPx && (
                          <div className="flex items-center gap-4 py-1 border-b border-[var(--border)]/30">
                            {isShort && (
                              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-orange-900/30 text-orange-400">SHORT</span>
                            )}
                            <div className="flex items-center gap-2">
                              <span className="text-[var(--muted)]">Entry:</span>
                              <span className="font-mono font-bold text-[var(--text)]">{fmt$(entryPx)}</span>
                            </div>
                            <span className="text-[var(--muted)]">→</span>
                            <div className="flex items-center gap-2">
                              <span className="text-[var(--muted)]">Exit:</span>
                              <span className="font-mono font-bold text-[var(--text)]">{fmt$(exitPx)}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-[var(--muted)]">P&L:</span>
                              <span
                                className={clsx(
                                  "font-mono font-bold",
                                  pnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"
                                )}
                              >
                                {pnl >= 0 ? "+" : ""}
                                {pnl.toFixed(2)}% (${(isShort
                                  ? (entryPx - exitPx) * Number(t.qty)
                                  : (exitPx - entryPx) * Number(t.qty)
                                ).toFixed(2)})
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Entry info (BUY/SHORT) */}
                        {!isExit && (
                          <div className="flex items-center gap-4 py-1 border-b border-[var(--border)]/30">
                            {t.action === "SHORT" && (
                              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-orange-900/30 text-orange-400">SHORT</span>
                            )}
                            <div className="flex items-center gap-2">
                              <span className="text-[var(--muted)]">Entry Price:</span>
                              <span className="font-mono font-bold text-[var(--text)]">{fmt$(exitPx)}</span>
                            </div>
                          </div>
                        )}

                        {/* Reason */}
                        {t.reason && (
                          <p className="text-[var(--text)] leading-relaxed">
                            {t.reason}
                          </p>
                        )}

                        {/* Meta */}
                        <div className="flex flex-wrap gap-3 text-[var(--muted)] pt-1">
                          <span>
                            Qty: {Number(t.qty).toFixed(4)}
                          </span>
                          <span>
                            Value: $
                            {Number(t.value_usd).toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            })}
                          </span>
                          {t.held_hours > 0 && (
                            <span>
                              Held:{" "}
                              {t.held_hours < 1
                                ? `${(t.held_hours * 60).toFixed(0)}min`
                                : `${t.held_hours.toFixed(1)}h`}
                            </span>
                          )}
                          <span>Regime: {t.regime}</span>
                          <span>Source: {t.source}</span>
                          {t.confidence > 0 && (
                            <span>Conf: {(t.confidence * 100).toFixed(0)}%</span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
