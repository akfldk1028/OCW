"use client";

import { useState } from "react";
import { useTSWeights } from "@/lib/hooks";
import { BetaInfo, RegimeInfo } from "@/lib/api";
import { clsx } from "clsx";

const GROUP_ICONS: Record<string, string> = {
  macro: "🏦",
  technical_trend: "📉",
  technical_reversion: "🔄",
  technical_volume: "📦",
  derivatives: "🔮",
  sentiment: "💭",
};

const META_LABELS: Record<string, { icon: string; desc: string }> = {
  position_scale: { icon: "📏", desc: "Position sizing" },
  entry_selectivity: { icon: "🎯", desc: "Entry pickiness" },
  hold_patience: { icon: "⏳", desc: "Hold duration" },
  trade_frequency: { icon: "⚡", desc: "Trade frequency" },
  trend_vs_reversion: { icon: "↔️", desc: "Trend vs Mean-rev" },
  risk_aversion: { icon: "🛡️", desc: "Risk aversion" },
  profit_target_width: { icon: "🎯", desc: "TP width" },
  loss_tolerance: { icon: "✂️", desc: "Loss tolerance" },
};

const GROUP_SHORT: Record<string, string> = {
  technical_trend: "trend",
  technical_reversion: "reversion",
  technical_volume: "volume",
  derivatives: "deriv",
  sentiment: "sentiment",
  macro: "macro",
};

function signalColor(mean: number, hasData: boolean): string {
  if (!hasData) return "text-[var(--muted)]/40";
  if (mean > 0.58) return "text-emerald-400";
  if (mean > 0.52) return "text-emerald-400/60";
  if (mean < 0.42) return "text-red-400";
  if (mean < 0.48) return "text-red-400/60";
  return "text-[var(--muted)]";
}

function barColor(mean: number, hasData: boolean): string {
  if (!hasData) return "bg-[var(--border)]/30";
  if (mean > 0.55) return "bg-emerald-500/70";
  if (mean < 0.45) return "bg-red-500/50";
  return "bg-zinc-500/40";
}

function metaColor(mean: number): string {
  if (mean > 0.55) return "text-emerald-400";
  if (mean < 0.45) return "text-red-400";
  return "text-[var(--text)]";
}

/** Compact inline bar: centered at 0.5 */
function MiniBar({ mean, hasData }: { mean: number; hasData: boolean }) {
  const dev = mean - 0.5;
  const pct = Math.abs(dev) * 200; // max 100% if mean=0 or 1
  return (
    <div className="w-16 h-3 rounded-sm bg-[var(--border)]/30 relative overflow-hidden">
      <div className="absolute top-0 h-full w-px bg-white/10" style={{ left: "50%" }} />
      {dev !== 0 && (
        <div
          className={clsx("absolute top-0 h-full rounded-sm", barColor(mean, hasData))}
          style={
            dev > 0
              ? { left: "50%", width: `${Math.min(pct, 50)}%` }
              : { right: "50%", width: `${Math.min(pct, 50)}%` }
          }
        />
      )}
    </div>
  );
}

/** Tiny bar for regime comparison table */
function TinyBar({ mean }: { mean: number }) {
  const dev = mean - 0.5;
  const pct = Math.abs(dev) * 200;
  return (
    <div className="w-10 h-2 rounded-sm bg-[var(--border)]/20 relative overflow-hidden inline-block align-middle ml-1">
      <div className="absolute top-0 h-full w-px bg-white/5" style={{ left: "50%" }} />
      {dev !== 0 && (
        <div
          className={clsx(
            "absolute top-0 h-full rounded-sm",
            mean > 0.55 ? "bg-emerald-500/60" : mean < 0.45 ? "bg-red-500/40" : "bg-zinc-500/30"
          )}
          style={
            dev > 0
              ? { left: "50%", width: `${Math.min(pct, 50)}%` }
              : { right: "50%", width: `${Math.min(pct, 50)}%` }
          }
        />
      )}
    </div>
  );
}

/** Format regime name for display */
function fmtRegime(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Medium Volatility", "MedVol")
    .replace("High Volatility", "HiVol")
    .replace("Low Volatility", "LoVol")
    .replace("Ranging", "Rng")
    .replace("Reflation", "Refl")
    .replace("Stagflation", "Stag");
}

export default function TSWeightsCard() {
  const { data, isLoading } = useTSWeights();
  const [showRegimes, setShowRegimes] = useState(true);
  const [regimeTab, setRegimeTab] = useState<"meta" | "groups" | "signals">("meta");

  if (isLoading) return <div className="card animate-pulse h-48" />;
  if (!data) return null;

  const cumPnl = data.cumulative_pnl_pct ?? 0;

  // Build signal beta lookup + group mapping
  const signalBetas: Record<string, BetaInfo> = {};
  const signalToGroup: Record<string, string> = {};
  if (data.group_summary) {
    for (const [groupName, group] of Object.entries(data.group_summary)) {
      if (group.signals) {
        for (const [sig, beta] of Object.entries(group.signals)) {
          signalBetas[sig] = beta;
          signalToGroup[sig] = groupName;
        }
      }
    }
  }

  const reliability = data.signal_reliability ?? data.mean_weights;

  // Split signals: active (n>0), learning (n=0 but mean!=0.5), inactive (n=0, mean=0.5)
  const allSignals = Object.entries(reliability).sort(([, a], [, b]) => b - a);
  const activeSignals = allSignals.filter(([name]) => (signalBetas[name]?.total_trades ?? 0) > 0);
  const learningSignals = allSignals.filter(([name]) => {
    const sb = signalBetas[name];
    return (sb?.total_trades ?? 0) === 0 && Math.abs((sb?.mean ?? 0.5) - 0.5) > 0.005;
  });
  const inactiveSignals = allSignals.filter(([name]) => {
    const sb = signalBetas[name];
    return (sb?.total_trades ?? 0) === 0 && Math.abs((sb?.mean ?? 0.5) - 0.5) <= 0.005;
  });

  // Regime data
  const regimeInfo = data.regime_info ?? {};
  const regimeEntries = Object.entries(regimeInfo).sort(
    ([, a], [, b]) => b.trade_count - a.trade_count
  );
  const hasRegimes = regimeEntries.length > 0;

  // All meta-param keys across regimes (union)
  const allMetaKeys = Array.from(
    new Set(regimeEntries.flatMap(([, r]) => Object.keys(r.meta_param_means)))
  );
  // All group keys across regimes
  const allGroupKeys = Array.from(
    new Set(regimeEntries.flatMap(([, r]) => Object.keys(r.group_means)))
  );

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-sm font-medium text-[var(--muted)]">
          🧠 H-TS Bayesian RL
        </h2>
        <span
          className={clsx(
            "text-xs font-mono px-2 py-0.5 rounded-full",
            cumPnl >= 0
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
          )}
        >
          {cumPnl >= 0 ? "+" : ""}{cumPnl.toFixed(2)}%
        </span>
      </div>
      <p className="text-xs text-[var(--muted)] mb-3">
        {data.total_trades} trades · {data.has_enough_data ? "Exploiting" : "Exploring"} · {data.current_regime}
      </p>

      {/* ===== Level 0: Meta Params ===== */}
      {data.meta_param_betas && Object.keys(data.meta_param_betas).length > 0 && (
        <div className="mb-3">
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--muted)] mb-1.5">
            Meta Parameters
          </h3>
          <div className="grid grid-cols-3 gap-x-3 gap-y-1">
            {Object.entries(data.meta_param_betas)
              .sort(([, a], [, b]) => (b.mean ?? 0) - (a.mean ?? 0))
              .map(([name, beta]) => {
                const meta = META_LABELS[name];
                return (
                  <div key={name} className="flex items-center gap-1.5">
                    <span className="text-xs">{meta?.icon ?? "⚙️"}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono text-[var(--muted)] truncate">
                          {meta?.desc ?? name}
                        </span>
                        <span
                          className={clsx(
                            "text-[11px] font-mono font-bold ml-1",
                            metaColor(beta.mean)
                          )}
                        >
                          {beta.mean.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-1 rounded-full bg-[var(--border)]/50 overflow-hidden">
                        <div
                          className={clsx(
                            "h-full rounded-full transition-all duration-500",
                            beta.mean > 0.55 ? "bg-emerald-500/60" :
                            beta.mean < 0.45 ? "bg-red-500/50" : "bg-zinc-500/40"
                          )}
                          style={{ width: `${beta.mean * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* ===== Level 1: Groups ===== */}
      {data.group_summary && Object.keys(data.group_summary).length > 0 && (
        <div className="mb-3">
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--muted)] mb-1.5">
            Signal Groups
          </h3>
          <div className="grid grid-cols-3 gap-x-3 gap-y-1">
            {Object.entries(data.group_summary)
              .sort(([, a], [, b]) => (b.group_beta?.mean ?? 0) - (a.group_beta?.mean ?? 0))
              .map(([name, group]) => {
                const gb = group.group_beta;
                const mean = gb?.mean ?? 0.5;
                return (
                  <div key={name} className="flex items-center gap-1.5">
                    <span className="text-xs">{GROUP_ICONS[name] ?? "•"}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono text-[var(--muted)] truncate">
                          {name.replace("technical_", "tech_")}
                        </span>
                        <span
                          className={clsx(
                            "text-[11px] font-mono font-bold ml-1",
                            mean > 0.55 ? "text-emerald-400" :
                            mean < 0.45 ? "text-red-400" : "text-[var(--muted)]"
                          )}
                        >
                          {(mean * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1 rounded-full bg-[var(--border)]/50 overflow-hidden">
                        <div
                          className={clsx(
                            "h-full rounded-full transition-all duration-500",
                            mean > 0.55 ? "bg-emerald-500/60" :
                            mean < 0.45 ? "bg-red-500/50" : "bg-zinc-500/40"
                          )}
                          style={{ width: `${(mean / 0.7) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* ===== Regime RL Comparison ===== */}
      {hasRegimes && (
        <div className="mb-3">
          <button
            onClick={() => setShowRegimes(!showRegimes)}
            className="text-[10px] uppercase tracking-wider text-[var(--muted)] mb-1.5 hover:text-[var(--text)] transition-colors w-full text-left"
          >
            {showRegimes ? "▼" : "▶"} Regime RL ({regimeEntries.length} regimes)
          </button>

          {showRegimes && (
            <div className="space-y-2">
              {/* Tab selector */}
              <div className="flex gap-1">
                {(["meta", "groups", "signals"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setRegimeTab(tab)}
                    className={clsx(
                      "text-[9px] font-mono px-2 py-0.5 rounded transition-colors",
                      regimeTab === tab
                        ? "bg-[var(--border)] text-[var(--text)]"
                        : "text-[var(--muted)]/60 hover:text-[var(--muted)]"
                    )}
                  >
                    {tab === "meta" ? "Meta Params" : tab === "groups" ? "Groups" : "Signals"}
                  </button>
                ))}
              </div>

              {/* Regime column headers (shared) */}
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="text-[var(--muted)]/60">
                      <th className="text-left font-normal pr-2 pb-1 sticky left-0 bg-[var(--card)]">
                        {regimeTab === "meta" ? "Meta" : regimeTab === "groups" ? "Group" : "Signal"}
                      </th>
                      {regimeEntries.map(([name, info]) => (
                        <th key={name} className="text-center font-normal px-1 pb-1">
                          <div className="truncate max-w-[80px]" title={name}>
                            {fmtRegime(name)}
                          </div>
                          <div className={clsx(
                            "text-[8px]",
                            info.using_own_weights ? "text-emerald-400/60" : "text-amber-400/60"
                          )}>
                            {info.trade_count}t {info.using_own_weights ? "✓" : "→G"}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {/* Meta-params tab */}
                    {regimeTab === "meta" && allMetaKeys.map((param) => {
                      const meta = META_LABELS[param];
                      return (
                        <tr key={param} className="border-t border-[var(--border)]/10">
                          <td className="text-[var(--muted)] pr-2 py-0.5 whitespace-nowrap sticky left-0 bg-[var(--card)]">
                            {meta?.icon ?? "⚙️"} {meta?.desc ?? param}
                          </td>
                          {regimeEntries.map(([rName, rInfo]) => {
                            const mean = rInfo.meta_param_means[param] ?? 0.5;
                            return (
                              <td key={rName} className="text-center px-1 py-0.5">
                                <span className={metaColor(mean)}>
                                  {mean.toFixed(2)}
                                </span>
                                <TinyBar mean={mean} />
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}

                    {/* Groups tab */}
                    {regimeTab === "groups" && allGroupKeys.map((group) => (
                      <tr key={group} className="border-t border-[var(--border)]/10">
                        <td className="text-[var(--muted)] pr-2 py-0.5 whitespace-nowrap sticky left-0 bg-[var(--card)]">
                          {GROUP_ICONS[group] ?? "•"} {GROUP_SHORT[group] ?? group}
                        </td>
                        {regimeEntries.map(([rName, rInfo]) => {
                          const mean = rInfo.group_means[group] ?? 0.5;
                          return (
                            <td key={rName} className="text-center px-1 py-0.5">
                              <span className={clsx(
                                mean > 0.55 ? "text-emerald-400" :
                                mean < 0.45 ? "text-red-400" : "text-[var(--muted)]"
                              )}>
                                {mean.toFixed(2)}
                              </span>
                              <TinyBar mean={mean} />
                            </td>
                          );
                        })}
                      </tr>
                    ))}

                    {/* Signals tab */}
                    {regimeTab === "signals" && (() => {
                      // Collect all signal names, sort by global reliability
                      const globalRel = data.signal_reliability ?? {};
                      const sigNames = Object.keys(globalRel).sort(
                        (a, b) => (globalRel[b] ?? 0.5) - (globalRel[a] ?? 0.5)
                      );
                      return sigNames.map((sigName) => {
                        const groupName = signalToGroup[sigName];
                        return (
                          <tr key={sigName} className="border-t border-[var(--border)]/10">
                            <td className="text-[var(--muted)] pr-2 py-0.5 whitespace-nowrap sticky left-0 bg-[var(--card)]">
                              <span className="opacity-50">{GROUP_ICONS[groupName] ?? "•"}</span>{" "}
                              {sigName.length > 16 ? sigName.slice(0, 14) + ".." : sigName}
                            </td>
                            {regimeEntries.map(([rName, rInfo]) => {
                              const sr = rInfo.signal_reliability?.[sigName];
                              const mean = sr?.mean ?? 0.5;
                              const n = sr?.total_trades ?? 0;
                              return (
                                <td key={rName} className={clsx(
                                  "text-center px-1 py-0.5",
                                  n === 0 && "opacity-40"
                                )}>
                                  <span className={clsx(
                                    mean > 0.55 ? "text-emerald-400" :
                                    mean < 0.45 ? "text-red-400" : "text-[var(--muted)]"
                                  )}>
                                    {mean.toFixed(2)}
                                  </span>
                                  <TinyBar mean={mean} />
                                </td>
                              );
                            })}
                          </tr>
                        );
                      });
                    })()}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===== Level 2: Signal Reliability — horizontal regime comparison ===== */}
      <div>
        <h3 className="text-[10px] uppercase tracking-wider text-[var(--muted)] mb-1.5">
          Signal Reliability
          <span className="ml-1 opacity-40 normal-case">
            (32 signals x {regimeEntries.length + 1} regimes)
          </span>
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead>
              <tr className="text-[var(--muted)]/60">
                <th className="text-left font-normal pr-2 pb-1 sticky left-0 bg-[var(--card)] z-10 min-w-[120px]">
                  Signal
                </th>
                {/* Global column */}
                <th className="text-center font-normal px-1 pb-1 min-w-[70px]">
                  <div className="text-blue-400/70">Global</div>
                  <div className="text-[8px] text-[var(--muted)]/40">
                    {data.total_trades}t
                  </div>
                </th>
                {/* Per-regime columns */}
                {regimeEntries.map(([name, info]) => (
                  <th key={name} className="text-center font-normal px-1 pb-1 min-w-[70px]">
                    <div className="truncate max-w-[80px]" title={name}>
                      {fmtRegime(name)}
                    </div>
                    <div className={clsx(
                      "text-[8px]",
                      info.using_own_weights ? "text-emerald-400/60" : "text-amber-400/60"
                    )}>
                      {info.trade_count}t {info.using_own_weights ? "✓" : "→G"}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(() => {
                // Sort signals by global reliability descending
                const globalRel = reliability;
                const sigNames = Object.keys(globalRel).sort(
                  (a, b) => (globalRel[b] ?? 0.5) - (globalRel[a] ?? 0.5)
                );
                return sigNames.map((sigName) => {
                  const groupName = signalToGroup[sigName];
                  const globalMean = globalRel[sigName] ?? 0.5;
                  const globalN = signalBetas[sigName]?.total_trades ?? 0;
                  return (
                    <tr key={sigName} className="border-t border-[var(--border)]/10">
                      <td className="pr-2 py-0.5 whitespace-nowrap sticky left-0 bg-[var(--card)] z-10">
                        <span className="opacity-40 mr-0.5">{GROUP_ICONS[groupName] ?? "•"}</span>
                        <span className="text-[var(--muted)]">
                          {sigName.length > 18 ? sigName.slice(0, 16) + ".." : sigName}
                        </span>
                      </td>
                      {/* Global */}
                      <td className={clsx(
                        "text-center px-1 py-0.5",
                        globalN === 0 && "opacity-30"
                      )}>
                        <span className={clsx(
                          globalMean > 0.55 ? "text-emerald-400" :
                          globalMean < 0.45 ? "text-red-400" : "text-[var(--muted)]"
                        )}>
                          {globalMean.toFixed(2)}
                        </span>
                        <TinyBar mean={globalMean} />
                        {globalN > 0 && (
                          <span className="text-[8px] text-[var(--muted)]/30 ml-0.5">
                            {globalN}
                          </span>
                        )}
                      </td>
                      {/* Per-regime */}
                      {regimeEntries.map(([rName, rInfo]) => {
                        const sr = rInfo.signal_reliability?.[sigName];
                        const mean = sr?.mean ?? 0.5;
                        const n = sr?.total_trades ?? 0;
                        // Highlight divergence from global
                        const diff = mean - globalMean;
                        const hasDivergence = Math.abs(diff) > 0.05 && n > 0;
                        return (
                          <td key={rName} className={clsx(
                            "text-center px-1 py-0.5",
                            n === 0 && "opacity-30",
                            hasDivergence && "bg-white/[0.02]"
                          )}>
                            <span className={clsx(
                              mean > 0.55 ? "text-emerald-400" :
                              mean < 0.45 ? "text-red-400" : "text-[var(--muted)]"
                            )}>
                              {mean.toFixed(2)}
                            </span>
                            <TinyBar mean={mean} />
                            {n > 0 && (
                              <span className="text-[8px] text-[var(--muted)]/30 ml-0.5">
                                {n}
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                });
              })()}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
