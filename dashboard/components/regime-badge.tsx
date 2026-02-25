"use client";

import { useStatus } from "@/lib/hooks";
import { clsx } from "clsx";

const REGIME_COLORS: Record<string, string> = {
  low_vol: "bg-blue-900/30 text-[var(--blue)]",
  high_vol: "bg-red-900/30 text-[var(--red)]",
  trending: "bg-green-900/30 text-[var(--green)]",
  ranging: "bg-yellow-900/30 text-[var(--yellow)]",
  unknown: "bg-gray-900/30 text-[var(--muted)]",
};

const MACRO_COLORS: Record<string, string> = {
  GOLDILOCKS: "bg-green-900/30 text-[var(--green)]",
  REFLATION: "bg-yellow-900/30 text-[var(--yellow)]",
  INFLATION: "bg-red-900/30 text-[var(--red)]",
  DEFLATION: "bg-blue-900/30 text-[var(--blue)]",
};

export default function RegimeBadge() {
  const { data, isLoading } = useStatus();

  if (isLoading) return <div className="card animate-pulse h-20" />;
  if (!data) return null;

  const cryptoColor = Object.entries(REGIME_COLORS).find(([k]) =>
    data.crypto_regime.includes(k)
  )?.[1] ?? REGIME_COLORS.unknown;

  const macroColor = MACRO_COLORS[data.macro_regime] ?? REGIME_COLORS.unknown;

  return (
    <div className="card">
      <h2 className="text-sm font-medium text-[var(--muted)] mb-3">Regime</h2>
      <div className="flex gap-2">
        <span className={clsx("text-xs font-mono px-2 py-1 rounded", cryptoColor)}>
          Crypto: {data.crypto_regime}
        </span>
        <span className={clsx("text-xs font-mono px-2 py-1 rounded", macroColor)}>
          Macro: {data.macro_regime}
        </span>
      </div>
    </div>
  );
}
