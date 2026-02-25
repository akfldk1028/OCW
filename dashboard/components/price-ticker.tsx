"use client";

import { useStatus } from "@/lib/hooks";

export default function PriceTicker() {
  const { data, isLoading } = useStatus();

  if (isLoading) return <div className="card animate-pulse h-16" />;
  if (!data) return null;

  const prices = Object.entries(data.tick_prices);

  return (
    <div className="card">
      <h2 className="text-sm font-medium text-[var(--muted)] mb-3">
        Live Prices
      </h2>
      <div className="flex gap-6">
        {prices.map(([ticker, price]) => (
          <div key={ticker}>
            <p className="text-xs text-[var(--muted)]">{ticker}</p>
            <p className="text-lg font-mono font-bold">
              ${price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </p>
          </div>
        ))}
        {prices.length === 0 && (
          <p className="text-sm text-[var(--muted)]">Waiting for WS data...</p>
        )}
      </div>
    </div>
  );
}
