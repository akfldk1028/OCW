"use client";

import PortfolioCard from "@/components/portfolio-card";
import PositionsTable from "@/components/positions-table";
import DecisionFeed from "@/components/decision-feed";
import TradesTable from "@/components/trades-table";
import RegimeBadge from "@/components/regime-badge";
import TSWeightsCard from "@/components/ts-weights";
import PriceTicker from "@/components/price-ticker";

export default function Home() {
  return (
    <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            OCW Trading Dashboard
          </h1>
          <p className="text-sm text-[var(--muted)]">
            Agent-vs-Agent Crypto Engine
          </p>
        </div>
        <div className="text-xs text-[var(--muted)] font-mono">
          ocw-trader.fly.dev
        </div>
      </div>

      {/* Row 1: Portfolio + Prices + Regime */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <PortfolioCard />
        <PriceTicker />
        <RegimeBadge />
      </div>

      {/* Row 2: Positions */}
      <PositionsTable />

      {/* Row 3: Decisions + TS Weights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DecisionFeed />
        <TSWeightsCard />
      </div>

      {/* Row 4: Trades */}
      <TradesTable />
    </main>
  );
}
