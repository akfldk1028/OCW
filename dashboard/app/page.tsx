"use client";

import HeroStats from "@/components/hero-stats";
import PortfolioCard from "@/components/portfolio-card";
import PositionsTable from "@/components/positions-table";
import DecisionFeed from "@/components/decision-feed";
import TradesTable from "@/components/trades-table";
import RegimeBadge from "@/components/regime-badge";
import TSWeightsCard from "@/components/ts-weights";
import PriceTicker from "@/components/price-ticker";
import PriceChart from "@/components/price-chart";
import EquityCurve from "@/components/equity-curve";

export default function Home() {
  return (
    <main className="max-w-[1400px] mx-auto px-4 py-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--blue)] to-[var(--purple)] flex items-center justify-center text-xs font-bold">
            OC
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              OCW Trading Engine
            </h1>
            <p className="text-xs text-[var(--muted)]">
              Agent-vs-Agent Crypto Trading
            </p>
          </div>
        </div>
        <PriceTicker />
      </div>

      {/* Row 1: Hero Stats — big PnL numbers */}
      <HeroStats />

      {/* Row 2: Charts side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <PriceChart />
        </div>
        <EquityCurve />
      </div>

      {/* Row 3: Portfolio detail + Regime */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-3">
          <PortfolioCard />
        </div>
        <RegimeBadge />
      </div>

      {/* Row 4: Positions */}
      <PositionsTable />

      {/* Row 5: Decisions + TS Weights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <DecisionFeed />
        </div>
        <TSWeightsCard />
      </div>

      {/* Row 6: Trades */}
      <TradesTable />

      {/* Footer */}
      <div className="text-center text-xs text-[var(--muted)] pb-4">
        Claude Sonnet 4.6 + Thompson Sampling + Adaptive Gate
      </div>
    </main>
  );
}
