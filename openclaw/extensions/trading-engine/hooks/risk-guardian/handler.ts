/**
 * Risk Guardian Hook Handler
 *
 * Monitors open positions for stop-loss/take-profit triggers
 * and enforces daily loss limits.
 */

import type { InternalHookEvent } from "../../../../src/hooks/types.js";

interface Position {
  ticker: string;
  shares: number;
  entryPrice: number;
  currentPrice: number;
  side: "long" | "short";
}

interface PortfolioState {
  positions: Position[];
  portfolioValue: number;
  dailyPnl: number;
  dailyPnlPct: number;
  tradingHalted: boolean;
}

interface RiskConfig {
  takeProfitPct: number;
  stopLossPct: number;
  maxDailyLossPct: number;
}

const DEFAULT_RISK_CONFIG: RiskConfig = {
  takeProfitPct: 0.01,      // +1%
  stopLossPct: -0.005,      // -0.5%
  maxDailyLossPct: -0.02,   // -2%
};

function checkPosition(
  position: Position,
  config: RiskConfig,
): { action: "hold" | "close"; reason?: string; pnlPct: number } {
  const pnlPct = position.side === "long"
    ? (position.currentPrice - position.entryPrice) / position.entryPrice
    : (position.entryPrice - position.currentPrice) / position.entryPrice;

  // Take-profit
  if (pnlPct >= config.takeProfitPct) {
    return {
      action: "close",
      reason: `Take-profit at ${(config.takeProfitPct * 100).toFixed(1)}% reached (current: ${(pnlPct * 100).toFixed(2)}%)`,
      pnlPct,
    };
  }

  // Stop-loss
  if (pnlPct <= config.stopLossPct) {
    return {
      action: "close",
      reason: `Stop-loss at ${(config.stopLossPct * 100).toFixed(1)}% exceeded (current: ${(pnlPct * 100).toFixed(2)}%)`,
      pnlPct,
    };
  }

  return { action: "hold", pnlPct };
}

function formatRiskEvent(
  position: Position,
  reason: string,
  pnlPct: number,
  isProfit: boolean,
): string {
  const emoji = isProfit ? "✅" : "❌";
  const header = isProfit ? "TAKE-PROFIT TRIGGERED" : "STOP-LOSS TRIGGERED";

  return [
    `🛡️ ${header}`,
    "",
    `${emoji} SOLD ${position.ticker} @ $${position.currentPrice.toFixed(2)}`,
    `   Entry: $${position.entryPrice.toFixed(2)} | ${isProfit ? "Profit" : "Loss"}: ${pnlPct >= 0 ? "+" : ""}${(pnlPct * 100).toFixed(2)}%`,
    `   Reason: ${reason}`,
  ].join("\n");
}

const riskGuardianHandler = async (event: InternalHookEvent): Promise<void> => {
  if (event.type !== "trade" && event.type !== "cron") {
    return;
  }

  const context = event.context || {};
  const portfolioState = context.portfolioState as PortfolioState | undefined;

  if (!portfolioState?.positions?.length) {
    return;
  }

  const config: RiskConfig = {
    ...DEFAULT_RISK_CONFIG,
    ...(context.riskConfig as Partial<RiskConfig> | undefined),
  };

  // Check daily loss limit
  if (portfolioState.dailyPnlPct <= config.maxDailyLossPct) {
    const haltMessage = [
      "🚫 TRADING HALTED",
      "",
      `Daily loss limit reached: ${(portfolioState.dailyPnlPct * 100).toFixed(2)}%`,
      "No new trades will be executed until next trading day.",
    ].join("\n");

    if (context.sendMessage && typeof context.sendMessage === "function") {
      await (context.sendMessage as (msg: string) => Promise<void>)(haltMessage);
    }
    return;
  }

  // Check each position
  const alerts: string[] = [];

  for (const position of portfolioState.positions) {
    const result = checkPosition(position, config);

    if (result.action === "close" && result.reason) {
      const isProfit = result.pnlPct >= 0;
      alerts.push(formatRiskEvent(position, result.reason, result.pnlPct, isProfit));

      // Request position close through the context
      if (context.closePosition && typeof context.closePosition === "function") {
        await (context.closePosition as (ticker: string) => Promise<void>)(position.ticker);
      }
    }
  }

  // Send consolidated alerts
  if (alerts.length > 0 && context.sendMessage && typeof context.sendMessage === "function") {
    const message = alerts.join("\n\n") +
      `\n\n💰 Daily P&L: ${portfolioState.dailyPnl >= 0 ? "+" : ""}$${portfolioState.dailyPnl.toFixed(0)} (${portfolioState.dailyPnlPct >= 0 ? "+" : ""}${(portfolioState.dailyPnlPct * 100).toFixed(2)}%)`;
    await (context.sendMessage as (msg: string) => Promise<void>)(message);
  }
};

export default riskGuardianHandler;
