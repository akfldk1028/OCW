/**
 * Trade Alert Hook Handler
 *
 * Sends formatted trade alerts to channels when strong signals are detected.
 * Triggered by cron:trade-scan events or manual trade:signal events.
 */

import type { InternalHookEvent } from "../../../../src/hooks/types.js";

interface TradeSignal {
  ticker: string;
  action: number;
  price: number;
  rsi?: number;
  macd?: number;
  sentiment?: number;
}

interface ScanResult {
  signals: TradeSignal[];
  timestamp: string;
  portfolioValue?: number;
  portfolioPnl?: number;
}

const MIN_SIGNAL_STRENGTH = 0.3;

function formatSignal(signal: TradeSignal): string {
  const direction = signal.action > 0 ? "BUY" : "SELL";
  const emoji = signal.action > 0 ? "📈" : "📉";
  const strength = Math.abs(signal.action);
  const strengthLabel = strength > 0.6 ? "Strong" : "Moderate";

  let line = `${emoji} ${direction} ${signal.ticker} @ $${signal.price.toFixed(2)}\n`;
  line += `   Strength: ${signal.action.toFixed(2)} (${strengthLabel})`;

  if (signal.rsi !== undefined || signal.macd !== undefined) {
    const parts: string[] = [];
    if (signal.rsi !== undefined) parts.push(`RSI: ${signal.rsi.toFixed(1)}`);
    if (signal.macd !== undefined) parts.push(`MACD: ${signal.macd >= 0 ? "+" : ""}${signal.macd.toFixed(2)}`);
    line += `\n   ${parts.join(" | ")}`;
  }

  return line;
}

function formatAlert(result: ScanResult): string {
  const strongSignals = result.signals.filter(
    (s) => Math.abs(s.action) >= MIN_SIGNAL_STRENGTH,
  );

  if (strongSignals.length === 0) return "";

  let message = "🚨 Trading Signal Detected\n\n";

  for (const signal of strongSignals) {
    message += formatSignal(signal) + "\n\n";
  }

  message += `⏰ ${result.timestamp}`;

  if (result.portfolioValue !== undefined) {
    const pnlStr = result.portfolioPnl !== undefined
      ? ` (${result.portfolioPnl >= 0 ? "+" : ""}${result.portfolioPnl.toFixed(2)}%)`
      : "";
    message += `\n💰 Portfolio: $${result.portfolioValue.toLocaleString()}${pnlStr}`;
  }

  return message;
}

const tradeAlertHandler = async (event: InternalHookEvent): Promise<void> => {
  // Only trigger on scan-related events
  if (event.type !== "cron" && event.type !== "trade") {
    return;
  }

  const context = event.context || {};
  const scanResult = context.scanResult as ScanResult | undefined;

  if (!scanResult?.signals?.length) {
    return;
  }

  const alertMessage = formatAlert(scanResult);

  if (!alertMessage) {
    return; // No strong signals
  }

  // The alert message is made available for the gateway to deliver
  // through the channel system
  if (context.sendMessage && typeof context.sendMessage === "function") {
    await (context.sendMessage as (msg: string) => Promise<void>)(alertMessage);
  }
};

export default tradeAlertHandler;
