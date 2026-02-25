const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://ocw-trader.fly.dev";

export async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

// Types
export interface Position {
  entry_price: number;
  current_price: number;
  qty: number;
  pnl_pct: number;
  held_hours: number;
  trailing_high: number;
}

export interface GateInfo {
  next_check_at: number;
  seconds_until_next: number;
  wake_conditions: string[];
}

export interface StatusData {
  portfolio_value: number;
  positions: Record<string, Position>;
  position_count: number;
  crypto_regime: string;
  macro_regime: string;
  combined_regime: string;
  fear_greed_index: number;
  fear_greed_label: string;
  gate: GateInfo;
  agent_memory: string;
  claude_available: boolean;
  uptime_seconds: number;
  tick_prices: Record<string, number>;
  timestamp: number;
}

export interface Decision {
  timestamp: string;
  trigger: string;
  btc_price: number;
  regime: string;
  fear_greed: number;
  portfolio_value: number;
  positions: string[];
  claude_assessment?: string;
  next_check_s?: number;
  wake_conditions?: string[];
  memory?: string;
  trades?: {
    action: string;
    ticker: string;
    price: number;
    confidence: number;
    reasons: string[];
  }[];
}

export interface Trade {
  timestamp: string;
  action: string;
  ticker: string;
  price: number;
  qty: number;
  value_usd: number;
  pnl_pct: number;
  held_hours: number;
  reason: string;
  regime: string;
  source: string;
  confidence: number;
  dry_run: string;
}

export interface TradeSummary {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  cumulative_pnl_pct: number;
}

export interface TSWeights {
  current_regime: string;
  mean_weights: Record<string, number>;
  total_trades: number;
  has_enough_data: boolean;
  cumulative_pnl_pct: number;
  recent_trades: unknown[];
  global_agents: Record<string, { alpha: number; beta: number; mean: number }>;
}
