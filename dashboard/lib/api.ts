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
  side?: string;
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
  realized_pnl_usd?: number;
  today_pnl_pct?: number;
  today_trades?: number;
}

export interface WakeCondition {
  metric: string;
  operator: string;
  threshold: number;
  reason: string;
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
  wake_conditions?: (WakeCondition | string)[];
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
  entry_price?: number; // attached to SELL trades (paired from matching BUY)
}

export interface TradeSummary {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  cumulative_pnl_pct: number;
}

export interface BetaInfo {
  name: string;
  alpha: number;
  beta: number;
  mean: number;
  std: number;
  total_trades: number;
}

export interface GroupSummary {
  group_beta: BetaInfo;
  signals: Record<string, BetaInfo>;
}

export interface RegimeSignalInfo {
  mean: number;
  total_trades: number;
}

export interface RegimeInfo {
  trade_count: number;
  using_own_weights: boolean;
  group_means: Record<string, number>;
  meta_param_means: Record<string, number>;
  signal_reliability: Record<string, RegimeSignalInfo>;
}

export interface TSWeights {
  current_regime: string;
  mean_weights: Record<string, number>;
  signal_reliability: Record<string, number>;
  group_weights: Record<string, number>;
  meta_params: Record<string, number>;
  meta_params_global: Record<string, number>;
  meta_param_betas: Record<string, BetaInfo>;
  meta_param_betas_global: Record<string, BetaInfo>;
  group_summary: Record<string, GroupSummary>;
  regime_info: Record<string, RegimeInfo>;
  total_trades: number;
  has_enough_data: boolean;
  cumulative_pnl_pct: number;
  recent_trades: unknown[];
  global_agents: Record<string, { alpha: number; beta: number; mean: number }>;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceHistory {
  ticker: string;
  timeframe: string;
  candles: Candle[];
}

export interface EquityPoint {
  time: string;
  value: number;
}

export interface EquityCurve {
  points: EquityPoint[];
}

export interface DayPerf {
  date: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  pnl_pct: number;
  pnl_usd: number;
  best_trade_pct: number;
  worst_trade_pct: number;
  cumulative_pnl_pct: number;
}

export interface DailyPnlData {
  days: DayPerf[];
  summary: {
    total_days: number;
    total_trades: number;
    total_pnl_pct: number;
    total_pnl_usd: number;
    current_portfolio_value: number;
    initial_balance: number;
  };
}
