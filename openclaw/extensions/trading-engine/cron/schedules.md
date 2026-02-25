# Trading Engine Cron Schedules

## Primary: Python APScheduler (server-side)

The **APScheduler** inside `server.py` is the primary scheduler for all automated trading.
It runs when `ENABLE_CRON=true` and handles position-aware decisions, risk management,
and model retraining autonomously.

### 1. Position-Aware Decision (Every 15 min)

```
cron_decide — every 15 min, 9:00-16:00 ET weekdays
```

- **What**: Fetches current Alpaca positions, runs full **v3 multi-agent pipeline** with EXIT management
- **Pipeline**: MarketAgent (regime+sector) → QuantAgent (XGBoost) → Synthesizer (weighted voting)
- **EXIT rules**: SWING_EXIT_CONFIG — stop_loss=-3%, take_profit=+5%, consecutive_miss_limit=3
- **Order priority**: SELLs submitted before BUYs (frees buying power)
- **Orphan detection**: Held positions not in scan candidates are evaluated for exit

### 2. Risk Check (Every 5 min)

```
cron_risk_check — every 5 min, 9:00-16:00 ET weekdays
```

- **What**: Checks all held positions for TP/SL/trailing stop triggers
- **Thresholds**: RISK_CONFIG — take_profit=+4%, stop_loss=-2.5%, ATR-based dynamic if enabled
- **Action**: Auto-submits SELL orders via Alpaca when triggered
- **Independent from**: `cron_decide` (15-min) — provides faster exit for sudden moves
- **Manual trigger**: `trading_risk_check` 도구 or `POST /risk/check`

### 3. Dynamic RL Retraining (Weekly Sunday)

```
cron_weekly_retrain — Sunday 02:00 ET
```

- **What**: Retrains XGBoost (18-month window) + RL ensemble with dynamic tickers
- **RL tickers**: Top 15 from recent EventBus decision.signal events (not fixed TRAIN_CONFIG)

## Secondary: OpenClaw Cron (assistant-side, optional)

OpenClaw cron jobs invoke **assistant tools** for supplementary actions. These are **not required**
for core trading — APScheduler handles that. Use these for:
- Supplementary analysis via the assistant
- Graphiti memory recording
- Notification/reporting to users

> **Note**: `trading_decide` 도구는 v3 파이프라인(`/decide/v3`)을 호출합니다.
> 포지션 정보가 없으면 서버가 Alpaca에서 자동으로 가져옵니다.

### A. Market Scan (Every 15 min during NYSE hours)

```json
{
  "id": "trade-scan-15m",
  "name": "Market Scan (15m)",
  "description": "Scan markets via v3 multi-agent pipeline every 15 minutes during NYSE hours",
  "enabled": true,
  "schedule": {
    "kind": "cron",
    "expr": "*/15 9-16 * * 1-5",
    "tz": "America/New_York"
  },
  "sessionTarget": "main",
  "wakeMode": "now",
  "payload": {
    "kind": "agentTurn",
    "message": "Run autonomous trading decision: Use the trading_decide tool. It runs the v3 multi-agent pipeline (MarketAgent → QuantAgent → Synthesizer) with position awareness and EXIT management. Report all BUY/SELL decisions with confidence levels and position sizes. Save decisions to Graphiti memory (group_id: trading-portfolio).",
    "timeoutSeconds": 120,
    "deliver": true,
    "bestEffortDeliver": true
  }
}
```

- **When**: Weekdays 9:00-16:00 ET (NYSE trading hours), every 15 minutes
- **Korea time**: 23:00-06:00 KST

### B. Daily Portfolio Rebalancing (Market open)

```json
{
  "id": "trade-rebalance-daily",
  "name": "Daily Rebalancing",
  "description": "Review portfolio and rebalance positions at market open",
  "enabled": true,
  "schedule": {
    "kind": "cron",
    "expr": "0 9 * * 1-5",
    "tz": "America/New_York"
  },
  "sessionTarget": "main",
  "wakeMode": "now",
  "payload": {
    "kind": "agentTurn",
    "message": "Daily portfolio rebalancing: 1) Use trading_status to check current portfolio state. 2) Use trading_decide to get v3 pipeline decisions. 3) Compare current positions with recommendations. 4) If significant misalignment (>10% weight difference), suggest rebalancing trades. Do NOT auto-execute - present the rebalancing plan for review.",
    "timeoutSeconds": 180,
    "deliver": true,
    "bestEffortDeliver": true
  }
}
```

- **When**: Weekdays 9:00 ET (market open)
- **Korea time**: 23:00 KST

### C. Weekly Model Retraining (Friday after close)

```json
{
  "id": "trade-retrain-weekly",
  "name": "Weekly Model Retrain",
  "description": "Retrain the RL ensemble model with latest market data",
  "enabled": true,
  "schedule": {
    "kind": "cron",
    "expr": "0 20 * * 5",
    "tz": "America/New_York"
  },
  "sessionTarget": "isolated",
  "wakeMode": "now",
  "payload": {
    "kind": "agentTurn",
    "message": "Weekly model retraining: 1) Use trading_quant_retrain for XGBoost (18-month window). 2) Use trading_train to retrain the RL ensemble with 50000 timesteps. After training completes, check trading_status for the new model performance. Report old vs new Sharpe ratio.",
    "timeoutSeconds": 600,
    "deliver": true,
    "bestEffortDeliver": true
  }
}
```

- **When**: Fridays 8:00 PM ET (after market close)
- **Korea time**: Saturday 10:00 AM KST

## Registration

OpenClaw cron jobs can be registered via CLI or API:

```bash
# List current cron jobs
openclaw cron list

# Add a cron job (use the JSON above)
openclaw cron add --file schedules/trade-scan-15m.json

# Manual trigger for testing
openclaw cron run trade-scan-15m

# Disable a cron job
openclaw cron disable trade-scan-15m
```

## Timezone Notes

| Event | ET (New York) | KST (Seoul) |
|-------|---------------|-------------|
| Market Open | 09:30 | 23:30 |
| Scan Start | 09:00 | 23:00 |
| Scan End | 16:00 | 06:00 (+1d) |
| Market Close | 16:00 | 06:00 (+1d) |
| Weekly Retrain (OC) | Fri 20:00 | Sat 10:00 |
| Risk Check (APS, 5m) | 09:00-16:00 | 23:00-06:00 |
| Position Decide (APS, 15m) | 09:00-16:00 | 23:00-06:00 |
| RL Retrain (APS) | Sun 02:00 | Sun 16:00 |
