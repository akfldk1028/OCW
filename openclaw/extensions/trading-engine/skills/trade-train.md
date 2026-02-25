---
name: trade-train
description: "Train or retrain the RL ensemble model (PPO+A2C+DDPG) on recent market data with performance validation"
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Train

Train or retrain the ensemble RL model on recent market data.

## Overview

This skill triggers incremental training of the PPO+A2C+DDPG ensemble model. It downloads recent data, trains each sub-model, validates via backtest, and replaces the production model only if the new version performs better.

## Inputs to collect

- **Tickers** (optional): Stock symbols to train on. Default: configured watchlist
- **Lookback Days** (optional): How many days of historical data. Default: 365
- **Timesteps** (optional): Training timesteps per model. Default: 50,000
- **Force** (optional): Skip validation and force model replacement. Default: false

## Actions

### 1. Trigger Training

```json
{
  "tool": "trading_train",
  "input": {
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "lookback_days": 365,
    "total_timesteps": 50000
  }
}
```

### 2. Monitor Progress

Training runs in a background thread. Check status:

```json
{
  "tool": "trading_status",
  "input": {}
}
```

Status will show:
- `training` - Training in progress (shows progress %)
- `validating` - Running validation backtest
- `ready` - Training complete, model deployed
- `rejected` - New model underperformed, kept previous

### 3. Validation

After training completes, the system automatically:
1. Runs backtest on held-out test data (last 20% of period)
2. Compares Sharpe ratio with current production model
3. Only deploys if new Sharpe > old Sharpe * 0.95 (5% tolerance)

### 4. Record to Memory

```json
{
  "tool": "add_memory",
  "input": {
    "group_id": "trading-patterns",
    "content": "Model retrained on {date}. Tickers: {tickers}. New Sharpe: {sharpe}. Models: PPO(w={ppo_w}), A2C(w={a2c_w}), DDPG(w={ddpg_w}). Deployed: {yes/no}"
  }
}
```

## Training Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| Algorithm | PPO+A2C+DDPG | Ensemble of 3 RL algorithms |
| Timesteps | 50,000 | Per-model training steps |
| Learning Rate | 3e-4 | Adam optimizer LR |
| Observation | OHLCV + 12 indicators + sentiment | Per-ticker features |
| Action Space | [-1, 1] continuous | Sell-to-buy per ticker |
| Reward | Sharpe-inspired | Daily return / rolling std |

## Schedule

- Automatic weekly retraining: Friday 8PM ET (via cron)
- Manual trigger: `/trade-train` anytime
- Emergency retrain: after significant market regime change
