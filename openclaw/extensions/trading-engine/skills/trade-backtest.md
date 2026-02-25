---
name: trade-backtest
description: "Backtest the RL ensemble strategy on historical data and generate performance reports"
metadata:
  {
    "openclaw":
      {
        "emoji": "📈",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Backtest

Run backtests on historical data to validate strategy performance.

## Overview

This skill runs the current (or specified) RL ensemble model against historical market data, simulating trades with the risk management system, and generates comprehensive performance reports.

## Inputs to collect

- **Tickers** (optional): Stock symbols to backtest. Default: configured watchlist
- **Start Date** (optional): Backtest start date (YYYY-MM-DD). Default: 1 year ago
- **End Date** (optional): Backtest end date (YYYY-MM-DD). Default: today
- **Initial Capital** (optional): Starting portfolio value. Default: $100,000

## Actions

### 1. Run Backtest

```json
{
  "tool": "trading_backtest",
  "input": {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2025-02-01",
    "end_date": "2026-02-01",
    "initial_capital": 100000
  }
}
```

### 2. Performance Report

```
=== Backtest Results ===
Period:          2025-02-01 ~ 2026-02-01 (252 trading days)
Initial Capital: $100,000
Final Value:     $113,200

=== Returns ===
Total Return:    +13.2%
Annual Return:   +13.2%
Sharpe Ratio:    1.24
Sortino Ratio:   1.68
Max Drawdown:    -6.8%

=== Benchmark Comparison ===
S&P 500 Return:  +10.1%
Alpha:           +3.1%
Beta:            0.72

=== Trade Statistics ===
Total Trades:    487
Win Rate:        58.3%
Avg Win:         +0.82%
Avg Loss:        -0.41%
Profit Factor:   2.01
Avg Hold Time:   3.2 days

=== Model Weights (Average) ===
PPO:  0.42
A2C:  0.38
DDPG: 0.20
```

### 3. Save Results to Memory

```json
{
  "tool": "add_memory",
  "input": {
    "group_id": "trading-patterns",
    "content": "Backtest {start}~{end}: Sharpe {sharpe}, Return {return}%, MaxDD {maxdd}%. Alpha vs SPY: {alpha}%"
  }
}
```

## Interpretation Guide

| Metric | Good | Great | Target |
|--------|------|-------|--------|
| Sharpe | > 1.0 | > 1.5 | > 1.0 |
| Max Drawdown | < 10% | < 5% | < 10% |
| Win Rate | > 55% | > 60% | > 55% |
| Profit Factor | > 1.5 | > 2.0 | > 1.5 |
| Alpha | > 0% | > 5% | > 0% |

## Notes

- Backtests do NOT account for slippage, commissions, or market impact
- Past performance does not guarantee future results
- Always validate with paper trading before going live
- Consider running multiple backtests with different time periods
