---
name: trade-status
description: "View crypto portfolio status: positions, PnL, regime, gate status, Thompson Sampling weights, and recent trades"
metadata:
  {
    "openclaw":
      {
        "emoji": "📊",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Status — Crypto Portfolio Dashboard

View current portfolio state: open positions, PnL, market regime, AdaptiveGate status, Thompson Sampling posteriors, and recent trade history.

## Actions

### 1. Quick Status

```bash
cd openclaw/extensions/trading-engine/python
python3 -c "
import json
from pathlib import Path

# Online Learner
ol = json.loads(Path('data/models/online_learner.json').read_text())
print('=== Trading Performance ===')
print(f'Total trades: {ol.get(\"total_trades\", 0)}')
print(f'Cumulative PnL: {ol.get(\"total_pnl\", 0)*100:+.2f}%')
print()

# Global weights
ga = ol.get('global_agents', {})
print('=== Thompson Sampling Weights ===')
for name, data in ga.items():
    mean = data['alpha'] / (data['alpha'] + data['beta'])
    print(f'  {name}: {mean:.3f} (α={data[\"alpha\"]:.1f}, β={data[\"beta\"]:.1f}, trades={data[\"total_trades\"]})')
print()

# Regime stats
print('=== Regime Performance ===')
for regime, count in ol.get('regime_trade_counts', {}).items():
    if regime == '__global__': continue
    print(f'  {regime}: {count} trades')
print()

# Runner state
rs = Path('data/runner_state.json')
if rs.exists():
    rstate = json.loads(rs.read_text())
    ep = rstate.get('entry_prices', {})
    if ep:
        print('=== Open Positions ===')
        for ticker, price in ep.items():
            print(f'  {ticker}: entry @ \${price:.2f}')
    else:
        print('=== No Open Positions ===')
    mem = rstate.get('agent_memory', '')
    if mem:
        print(f'\nAgent Memory: {mem}')

# Recent trades
trades = ol.get('trades', [])[-5:]
if trades:
    print('\n=== Recent Trades ===')
    for t in trades:
        print(f'  {t[\"ticker\"]}: {t[\"pnl_pct\"]*100:+.2f}% ({t[\"regime\"]})')
"
```

### 2. Broker Positions (Live)

```python
from brokers.binance import BinanceBroker
broker = BinanceBroker()
result = broker.connect()
print(f"Portfolio: ${result.get('portfolio_value', 0):,.2f}")
print(f"Mode: {result.get('mode')}, Market: {result.get('market')}")

positions = broker.get_positions_detail()
for ticker, info in positions.items():
    print(f"{ticker}: {info['qty']:.6f} @ ${info['current_price']:.2f} = ${info['market_value']:.2f}")
```

### 3. Gate Status

The AdaptiveGate shows Claude's self-scheduled monitoring:
- Next check time
- Active wake conditions
- Recent z-score triggers

Available from `runner.adaptive_gate.get_status()` during runtime.

### 4. Full Report Format

```
=== Portfolio Status ===
Total Value:     $329,000
Cash:            $329,000 (100%)
Positions:       0

=== Market Regime ===
Crypto:          low_volatility
Macro:           GOLDILOCKS
Combined:        low_volatility_goldilocks
Exposure Scale:  1.0x

=== Claude Agent ===
Status:          READY (Sonnet, OpenClaw OAuth)
Circuit Breaker: CLOSED (0 failures)
Next Check:      1800s (30 min)
Wake Conditions: 3 active

=== Thompson Sampling ===
Total Trades:    15
Cumulative PnL:  +2.30%
Global Weights:
  momentum:      0.55 (α=8.2, β=6.8)
  funding:       0.62 (α=9.1, β=5.6)
  regime:        0.48 (α=7.0, β=7.6)

=== Recent Trades ===
1. BTC/USDT: +3.2% (low_vol_goldilocks, 36h held)
2. ETH/USDT: -1.8% (high_vol_stagflation, 12h held)
3. SOL/USDT: +5.1% (low_vol_goldilocks, 48h held)
```

## Data Sources

- **Online Learner**: `data/models/online_learner.json`
- **Runner State**: `data/runner_state.json` (entry_prices, agent_memory)
- **Broker**: Binance ccxt (live balance + positions)
- **Gate**: `runner.adaptive_gate.get_status()` (runtime only)
