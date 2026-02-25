---
name: trade-analyze
description: "PnL analysis with regime breakdown, win/loss patterns, and RL weight effectiveness"
metadata:
  {
    "openclaw":
      {
        "emoji": "📉",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Analyze — PnL & Regime Performance

Analyze completed trades with regime-level breakdown. Shows which signal categories (momentum, funding, etc.) are working in which regimes via Thompson Sampling weights.

## Actions

### 1. Quick PnL Summary

```bash
cd openclaw/extensions/trading-engine/python
python3 -c "
import json, csv
from pathlib import Path

# Trades CSV
csv_path = Path('binance/logs/trades.csv')
if csv_path.exists():
    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
    buys = [r for r in reader if r['action'] == 'BUY']
    sells = [r for r in reader if r['action'] == 'SELL']
    total_pnl = sum(float(r.get('pnl_pct', 0)) for r in sells)
    wins = [r for r in sells if float(r.get('pnl_pct', 0)) > 0]
    losses = [r for r in sells if float(r.get('pnl_pct', 0)) < 0]
    print(f'Trades: {len(buys)} BUY, {len(sells)} SELL')
    print(f'Win Rate: {len(wins)}/{len(sells)} ({len(wins)/max(len(sells),1)*100:.0f}%)')
    print(f'Total PnL: {total_pnl*100:+.2f}%')
    if wins:
        avg_win = sum(float(r['pnl_pct']) for r in wins) / len(wins)
        print(f'Avg Win: {avg_win*100:+.2f}%')
    if losses:
        avg_loss = sum(float(r['pnl_pct']) for r in losses) / len(losses)
        print(f'Avg Loss: {avg_loss*100:+.2f}%')
else:
    print('No trades.csv found')
"
```

### 2. Regime Breakdown

```bash
cd openclaw/extensions/trading-engine/python
python3 -c "
import json, csv
from pathlib import Path
from collections import defaultdict

csv_path = Path('binance/logs/trades.csv')
if not csv_path.exists():
    print('No trades found'); exit()

with open(csv_path) as f:
    sells = [r for r in csv.DictReader(f) if r['action'] == 'SELL']

by_regime = defaultdict(list)
for r in sells:
    by_regime[r.get('regime', 'unknown')].append(float(r.get('pnl_pct', 0)))

print('=== Performance by Regime ===')
for regime, pnls in sorted(by_regime.items()):
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    print(f'{regime}: {len(pnls)} trades, WR={wins}/{len(pnls)}, PnL={total*100:+.2f}%')
"
```

### 3. TS Weight Analysis

Shows which signal categories Thompson Sampling considers reliable per regime.

```bash
cd openclaw/extensions/trading-engine/python
python3 -c "
import json
from pathlib import Path

ol_path = Path('binance/models/online_learner.json')
if not ol_path.exists():
    print('No online_learner.json found'); exit()

state = json.loads(ol_path.read_text())
print('=== Thompson Sampling Weights ===')

# Global
ga = state.get('global_agents', {})
print('\nGlobal:')
for name, data in sorted(ga.items(), key=lambda x: -x[1]['alpha']/(x[1]['alpha']+x[1]['beta'])):
    mean = data['alpha'] / (data['alpha'] + data['beta'])
    print(f'  {name}: {mean:.3f} (trades={data[\"total_trades\"]})')

# Per regime
ra = state.get('regime_agents', {})
for regime, agents in sorted(ra.items()):
    if regime == '__global__': continue
    print(f'\n{regime}:')
    for name, data in sorted(agents.items(), key=lambda x: -x[1]['alpha']/(x[1]['alpha']+x[1]['beta'])):
        mean = data['alpha'] / (data['alpha'] + data['beta'])
        print(f'  {name}: {mean:.3f} (trades={data[\"total_trades\"]})')
"
```

### 4. Alerts Log

```bash
cd openclaw/extensions/trading-engine/python
tail -20 binance/logs/alerts.jsonl 2>/dev/null || echo "No alerts yet"
```

## Output Format

```
=== Trade Analysis ===
Trades: 25 BUY, 18 SELL
Win Rate: 8/18 (44%)
Total PnL: +5.30%
Avg Win: +4.2%, Avg Loss: -2.1%

=== Performance by Regime ===
low_volatility_goldilocks: 10 trades, WR=6/10, PnL=+6.2%
high_volatility_reflation: 5 trades, WR=1/5, PnL=-2.1%
ranging_deflation: 3 trades, WR=1/3, PnL=+1.2%

=== TS Weights (low_volatility_goldilocks) ===
  momentum: 0.65 (trades=8)
  funding_rate: 0.58 (trades=6)
  regime: 0.52 (trades=10)
  sentiment: 0.41 (trades=4)
```
