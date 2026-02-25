---
name: trade-execute
description: "Execute crypto trades on Binance via Claude Agent decisions with safety layer and position tracking"
metadata:
  {
    "openclaw":
      {
        "emoji": "⚡",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Execute — Binance Crypto Order Execution

Executes BUY/SELL orders on Binance (spot or futures) based on Claude Agent decisions. Includes position tracking, safety layer (stop-loss + trailing stop), and RL feedback loop.

## Architecture

```
Claude Decision → broker.execute_decisions() → Binance API
                → position_tracker.track()     → Safety layer (30s)
                → event_bus.publish()           → RL feedback on exit
```

## Execution Modes

| Mode | Command | Description |
|------|---------|-------------|
| Testnet Spot | `python3 binance/main.py --testnet` | Paper trading, no real orders |
| Testnet Futures | `python3 binance/main.py --testnet --futures --leverage 3` | Futures with leverage |
| Live Spot | Set `LIVE_TRADING=true` in .env | Real orders |
| Live Futures | Set `LIVE_TRADING=true` + `--futures` | Real futures orders |

## Safety Layer (Automatic)

Every 30 seconds, the position tracker checks:
- **Hard Stop Loss**: -5% from entry → immediate exit
- **Trailing Stop**: After +8% profit, trail at -12% from peak
- **Profit Protect**: Peak > +2%, drops to < +0.5% → exit
- **Time Stop**: Held > 48h with PnL < +1% → exit
- Claude manages proactive exits BEFORE safety triggers (see /trade-exit)

## Actions

### 1. Start Testnet Engine

```bash
cd openclaw/extensions/trading-engine/python
python3 binance/main.py --testnet
```

### 2. Manual BUY (via broker)

```python
from brokers.binance import BinanceBroker
broker = BinanceBroker()
broker.connect()

decisions = [{
    "ticker": "BTC/USDT",
    "action": "BUY",
    "qty": 0.001,
    "price": 0,  # market order
}]
result = broker.execute_decisions(decisions, dry_run=True)
print(result)
```

### 3. Check Position State

```python
import json
from pathlib import Path

# Runner state (entry prices)
state = json.loads(Path("data/runner_state.json").read_text())
print("Entry prices:", state.get("entry_prices", {}))

# Broker positions
detail = broker.get_positions_detail()
for ticker, info in detail.items():
    print(f"{ticker}: {info['qty']} @ ${info['current_price']}")
```

## Risk Rules

| Rule | Value | Source |
|------|-------|--------|
| Hard Stop Loss | -5% | `config.CRYPTO_RISK_CONFIG` |
| Trail Activation | +8% | `SWING_BLEND_CONFIG.trail_activation_pct` |
| Trailing Stop | -12% from peak | `SWING_BLEND_CONFIG.trail_pct` |
| Profit Protect | peak>+2%, now<+0.5% | `position_tracker._check_safety_rules` |
| Time Stop | >48h, PnL<+1% | `config.CRYPTO_RISK_CONFIG.max_hold_hours` |
| Max Position | 20% | `SWING_BLEND_CONFIG.position_pct` |
| Portfolio DD | -10% | `SWING_BLEND_CONFIG.portfolio_dd_trigger` |
| Maker Fee | 0.02% | Binance Futures |

## Order Flow

1. Claude decides BUY with position_pct
2. `_convert_claude_decisions()` → limit order at current price (maker fee)
3. `broker.execute_decisions()` → Binance API
4. Only if status=submitted/dry_run → `position_tracker.track()`
5. `event_bus.publish("decision.signal")` → saves TS signals for RL
6. Safety layer monitors every 30s

## .env Required

```
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_PAPER=true
LIVE_TRADING=false
CLAUDE_CONFIG_DIR=path/to/claude/data
```

## Notes

- Orders default to limit (maker fee 0.02% vs taker 0.04%)
- Position state persisted to `data/runner_state.json` for restart recovery
- Claude's `next_check_seconds` determines when to re-evaluate
- All trades are dry_run unless `LIVE_TRADING=true`
