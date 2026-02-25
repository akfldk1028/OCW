---
name: risk-guardian
description: "Monitor open positions for stop-loss/take-profit triggers and enforce daily loss limits"
metadata:
  {
    "openclaw":
      {
        "emoji": "🛡️",
        "events": ["trade:executed", "cron:tick"],
        "requires": { "config": ["plugins.entries.trading-engine"] },
        "install": [{ "id": "bundled", "kind": "bundled", "label": "Bundled with Trading Engine" }],
      },
  }
---

# Risk Guardian Hook

Real-time risk monitoring that automatically enforces stop-loss, take-profit, and daily loss limits.

## What It Does

Continuously monitors open positions and portfolio health:

1. **Stop-Loss Check** - Closes positions that hit the stop-loss threshold (-0.5% default)
2. **Take-Profit Check** - Closes positions that hit the take-profit target (+1.0% default)
3. **Daily Loss Limit** - Halts all trading if daily P&L exceeds maximum loss
4. **Exposure Check** - Warns if portfolio exposure exceeds limits
5. **Alert Generation** - Sends notifications on any risk event

## Risk Events

### Auto-Close (Stop-Loss)
```
🛡️ STOP-LOSS TRIGGERED

❌ SOLD TSLA @ $245.00
   Entry: $248.50 | Loss: -1.41%
   Reason: Stop-loss at -0.5% exceeded

⚠️ Daily P&L: -$350 (-0.35%)
```

### Auto-Close (Take-Profit)
```
🛡️ TAKE-PROFIT TRIGGERED

✅ SOLD AAPL @ $187.50
   Entry: $185.00 | Profit: +1.35%
   Reason: Take-profit at +1.0% reached

💰 Daily P&L: +$625 (+0.63%)
```

### Trading Halt
```
🚫 TRADING HALTED

Daily loss limit reached: -$1,500 (-1.5%)
No new trades will be executed until next trading day.
Resume: Tomorrow 9:30 AM ET
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `takeProfitPct` | number | 0.01 | Take-profit threshold (1%) |
| `stopLossPct` | number | -0.005 | Stop-loss threshold (-0.5%) |
| `maxDailyLossPct` | number | -0.02 | Max daily loss before halt (-2%) |
| `checkIntervalMs` | number | 60000 | Position check interval (1 min) |

## Disabling

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "risk-guardian": { "enabled": false }
      }
    }
  }
}
```

**Warning**: Disabling risk-guardian removes automatic stop-loss protection. Only disable if you have alternative risk management in place.
