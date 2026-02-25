---
name: trade-alert
description: "Send formatted trade alerts to configured channels when trading signals are detected"
metadata:
  {
    "openclaw":
      {
        "emoji": "🔔",
        "events": ["cron:trade-scan", "trade:signal"],
        "requires": { "config": ["plugins.entries.trading-engine"] },
        "install": [{ "id": "bundled", "kind": "bundled", "label": "Bundled with Trading Engine" }],
      },
  }
---

# Trade Alert Hook

Sends formatted trade alerts to configured messaging channels when the market scan detects actionable trading signals.

## What It Does

When a market scan (via cron or manual `/trade-scan`) completes:

1. **Filters signals** - Only alerts on strong signals (action > 0.3 or < -0.3)
2. **Formats message** - Creates a structured alert with ticker, direction, strength, and key indicators
3. **Sends to channel** - Delivers alert to the configured channel (Telegram, Discord, etc.)
4. **Logs to memory** - Records alert in Graphiti for tracking

## Alert Format

```
🚨 Trading Signal Detected

📈 BUY AAPL @ $185.20
   Strength: 0.72 (Strong)
   RSI: 45.2 | MACD: +0.31
   Model: PPO(0.45) A2C(0.35) DDPG(0.20)

📉 SELL TSLA @ $248.50
   Strength: -0.65 (Strong)
   RSI: 72.8 | MACD: -0.18
   Model: PPO(0.40) A2C(0.35) DDPG(0.25)

⏰ 2026-02-12 10:15 ET
💰 Portfolio: $102,450 (+2.45%)
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `minSignalStrength` | number | 0.3 | Minimum absolute action value to trigger alert |
| `channels` | string[] | ["last"] | Channel IDs to send alerts to |
| `includeIndicators` | boolean | true | Include technical indicator values |

## Disabling

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "trade-alert": { "enabled": false }
      }
    }
  }
}
```
