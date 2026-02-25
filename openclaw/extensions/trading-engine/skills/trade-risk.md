---
name: trade-risk
description: "Risk management framework for position sizing, portfolio exposure, and drawdown control. Agent adapts sizing through RL."
metadata:
  {
    "openclaw":
      {
        "emoji": "🛡️",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Risk — Adaptive Risk Management

Risk management framework that the Claude agent uses to size positions and control drawdowns. Parameters are starting points — the agent learns optimal sizing per regime through Thompson Sampling.

## Position Sizing Rules

### Base Sizing
- **Single position**: 5-20% of portfolio (`position_pct` in decision JSON)
- **Total exposure**: Max 60% of portfolio in open positions
- **Cash reserve**: Always keep minimum 40% cash for opportunities

### Adaptive Sizing by Confidence
| Confidence | Suggested Size |
|------------|---------------|
| 0.3 - 0.4 | 5% (minimum, low conviction) |
| 0.4 - 0.6 | 7-10% (moderate) |
| 0.6 - 0.8 | 12-15% (high conviction) |
| 0.8 - 1.0 | 15-20% (very high, rare) |

### Regime Scaling
- `macro_exposure_scale` from MacroRegimeDetector (0.5-1.0)
- In stagflation/deflation: multiply all sizes by exposure_scale
- In goldilocks/reflation: normal sizing

### TS Reliability Scaling
- If all TS signals < 40% reliability: reduce max size to 10%
- If best TS signal > 60%: can use full sizing range
- If < 3 total trades in current regime: use minimum sizing (5-7%)

## Drawdown Control

### Per-Position
| Rule | Threshold | Action |
|------|-----------|--------|
| Hard stop | -5% | Automatic exit (safety layer) |
| Agent stop | -3% (no catalyst) | Agent should exit proactively |
| Profit protect | Peak > +2%, now < +0.5% | Automatic exit (safety layer) |
| Time stop | >48h, PnL < +1% | Automatic exit (safety layer) |

### Portfolio-Level
| Rule | Threshold | Action |
|------|-----------|--------|
| Portfolio DD | -10% from peak | Halt all new BUYs until recovery |
| Daily loss | -3% | Reduce position sizes by 50% |
| Consecutive losses | 3+ | Trigger /trade-reflect, reduce sizes |

## Correlation Risk

- Never hold more than 2 highly correlated positions (e.g., BTC + ETH are ~0.85 correlated)
- If BTC + ETH both open: treat combined exposure as single position for DD calc
- SOL/PAXG may provide diversification — check correlation before combining

## RL Feedback Loop

Risk decisions feed into Thompson Sampling:
1. Position size → trade PnL → RL record
2. Oversized losing trades penalize the signals used more heavily
3. Well-sized winners reward the signals used
4. Over time, the agent learns: "In regime X, smaller sizes with signal Y work better"

## Integration

### In Decision JSON
```json
{
  "decisions": [{
    "ticker": "BTC/USDT",
    "action": "BUY",
    "position_pct": 0.07,
    "confidence": 0.45,
    "reasoning": "Low confidence contrarian play, sizing at 7% due to TS signals all <40%"
  }]
}
```

### Checking Portfolio State
Every MarketSnapshot shows:
- Portfolio value, cash, existing positions
- Each position: entry price, PnL %, peak PnL %, held hours
