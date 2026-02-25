---
name: trade-exit
description: "Exit management framework for open positions. Agent learns optimal exit timing through RL feedback loop."
metadata:
  {
    "openclaw":
      {
        "emoji": "🚪",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Exit — Intelligent Position Exit Management

Defines the exit decision framework for the Claude trading agent. Unlike hard-coded rules, the agent uses these as starting guidelines and ADAPTS through Thompson Sampling feedback. The safety layer is the last resort only.

## Exit Hierarchy (Agent > Safety)

```
Layer 1: Claude Agent (YOU) — proactive exits based on analysis
Layer 2: Safety Rules (automatic) — hard stop, trailing stop, time stop, profit protect
```

If the safety layer exits your position, it means you failed to manage it. Each safety exit feeds a NEGATIVE signal into RL, penalizing the signal categories you used at entry.

## Exit Decision Framework

### Thesis Invalidation Exit (highest priority)
- The reason you entered no longer holds (support broke, regime changed, catalyst failed)
- Action: EXIT immediately, regardless of PnL
- Example: Entered on "extreme fear contrarian bounce" but fear deepened further with volume confirmation

### Profit Management
- **Target**: +3-8% on swing trades (align with market regime volatility)
- **Scale out**: Consider partial exits at +3%, +5%, +8% rather than all-or-nothing
- **Peak protection**: If PnL was +2% and drops to +0.5%, momentum is lost — EXIT
- **Never let a winner become a loser**: Once +2%, protect with mental trailing stop

### Loss Management
- **Cut early**: If PnL < -3% with no recovery catalyst visible, EXIT
- **Don't anchor**: Your entry price is irrelevant. Only current conditions matter
- **Safety backstop**: Hard stop at -5% (automatic, you won't be asked)

### Time-Based Exit
- **Ranging market**: If held >12h with PnL < +1%, it's dead money — EXIT
- **Maximum hold**: 48h absolute max for swing trades (automatic safety)
- **Regime shift**: If regime changed since entry, re-evaluate immediately

## Signal Weights for RL

When the agent exits a position, the Thompson Sampling system records:
- Which signal categories were used at entry
- Whether the trade was profitable
- The regime at entry and exit

This creates a feedback loop:
```
Entry signals → Trade outcome → Beta(alpha, beta) update → Future weight adjustment
```

Signals that led to losing trades get downweighted. Signals that led to winners get upweighted. The agent learns WHICH signals to trust in WHICH regime.

## Self-Assessment After Each Exit

After closing a position, update your `memory_update` with:
1. Why did you exit? (thesis invalid / target hit / loss cut / time stop)
2. What would you do differently? (entered too early / sized too large / held too long)
3. One actionable rule for next similar setup

## Integration

### Reading Position State
Positions are shown in every MarketSnapshot with:
- PnL %, peak PnL %, held hours, entry price, current price
- Safety thresholds (hard SL, trail activation, trail width)

### Issuing SELL Decision
```json
{
  "decisions": [
    {
      "ticker": "BTC/USDT",
      "action": "SELL",
      "confidence": 0.8,
      "reasoning": "Thesis invalidated: entered on extreme fear bounce but BTC broke below $61K support with volume. Regime still unknown. Cutting at -2% before safety hits -5%.",
      "signal_weights": {"momentum": -0.8, "sentiment": 0.3}
    }
  ]
}
```

### RL Feedback on Exit
The runner handles RL recording automatically:
```
position.exit event → online_learner.record_trade() → Beta distribution update
```

## Key Principle

The best traders are defined by their exits, not their entries. An agent that enters with 50% win rate but manages exits well (small losses, large wins) will be profitable. An agent that enters with 70% win rate but lets winners become losers will lose money.

Your R:R ratio matters more than your win rate:
- Target: Average Win >= 2x Average Loss
- This means: cut losers at -2% to -3%, let winners run to +4% to +8%
