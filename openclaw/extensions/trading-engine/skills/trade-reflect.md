---
name: trade-reflect
description: "Analyze recent trades to discover patterns, mistakes, and winning strategies. Updates Claude's memory for better future decisions."
metadata:
  {
    "openclaw":
      {
        "emoji": "🔄",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Reflect — Self-Learning from Trade History

Analyzes completed trades to discover recurring patterns (winning setups, common mistakes, regime-specific behaviors). Feeds insights back into Claude's agent_memory for improved future decisions.

Based on: ATLAS (arXiv:2510.15949) self-optimizing prompt strategy.

## When to Run

- After every 10-20 completed trades
- After a losing streak (3+ consecutive losses)
- When switching to a new market regime
- Weekly as a scheduled review

## Actions

### 1. Load Trade History

Read the online learner state file:

```bash
cat openclaw/extensions/trading-engine/python/data/models/online_learner.json
```

Or via Python:

```python
import json
from pathlib import Path

state = json.loads(Path("data/models/online_learner.json").read_text())
trades = state.get("trades", [])
regime_agents = state.get("regime_agents", {})
```

### 2. Compute Performance Metrics

For each regime, calculate:
- **Win Rate**: % of profitable trades
- **Avg Win vs Avg Loss**: size asymmetry
- **Avg Hold Time**: winners vs losers
- **Best/Worst 3 trades**: identify extremes

### 3. Pattern Analysis

Analyze the worst 3 and best 3 trades. Look for:

**Losing Patterns:**
- Entry during high funding rate + long → typically loses
- Entry right before regime change → wrong side of volatility shift
- Holding too long in ranging market → time decay eats profit

**Winning Patterns:**
- Entry on extreme fear + low funding → oversold bounce
- Quick exit when momentum z-score reverses → preserves profit
- Trend-following in low_volatility regime → ride momentum

### 4. Generate Reflection Summary

Create a structured reflection:

```
## Trade Reflection (DATE)

### Performance Summary
- Total trades: N, Win rate: X%, Cumulative PnL: Y%
- Best regime: REGIME_A (Win Rate Z%)
- Worst regime: REGIME_B (Win Rate W%)

### Patterns Discovered
1. [PATTERN]: [DESCRIPTION] → [ACTION TO TAKE]
2. [PATTERN]: [DESCRIPTION] → [ACTION TO TAKE]

### Strategy Adjustments
- In REGIME_A: [MORE/LESS] aggressive, [WIDER/TIGHTER] stops
- When funding > X: [AVOID/REDUCE] long exposure
- Hold time target: Xh for winners (currently Yh avg)

### Memory Update for Next Decision
[Concise 1-2 sentence summary for agent_memory field]
```

### 5. Save to Memory

Store reflection in Graphiti for long-term learning:

```json
{
  "tool": "add_memory",
  "input": {
    "group_id": "trading-patterns",
    "content": "[REFLECTION SUMMARY]"
  }
}
```

### 6. Update Agent Memory

The reflection summary (max 500 chars) should be injected into the runner's `_agent_memory` field. This appears in every subsequent Claude decision as "Your Previous Note".

To update manually, edit `data/runner_state.json`:
```json
{
  "entry_prices": {...},
  "agent_memory": "REFLECTION: [key insight]. AVOID: [losing pattern]. PREFER: [winning pattern]."
}
```

## Integration with Runner

The runner loads `_agent_memory` from `runner_state.json` on startup and passes it to every `MarketSnapshot.agent_memory` field. Claude sees this in the "Your Previous Note" section of the prompt.

The reflection loop:
```
Trades complete → /trade-reflect → pattern analysis → agent_memory update
  → next Claude decide() sees updated memory → better decisions
```

## Example Output

```
## Trade Reflection (2026-02-24)

### Performance Summary
- 15 trades, Win rate: 40%, Cumulative PnL: +2.3%
- Best regime: low_volatility_goldilocks (5/7 wins)
- Worst regime: high_volatility_stagflation (0/3 wins)

### Patterns Discovered
1. HIGH_VOL_ENTRY: All 3 losses in high_vol came from buying dips too early
   → Wait for z-score reversal confirmation before entering in high_vol
2. FUNDING_TRAP: 2 losses had funding_rate > 0.05% at entry
   → Skip longs when funding > 0.03%
3. QUICK_WINNERS: 4/6 winners were held < 48h, avg +3.2%
   → Tighten trailing after 48h if < 2% profit

### Memory Update
"REFLECTION: High_vol entries are -3.1% avg, avoid. Funding>0.03% + long = trap. Winners are quick (<48h). In low_vol_goldilocks, trend-follow aggressively."
```

## Notes

- Keep memory_update under 500 chars (runner truncates)
- Focus on actionable patterns, not statistics
- Regime-specific insights are most valuable
- Run before switching market conditions (e.g., before expected FOMC)
