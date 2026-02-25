---
name: trade-scan
description: "Crypto market scan: regime detection, derivatives signals, Claude Agent decision with adaptive gate status"
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "config": ["plugins.entries.trading-engine"] },
      },
  }
---

# Trade Scan — Crypto Market Analysis

Scans current crypto market state: HMM regime, derivatives signals (funding/OI/CVD/L-S ratio), macro regime, and Thompson Sampling posteriors. Triggers a Claude Agent decision if warranted.

## Architecture

```
AdaptiveGate (4-layer) → MarketSnapshot → Claude Sonnet → BUY/SELL/HOLD
                                              ↑              ↓
                                              └── TS feedback ←┘
```

Claude is the PRIMARY trader. Rule-based pipeline is fallback only.

## Actions

### 1. Check System Status

```bash
cd openclaw/extensions/trading-engine/python
python3 -c "
import json
from pathlib import Path

# Online learner status
state = json.loads(Path('data/models/online_learner.json').read_text())
print(f'Total trades: {state.get(\"total_trades\", 0)}')
print(f'Cumulative PnL: {state.get(\"total_pnl\", 0)*100:+.2f}%')

# Runner state
rs = Path('data/runner_state.json')
if rs.exists():
    rstate = json.loads(rs.read_text())
    print(f'Entry prices: {rstate.get(\"entry_prices\", {})}')
    print(f'Agent memory: {rstate.get(\"agent_memory\", \"\")}')
"
```

### 2. Check Regime

```python
from analysis.regime_detector_crypto import CryptoRegimeDetector
from analysis.macro_regime import MacroRegimeDetector

crypto = CryptoRegimeDetector()
result = crypto.detect()
print(f"Crypto regime: {result['regime_label']} (vol={result.get('current_vol', 0):.4f})")

macro = MacroRegimeDetector()
mr = macro.detect()
print(f"Macro regime: {mr.value}, Exposure scale: {macro.exposure_scale:.1f}x")
```

### 3. Check Derivatives

```python
# Requires running instance — read from cached context
# Or manually:
from brokers.binance import BinanceBroker
broker = BinanceBroker()
broker.connect()

for ticker in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
    fr = broker._exchange.fetch_funding_rate(ticker)
    print(f"{ticker} funding: {fr['fundingRate']:.6f}")
```

### 4. Check Gate Status

```python
# Gate status shows Claude's self-scheduled next check time and wake conditions
# Available from runner.adaptive_gate.get_status()
```

### 5. Force Decision (Manual Trigger)

If the gate timer hasn't expired but you want Claude to decide now:

```bash
# Run testnet with immediate decision
python3 binance/main.py --testnet
# Wait for gate wake, or modify gate timer to 0
```

## Interpreting Results

**Regime:**
- `low_volatility` → trend-follow, full exposure
- `high_volatility` → reduce exposure, tighter stops

**Macro Regime:**
- `GOLDILOCKS` → max exposure (growth + low inflation)
- `REFLATION` → moderate (growth + rising inflation)
- `STAGFLATION` → min exposure (no growth + high inflation)
- `DEFLATION` → moderate (no growth + low inflation)

**Derivatives Signals:**
- Funding > 0.03% → market overheated long, fade
- Funding < -0.02% → shorts paying, potential squeeze
- OI spike + price drop → liquidation cascade risk
- CVD divergence → smart money disagrees with price

**Claude Decision Fields:**
- `action`: BUY/SELL/HOLD per ticker
- `position_pct`: 0.0 to 0.30 (max 30% per position)
- `confidence`: 0.0 to 1.0
- `next_check_seconds`: Claude's self-scheduled next check
- `wake_conditions`: conditions to wake Claude early
- `memory_update`: Claude's note for next decision

## Config

Configured in `binance/crypto_config.py`:
- `SWING_BLEND_CONFIG`: 1h bars, 20% max position, 6% trailing stop
- Multi-TF: 15m/1h/4h simultaneous WS subscription
- `EVENT_CONFIG.gate`: z-score threshold 2.0, window 50, max check 3600s
- Risk: stop_loss 4%, trail_activation 4%, trail_pct 6%
- Target frequency: 1-3 trades/day
