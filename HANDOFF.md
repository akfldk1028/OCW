# OCW Trading Engine — HANDOFF

> Last updated: 2026-02-25
> Status: Phase 1 complete. 0 live trades. Cash holding. Testnet validated.

---

## 1. Project Identity

| Item | Value |
|------|-------|
| Repo | https://github.com/akfldk1028/OCW |
| Fly.io app | `ocw-trader` (nrt = Tokyo) |
| Runtime | Python 3.11, shared-cpu-1x, 2048MB RAM |
| Cost | ~$6/month (Fly.io) + ~$5-10/month (Claude API) |
| Exchange | Binance (testnet), BTC/USDT + ETH/USDT + SOL/USDT |
| AI | Claude Sonnet 4.6 via `claude-agent-sdk` |

---

## 2. Repository Structure

```
OCW/
├── .gitignore                  # Root: secrets, IDE, data, models
├── HANDOFF.md                  # This file
├── docs/                       # Architecture docs (00-07)
│   └── MAC-MINI-SETUP.md      # Mac Mini deployment guide
├── openclaw/                   # Monorepo (flattened from submodule)
│   ├── src/                    # Core platform (TypeScript)
│   ├── apps/                   # iOS, macOS, Android clients
│   ├── extensions/
│   │   └── trading-engine/
│   │       └── python/         # *** TRADING ENGINE ***
│   │           ├── fly.toml    # Fly.io config
│   │           ├── agents/     # Claude agent prompts
│   │           ├── analysis/   # HMM regime, macro regime
│   │           ├── brokers/    # ccxt Binance wrapper
│   │           ├── core/       # Engine core modules
│   │           ├── binance/    # Entry point + config
│   │           │   ├── main.py         # CLI entry
│   │           │   ├── runner.py       # 5 async loops orchestrator
│   │           │   ├── crypto_config.py # All tuning parameters
│   │           │   ├── Dockerfile
│   │           │   ├── .env.example
│   │           │   └── tests/
│   │           ├── backtests/  # Claude backtester
│   │           └── config.py   # Global settings
│   └── vendor/                 # a2ui renderer
```

---

## 3. Architecture

### Claude = PRIMARY trader (rule-based = fallback only)

```
WS tick -> AdaptiveGate (4-layer) -> ~90% skip
                                  -> ~10% pass -> build MarketSnapshot
                                              -> ClaudeAgent.decide()
                                              -> execute order
                                              -> RL feedback (Thompson Sampling)
```

### 4-Layer Gate
1. **candle_close** — only trigger at kline close
2. **timer** — respect Claude's `next_check_seconds`
3. **z-score** — price/volume Z > threshold
4. **wake_conditions** — Claude-defined conditions

### 5 Async Loops (runner.py)
1. `market_listener.run()` — Binance WS kline, `market.tick` events
2. `derivatives_monitor.run()` — Funding/OI/CVD/L-S ratio polling
3. `position_tracker.run()` — 30s safety (SL/trailing) + 180s agent re-eval
4. `_heartbeat_loop()` — 30s file write (Docker healthcheck)
5. `_watchdog_loop()` — 5min no-WS fallback decision

### Regime-Aware Thompson Sampling
- Per-regime Beta(alpha, beta) for each signal
- Sliding window = 50 trades
- Signals: funding(0.35) + CVD(0.30) + L/S(0.25) + OI(0.10)
- tanh nonlinearity for derivatives composite

---

## 4. Current State

- **0 live trades executed** — engine validated on testnet only
- **Cash holding** — Thompson Sampling prior = Beta(2,2) (uninformative)
- **All subsystems operational on Fly.io**: WS, OHLCV, derivatives, heartbeat
- **Claude Agent blocked**: needs `ANTHROPIC_API_KEY` (see TODO #1)

---

## 5. Environment Variables

Copy `binance/.env.example` to `binance/.env`:

```env
# Required
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET_KEY=your_testnet_secret
BINANCE_PAPER=true
LIVE_TRADING=false

# Claude Agent (pick one)
ANTHROPIC_API_KEY=sk-ant-api03-...   # recommended for server

# Optional
FRED_API_KEY=...                      # macro regime (defaults GOLDILOCKS)
LOG_LEVEL=INFO
```

---

## 6. Deployment

### Fly.io (current)
```bash
cd openclaw/extensions/trading-engine/python
fly deploy
fly secrets set BINANCE_API_KEY=... BINANCE_SECRET_KEY=... ANTHROPIC_API_KEY=...
fly logs --app ocw-trader
```

### Local / Mac Mini
```bash
cd openclaw/extensions/trading-engine/python
python3 -m venv .venv && source .venv/bin/activate
pip install -r binance/requirements.txt
cp binance/.env.example binance/.env  # edit with your keys
python3 binance/main.py --testnet
```

See `docs/MAC-MINI-SETUP.md` for full 24/7 setup with launchd.

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Thompson Sampling > Deep RL | Sample-efficient, no GPU, interpretable |
| Regime-aware TS | Different market regimes need different signal weights |
| tanh nonlinearity | Saturating activation prevents outlier dominance |
| HMM BIC auto-state | 2/3/4 states, auto-selects best fit |
| REST -> WS migration | Binance IP ban from REST polling |
| AdaptiveGate ~90% skip | ~$0.03/Claude call, save to ~$5-10/month |
| claude-agent-sdk (CLI wrapper) | Smarter than raw API for complex reasoning |

---

## 8. Known Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| No position restore on restart | Positions lost | `broker.get_positions_detail()` restore |
| TS signal undifferentiated | All signals equal confidence | Per-signal confidence in Claude schema |
| `funding_rate` not ticker-scoped | Last ticker only | Add ticker prefix |
| Safety thresholds hidden from Claude | Suboptimal decisions | Add to `MarketSnapshot.to_prompt()` |
| Legacy testnet sunset | Migration needed | `BINANCE_USE_DEMO=true` |

---

## 9. TODO (priority order)

1. **ANTHROPIC_API_KEY**: get from console.anthropic.com, set as Fly.io secret
2. Walk-forward backtest (in-sample / out-of-sample split)
3. Dashboard: FastAPI endpoints + Next.js + Vercel
4. 3-6 month testnet paper trading (24/7 cloud)
5. Phase 2: Graphiti memory + SemanticCache + trade-reflect

---

## 10. Critical File Map

| File | Purpose |
|------|---------|
| `binance/runner.py` | Main orchestrator, 5 async loops |
| `binance/crypto_config.py` | All tunable parameters |
| `core/claude_agent.py` | Claude SDK integration |
| `core/claude_auth.py` | Auth priority: API_KEY > OAuth > credentials |
| `core/zscore_gate.py` | 4-layer adaptive gate |
| `core/online_learner.py` | Regime-aware Thompson Sampling |
| `core/position_tracker.py` | Safety SL/trailing + agent re-eval |
| `core/market_listener.py` | Binance WS kline stream |
| `core/derivatives_monitor.py` | Funding/OI/CVD/L-S polling |
| `analysis/regime_detector_crypto.py` | HMM BIC regime detection |
| `brokers/binance.py` | ccxt order execution |
| `fly.toml` | Fly.io deployment config |
| `binance/Dockerfile` | Container build |
