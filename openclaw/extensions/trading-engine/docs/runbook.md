# Trading Engine Runbook

Operational manual for starting, stopping, monitoring, and troubleshooting.

## Quick Start

### 1. Install Python Dependencies

```bash
cd openclaw/extensions/trading-engine/python
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Start the Trading Engine

```bash
# Manual start
cd openclaw/extensions/trading-engine/python
python server.py
# Server runs on http://127.0.0.1:8787
```

Or via OpenClaw (auto-starts with gateway):
```bash
openclaw start
```

### 3. Verify Health

```bash
curl http://localhost:8787/health
# Expected: {"status":"ok","models_loaded":false,"server_uptime":...}
```

### 4. Run v3 Multi-Agent Decision

```bash
curl -X POST localhost:8787/decide/v3
```

This runs the full pipeline: MarketAgent -> QuantAgent v8 -> RL Ensemble -> FinBERT -> Synthesizer -> RiskManager

### 5. Run XGBoost Retrain (18-month window)

```bash
curl -X POST localhost:8787/quant/retrain
# Takes 2-5 minutes. Downloads 18 months of OHLCV data for 64 stocks.
```

### 6. Run v8 Backtest

```bash
cd python
python backtest_v8.py
# Outputs: summary stats + chart saved to charts/backtest_v8_results.png
```

### 7. Run Full Pipeline Backtest (v3)

```bash
cd python
python backtest_pipeline.py
# Simulates: scan → rank → synthesize → execute with EXIT management
# Outputs: total return, alpha vs SPY, Sharpe, MDD, win rate, turnover
```

## Operations

### Starting

```bash
# Via OpenClaw (recommended)
openclaw start                     # Auto-starts Python server

# Manual Python server
cd extensions/trading-engine/python
python server.py                   # http://127.0.0.1:8787

# With live trading enabled
LIVE_TRADING=true python server.py

# With cron jobs enabled
ENABLE_CRON=true python server.py
```

### Stopping

```bash
# Via OpenClaw
openclaw stop                      # Graceful shutdown

# Manual (Windows)
taskkill /F /PID <pid>

# Manual (Linux/Mac)
kill $(lsof -t -i:8787)
```

### Monitoring

| Command | Purpose |
|---------|---------|
| `curl localhost:8787/health` | Server health check |
| `curl localhost:8787/status` | Detailed model metrics |
| `curl localhost:8787/regime` | Current market regime |
| `curl localhost:8787/ws/status` | WebSocket connections |
| `curl -X POST localhost:8787/quant/rank` | Run quant ranking only |
| `/trade-status` (OpenClaw chat) | Portfolio + model status |

### WebSocket Monitoring

```bash
# Install wscat
npm install -g wscat

# Connect to stream
wscat -c ws://localhost:8787/ws/stream

# Subscribe to specific events
> {"action": "subscribe", "topics": ["signal.buy", "signal.sell"]}
```

## API Endpoints Reference

### v3 Pipeline (Current)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide/v3` | POST | Full multi-agent pipeline |
| `/quant/rank` | POST | XGBoost v8 ranking only |
| `/quant/retrain` | POST | Retrain XGBoost (18-month window) |
| `/ws/stream` | WS | Real-time event stream |
| `/ws/status` | GET | WebSocket connection status |

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/regime` | GET | HMM regime detection |
| `/scan` | POST | Sector momentum scanner |
| `/predict` | POST | RL ensemble predictions |
| `/train` | POST | Train RL models |
| `/execute` | POST | Execute trades (Alpaca) |
| `/backtest` | POST | Run backtest |
| `/scan/backtest` | POST | Validate scanner |
| `/health` | GET | Health check |
| `/status` | GET | Detailed status |

### Legacy (v2)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | v2 auto_trader (6-signal) |

## Cron Jobs (Python APScheduler)

| Schedule | Job | Description |
|----------|-----|-------------|
| Every 15 min (9:00-16:00 ET) | `cron_decide` | Position-aware v3 pipeline: get positions → scan → rank → synthesize (w/ EXIT) → execute |
| Every 5 min (9:00-16:00 ET) | `cron_risk_check` | TP/SL/trailing stop check for all held positions (RISK_CONFIG thresholds) |
| Daily 09:00 ET | `cron_regime` | Regime detection update (HMM) |
| Weekly Sunday 02:00 ET | `cron_weekly_retrain` | XGBoost 18-month retrain + RL retrain with dynamic top-15 tickers |

Enable with `ENABLE_CRON=true` or in OpenClaw: `openclaw cron enable`.

**Note**: `cron_decide` and `cron_risk_check` are independent loops. Risk check uses tighter thresholds (RISK_CONFIG: -2.5% SL, +4% TP) than Synthesizer EXIT rules (SWING_EXIT_CONFIG: -3% SL, +5% TP).

## Configuration Files

| File | Purpose |
|------|---------|
| `config/trading.default.json` | Watchlist, training params, server port, broker mode |
| `config/risk.default.json` | TP/SL, Kelly sizing, portfolio limits, circuit breakers |
| `config/mcp-servers.md` | MCP server setup (Alpaca, Alpha Vantage, etc.) |

### Key Config Values

```json
// trading.default.json
"server.port": 8787
"broker.mode": "paper"          // "paper" or "live"
"ensemble.models": ["ppo", "a2c", "ddpg"]
"watchlist.tickers": ["AAPL", "MSFT", ...]

// risk.default.json
"position_sizing.kelly_fraction": 0.25   // quarter-Kelly
"portfolio_limits.max_positions": 10
"portfolio_limits.max_exposure_pct": 0.80
"daily_limits.max_daily_loss_pct": -0.02
"circuit_breakers.max_drawdown_pct": -0.10
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | (none) | Alpaca API key (required for trading) |
| `ALPACA_SECRET_KEY` | (none) | Alpaca secret key |
| `ALPACA_PAPER` | `true` | Use paper trading endpoint |
| `ENABLE_CRON` | `false` | Enable scheduled cron jobs |
| `LIVE_TRADING` | `false` | Allow live order execution |
| `TRADING_PORT` | `8787` | Server port |

### Python Config Constants (`config.py`)

| Config | Purpose |
|--------|---------|
| `SWING_EXIT_CONFIG` | EXIT rules for held positions: max_hold_days=10, stop_loss=-3%, take_profit=+5%, consecutive_miss_limit=3 |
| `RISK_CONFIG` | RiskManager thresholds: stop_loss=-2.5%, take_profit=+4%, trailing stop, ATR-based dynamic TP/SL |
| `SECTOR_MAP` | 14 sectors → ETF → stocks mapping (64 stocks total) |
| `SECTOR_SCAN_CONFIG` | Sector rotation scanner parameters |

## Troubleshooting

### Python server won't start

1. Check Python version: `python --version` (need 3.10+)
2. Check venv: `which python` (should point to venv)
3. Check port: `netstat -an | findstr 8787` (Windows) or `lsof -i:8787`
4. Install deps: `pip install -r requirements.txt`

### QuantAgent v8 training fails

1. Check internet: yfinance needs to download OHLCV data
2. Check disk space: needs ~200MB for 64 stocks x 600 days
3. Check log: `QuantAgent v8: trained on X samples` should appear
4. If features mismatch: delete `models/quant_xgboost_v8.pkl` and retrain

### RL models not loading

1. Check model files exist: `ls python/models/*.zip`
2. If empty, run training: `curl -X POST localhost:8787/train`
3. First training takes 5-15 minutes

### WebSocket not connecting

1. Check server is running: `curl localhost:8787/health`
2. Check WS status: `curl localhost:8787/ws/status`
3. Try reconnecting: `wscat -c ws://localhost:8787/ws/stream`

### Cron jobs not firing

1. Check `ENABLE_CRON=true` is set
2. Check timezone: jobs use America/New_York
3. Verify market hours (9:00-16:00 ET weekdays)
4. Manual trigger: `curl -X POST localhost:8787/decide/v3`

### Memory/Graphiti issues

1. Verify Neo4j: `docker ps | grep neo4j`
2. Restart: `docker restart neo4j`
3. Check: `curl http://localhost:7474`
4. Reconnect: `/mcp` in OpenClaw

### High memory usage

1. XGBoost training downloads lots of data — normal peak ~1GB
2. FinBERT loads transformer model (~400MB)
3. RL models: ~200MB total
4. Restart server if memory doesn't release after training

## Emergency Procedures

### Stop All Trading Immediately

```bash
# Option 1: Kill server
taskkill /F /PID <server_pid>    # Windows
kill $(lsof -t -i:8787)         # Linux/Mac

# Option 2: Disable cron
ENABLE_CRON=false               # Restart server

# Option 3: Circuit breaker
# Edit config/risk.default.json:
# "max_daily_loss_pct": 0       # Halts all trading
```

### Rollback XGBoost Model

```bash
# v8 model
ls python/models/quant_xgboost_v8.pkl

# Delete to force retrain
rm python/models/quant_xgboost_v8.pkl
curl -X POST localhost:8787/quant/retrain
```

### Data Recovery

Trade history in Graphiti (Neo4j). Data persists in Docker volume:
```bash
docker volume inspect neo4j_data
```
