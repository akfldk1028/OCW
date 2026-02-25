# Trading Engine v3 — Real-time Multi-Agent Architecture

Real-time multi-agent trading engine for OpenClaw. Combines WebSocket event streaming, cross-sectional XGBoost factor ranking (v8), and multi-agent weighted voting for autonomous trading decisions.

## Architecture

```
                     +-----------------------------------+
                     |      OpenClaw Gateway (WS)        |
                     |  broadcast("trading.*", payload)  |
                     |  -> UI / Telegram / Discord       |
                     +-----------------+-----------------+
                                       | events
+--------------------------------------+--------------------------------------+
|              index.ts (Plugin)       |                                      |
|  +-----------+ +-----------+ +------+------+ +---------------+             |
|  | 15 Tools  | | Gateway   | | Event      | |  Cron Jobs    |             |
|  | (v2 + v3) | | Methods   | | Bridge     | |  15m/daily/wk |             |
|  +-----+-----+ +-----+-----+ +------+-----+ +-------+-------+             |
|        +---------------+-----------+-----------------+                     |
|                         | HTTP + WS                                        |
+-------------------------+--------------------------------------------------+
                          |
+--------------------------+--------------------------------------------------+
|           Python FastAPI Server (server.py)                                  |
|                                                                              |
|  +----------------- WebSocket Layer -------------------+                    |
|  |  /ws/stream  ->  real-time market events + signals  |                    |
|  |  EventBus   ->  internal pub/sub coordination       |                    |
|  +------------------------+----------------------------+                    |
|                           |                                                 |
|  +---------- Multi-Agent Pipeline (v3 core) ----------+                    |
|  |                                                     |                    |
|  |  +-------------+  +-------------+  +-----------+   |                    |
|  |  | MarketAgent |  | QuantAgent  |  | FinBERT   |   |                    |
|  |  | (regime +   |  | (XGB v8     |  | Sentiment |   |                    |
|  |  |  sector)    |  |  z-score)   |  |           |   |                    |
|  |  +------+------+  +------+------+  +-----+-----+   |                    |
|  |         +----------------+----------------+         |                    |
|  |                    +-----v------+                   |                    |
|  |                    | Synthesizer| <- weighted vote   |                    |
|  |                    | (decision) |                   |                    |
|  |                    +-----+------+                   |                    |
|  |                    +-----v------+                   |                    |
|  |                    | RiskGuard  | <- position/risk   |                    |
|  |                    +-----+------+                   |                    |
|  +---------------------------+--------------------------+                    |
|                        +-----v------+                                       |
|                        |  Executor  | <- Alpaca broker                      |
|                        +------------+                                       |
+------------------------------------------------------------------------------+
```

## Multi-Agent Pipeline

### MarketAgent (weight: 0.20 + 0.10)
- **HMM Regime Detection**: 2-state Hidden Markov Model (low_vol / high_vol)
- **14 Sector ETF Momentum**: Moskowitz-Grinblatt relative momentum vs SPY
- **Regime Bias**: defensive sectors boosted in high-vol, growth favoured in low-vol

### QuantAgent v8 (weight: 0.30) — Highest Weight
- **13 Factor Model**: momentum_5d/21d/63d, RSI, MFI, OBV slope, GK vol, Amihud, ADX, volume ratio, BB width, 52w high pct, sector momentum
- **Z-Score Normalization**: raw features -> cross-sectional z-score (winsorized at +/-3 sigma) — preserves signal magnitude unlike percentile ranking
- **Top-Quartile Labeling**: predicts P(top 25% absolute return) over 21-day horizon
- **XGBoost Classifier**: depth=4, n=300, lr=0.03, scale_pos_weight=auto (~3.0 for 25% positive rate)
- **18-Month Rolling Window**: walk-forward retraining with 5-day purge gap
- **Validated**: backtest v8 over 3 years, +38~123% excess return, Sharpe 1.28~1.56

### RL Ensemble (weight: 0.15)
- **PPO + A2C + SAC** with Sharpe-weighted voting
- Trained on 5 tickers, obs_size=96 (OHLCV + technical indicators + sentiment)

### Synthesizer (final decision + EXIT management)
Weighted voting across all agent signals:

```
final_score = 0.20 * sector_signal      # MarketAgent sector strength
            + 0.30 * quant_signal        # QuantAgent v8 P(top-quartile) [highest]
            + 0.15 * rl_signal           # RL ensemble (PPO+A2C+SAC)
            + 0.10 * sentiment_signal    # FinBERT news sentiment
            + 0.15 * momentum_signal     # z-scored price momentum
            + 0.10 * regime_bias         # Regime-based sector adjustment

BUY if final_score > 0.35, SELL if < -0.2, else HOLD
```

**Position-aware EXIT management** (for held positions):
- Receives `current_positions` from Alpaca broker each cron cycle
- Orphan detection: held positions not in scan candidates → evaluated for exit
- EXIT rules (SWING_EXIT_CONFIG): stop_loss -3%, take_profit +5%, consecutive scan miss 3x
- SELL-before-BUY ordering: frees buying power before submitting new purchases

## Real-time Data Flow

```
[cron_decide 15m] (position-aware)
    |
    v
Alpaca broker.get_positions_detail()
    | current_positions: {NVDA: {qty, entry_price, unrealized_plpc, ...}}
    v
MarketAgent.analyze()
    | regime: low_vol, top_sectors: [Tech, Semi, Energy]
    v
QuantAgent.rank_stocks(candidates, sector_scores)
    | z-score features -> XGBoost P(top-quartile)
    | MRK 0.76, TMUS 0.63, CAT 0.62...
    v
FinBERT sentiment (news headlines)
    | MRK: +0.72, TMUS: +0.55...
    v
Synthesizer.synthesize(all_signals, current_positions)
    | EXIT check: orphan positions -> stop_loss/take_profit/scan_miss
    | MRK: BUY 0.68 confidence, 5.1% size
    | DIS: SELL (orphan, -4.2% loss > -3% SL)
    | Sort: SELLs first, then BUYs by confidence
    v
RiskManager.filter()
    | position limits OK, exposure < 80% max
    v
Executor (Alpaca) -> dry_run or live
    |
    v
EventBus -> WS broadcast -> OpenClaw gateway -> UI/Telegram

[cron_risk_check 5m] (independent)
    |
    v
Alpaca broker.get_positions_detail()
    | For each position: check TP(+4%) / SL(-2.5%) / trailing stop
    | Auto-SELL on trigger
```

## WebSocket Events

Connect to `ws://localhost:8787/ws/stream` for real-time events:

```json
{"type": "regime.update",    "data": {"regime": "low_vol", "confidence": 0.92}}
{"type": "sector.hot",       "data": {"top_sectors": ["Tech", "Semi"]}}
{"type": "signal.buy",       "data": {"ticker": "MRK", "confidence": 0.68}}
{"type": "signal.sell",      "data": {"ticker": "DIS", "confidence": -0.35}}
{"type": "order.filled",     "data": {"ticker": "MRK", "side": "buy", "qty": 17}}
{"type": "risk.alert",       "data": {"type": "stop_loss", "ticker": "BA"}}
{"type": "portfolio.update", "data": {"equity": 105000, "positions": 8}}
```

## Quick Start

```bash
# 1. Install Python dependencies
cd python
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 2. Set environment variables (optional, for live trading)
set ALPACA_API_KEY=your_key
set ALPACA_SECRET_KEY=your_secret
set ALPACA_PAPER=true

# 3. Start Python server
python server.py
# Server runs on http://127.0.0.1:8787

# 4. Multi-agent decision (v3 pipeline)
curl -X POST localhost:8787/decide/v3

# 5. Quant ranking only
curl -X POST localhost:8787/quant/rank

# 6. Retrain XGBoost (18-month window)
curl -X POST localhost:8787/quant/retrain

# 7. WebSocket stream
wscat -c ws://localhost:8787/ws/stream
```

### Crypto (Binance Testnet)
```bash
# 1. Set Binance testnet credentials
set BINANCE_API_KEY=your_testnet_key
set BINANCE_SECRET_KEY=your_testnet_secret
set BINANCE_PAPER=true
set ENABLE_CRON=true

# 2. Start server (crypto cron runs 24/7 every 30min)
python server.py

# 3. Verify broker connection
curl localhost:8787/broker/status

# 4. Monitor crypto positions
curl localhost:8787/tracker/status

# 5. Check online learner adaptation
curl localhost:8787/learner/status
curl localhost:8787/learner/weights
```

### Via OpenClaw (recommended)
```bash
# OpenClaw auto-starts the trading engine
openclaw start

# Use skills in chat
/trade-scan       # Sector scan + top picks
/trade-execute    # Execute trade decisions
/trade-status     # Portfolio & model status
/trade-backtest   # Run historical backtest
/trade-train      # Train RL models
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide/v3` | POST | **v3 multi-agent pipeline** (MarketAgent + QuantAgent v8 + Synthesizer) |
| `/decide` | POST | v2 auto_trader (legacy fallback) |
| `/quant/rank` | POST | XGBoost v8 cross-sectional z-score ranking |
| `/quant/retrain` | POST | Retrain XGBoost (18-month rolling window) |
| `/ws/stream` | WS | Real-time event stream |
| `/ws/status` | GET | WebSocket connection status |
| `/scan` | POST | Sector momentum scanner |
| `/regime` | GET | HMM regime detection |
| `/predict` | POST | RL ensemble predictions (PPO+A2C+SAC) |
| `/train` | POST | Train RL ensemble models |
| `/backtest` | POST | Historical performance backtest |
| `/scan/backtest` | POST | Validate scanner strategy |
| `/execute` | POST | Execute via Alpaca (dry_run default) |
| `/broker/status` | GET | All broker connection status |
| `/tracker/status` | GET | Position tracker: active positions + monitoring state |
| `/learner/status` | GET | Online learner: Thompson Sampling agent weights |
| `/learner/weights` | GET | Current adapted + sampled agent weights |
| `/status` | GET | Engine status & metrics |
| `/health` | GET | Server health check |

## Backtest Results

### v8 (Current — Z-Score + Top-Quartile)

3-year walk-forward backtest (2022-01 ~ 2025-02):

| Metric | Value |
|--------|-------|
| Portfolio Return | +200% |
| SPY B&H Return | +78% |
| **Excess Return** | **+123%** |
| Sharpe Ratio | 1.56 |
| Win Rate | 60.6% |
| Payoff Ratio | 7.93x |
| Total Fees | ~643 |
| Positions | 4-10 dynamic |

Key v8 improvements over v6:
- Z-score normalization preserves signal magnitude (percentile ranking made all features uniform)
- Top-quartile labeling gives cleaner learning signal than SPY-relative
- 13 features (vs 8 in v6): added RSI, BB width, 52-week high %, 3-month momentum
- 18-month training window (vs 12)
- scale_pos_weight handles class imbalance (25% positive rate)

### Historical Comparison

| Version | Strategy | Excess Return | Sharpe |
|---------|----------|--------------|--------|
| v6 | Percentile rank + SPY-relative | +1.71% | 0.95 |
| v7 | Concentrated v6 (max 6 pos) | -29.10% | 1.15 |
| **v8** | **Z-score + top-quartile** | **+38~123%** | **1.28~1.56** |

### Crypto Regime Blend (Validated)

3-year backtest on BTC/ETH/SOL (2023-01 ~ 2026-02, $5K initial):

| Metric | Value |
|--------|-------|
| Total Return | +382% ($5K -> $24,104) |
| BTC B&H Return | +373% |
| **Alpha vs BTC** | **+9.2%** |
| Sharpe Ratio | 1.65 |
| Max Drawdown | 35.2% (vs BTC 37.0%) |

Strategy:
- **Regime detection**: efficiency ratio + trend strength -> "trending" / "ranging"
- **Trending**: RSI momentum (buy RSI > 50 + positive 14d momentum > 2%)
- **Ranging**: Bollinger Band mean reversion (buy near lower band + RSI < 40)
- **Risk**: 15% trailing stop, 8% drawdown trigger, 70% max exposure, 35% position sizing

Pipeline nodes: `RegimeBlendDetect -> RegimeBlendSignal -> RegimeBlendExit -> RegimeBlendEntry`

### Position Tracker + Online Learner

- **PositionTracker**: Continuous monitoring (price polling 30s, agent re-evaluation 3min)
  - 5-signal weighted evaluation: regime change, momentum reversal, correlation breakdown, PnL, time decay
  - Auto-exit on adverse signals without waiting for cron cycle

- **OnlineLearner**: Thompson Sampling weight adaptation (real-time RL)
  - Beta distribution posterior per agent signal
  - Adapts after 5+ closed trades
  - Persists across restarts (`models/online_learner.json`)

## File Structure

```
extensions/trading-engine/
├── index.ts                  # OpenClaw plugin (tool registration, cron, hooks)
├── src/python-bridge.ts      # Node<->Python HTTP + WS bridge
├── config/
│   ├── trading.default.json  # Watchlist, training params, server config
│   ├── risk.default.json     # TP/SL, Kelly sizing, portfolio limits, circuit breakers
│   └── mcp-servers.md        # External MCP server setup
├── python/
│   ├── server.py             # FastAPI server (15 endpoints + WS + cron)
│   ├── event_bus.py          # Async pub/sub for agent coordination
│   ├── ws_stream.py          # WebSocket broadcast manager
│   ├── agents/
│   │   ├── market_agent.py   # HMM regime + 14 sector ETF momentum
│   │   ├── quant_agent.py    # XGBoost v8 (13 factors, z-score, top-quartile)
│   │   └── synthesizer.py    # Multi-agent weighted voting (6 signals)
│   ├── auto_trader.py        # v2 autonomous decision engine (legacy)
│   ├── regime_detector.py    # HMM 2-state regime (low_vol/high_vol)
│   ├── sector_scanner.py     # 14 sector ETF scan + stock selection
│   ├── stock_ranker.py       # v1 XGBoost ranker (legacy)
│   ├── sentiment_finbert.py  # ProsusAI/finbert transformer sentiment
│   ├── ensemble_agent.py     # PPO+A2C+SAC RL ensemble
│   ├── broker_alpaca.py      # Alpaca paper/live trading broker
│   ├── broker_binance.py     # Binance Spot/Futures broker (ccxt, testnet support)
│   ├── broker_kis.py         # 한국투자증권 broker (mojito2)
│   ├── risk_manager.py       # Position sizing, limits, TP/SL
│   ├── data_processor.py     # OHLCV download + technical indicators
│   ├── config.py             # SECTOR_MAP, REGIME_BLEND_CONFIG, SWING_EXIT_CONFIG, etc.
│   ├── pipeline.py           # n8n-style modular pipeline (equity + crypto nodes)
│   ├── position_tracker.py   # Continuous position monitoring (30s price, 3min agent)
│   ├── agent_evaluator.py    # 5-signal position evaluation
│   ├── online_learner.py     # Thompson Sampling weight adaptation
│   ├── backtest_v2.py        # Evidence-based backtest (Regime Blend + Equity Switch)
│   ├── backtest_pipeline.py  # Full v3 pipeline backtest (scan→rank→synthesize→execute)
│   ├── backtest_v8.py        # v8 walk-forward backtest + chart generation
│   ├── models/               # Saved models (XGBoost, PPO, A2C, SAC, online_learner.json)
│   └── charts/               # Backtest result charts (PNG)
├── skills/                   # OpenClaw chat commands (/trade-*)
├── hooks/                    # Event-driven automation (alerts, risk guardian)
├── cron/                     # Scheduled jobs (15m scan, daily rebalance, weekly retrain)
├── docs/
│   ├── architecture.md       # System architecture deep-dive
│   ├── runbook.md            # Operations manual
│   └── papers.md             # Research references
└── research/                 # Survey papers, design docs
```

## Configuration

### Trading Config (`config/trading.default.json`)
- `watchlist.tickers`: Stocks to trade (default: S&P 500 top 10)
- `training.*`: RL training hyperparameters
- `ensemble.*`: Model selection and weighting
- `server.port`: API server port (default: 8787)
- `broker.mode`: "paper" or "live"

### Risk Config (`config/risk.default.json`)
- `position_sizing.kelly_fraction`: 0.25 (quarter-Kelly)
- `portfolio_limits.max_positions`: 10
- `portfolio_limits.max_exposure_pct`: 0.80
- `daily_limits.max_daily_loss_pct`: -0.02
- `circuit_breakers.max_drawdown_pct`: -0.10

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `ALPACA_API_KEY` | For live/paper trading | Alpaca API key |
| `ALPACA_SECRET_KEY` | For live/paper trading | Alpaca secret key |
| `ALPACA_PAPER` | No (default: true) | Use paper trading |
| `BINANCE_API_KEY` | For crypto trading | Binance API key |
| `BINANCE_SECRET_KEY` | For crypto trading | Binance secret key |
| `BINANCE_PAPER` | No (default: true) | Use Binance testnet |
| `ENABLE_CRON` | No (default: false) | Enable scheduled jobs (equity@15m, crypto@30m, risk@5/10m) |
| `ENABLE_TRACKER` | No (default: true) | Enable position tracker (30s price, 3min agent) |
| `LIVE_TRADING` | No (default: false) | Enable live order execution |

## Key Algorithms

### Cross-Sectional Z-Score (QuantAgent v8)
```python
# 1. Compute 13 raw features for all 64 stocks
for tic in universe:
    features[tic] = calc_features(tic, date, sector_scores)

# 2. Z-score normalize cross-sectionally (preserves magnitude!)
for fname in FEATURE_NAMES:
    vals = [features[tic][fname] for tic in universe]
    mean, std = np.mean(vals), np.std(vals)
    for tic in universe:
        z = (features[tic][fname] - mean) / std
        features[tic][fname] = np.clip(z, -3.0, 3.0)  # winsorize

# 3. XGBoost predicts P(top-quartile absolute return)
p_win = model.predict_proba(z_scored_features)[:, 1]
```

### Why Z-Score beats Percentile Ranking
- Percentile: NVDA momentum=99th, AMD momentum=98th -> both ~1.0 (indistinguishable)
- Z-score: NVDA momentum=+2.3 sigma, AMD momentum=+1.1 sigma -> clear separation
- XGBoost needs magnitude information to learn meaningful splits

## References

- Xiao et al., "TradingAgents: Multi-Agents LLM Financial Trading Framework", NeurIPS 2025
- arXiv 2601.19504: Regime-Adaptive Trading (HMM Sharpe 1.05, +1-4% alpha)
- arXiv 2502.14897: FinBERT (+11% accuracy vs keyword-based sentiment)
- arXiv 2511.12120: LLM+RL Hybrid (Sharpe 1.10)
- Moskowitz & Grinblatt: Cross-Sectional Momentum (sector rotation)
- Meta-analysis of 167 papers: Implementation > Algorithm (31% vs 8%)
- arXiv 1911.05309: Thompson Sampling for online weight adaptation
- arXiv 2410.04217: Real-time agent evaluation in trading systems
- arXiv 2511.13239: Talyxion framework (Sharpe 5.72, drawdown risk management)
- 2025 Systematic Crypto Study: Momentum+MeanReversion blend (Sharpe 1.71)
