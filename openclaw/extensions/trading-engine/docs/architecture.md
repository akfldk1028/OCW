# Trading Engine Architecture (v3 — Multi-Agent + v8 QuantAgent)

## System Overview

```
                     +-----------------------------------+
                     |        OpenClaw Gateway            |
                     |  Skills | Hooks | Cron | Channels  |
                     +----------------+------------------+
                                      | HTTP + WS
+-------------------------------------v-------------------------------------+
|                    index.ts (Plugin SDK)                                    |
|  15 tools registered | WS bridge | cron schedules | event broadcast        |
+-------------------------------------+-------------------------------------+
                                      | HTTP localhost:8787
+-------------------------------------v-------------------------------------+
|                 Python FastAPI Server (server.py)                           |
|                                                                            |
|  +--- WebSocket Layer (ws_stream.py + event_bus.py) ---+                  |
|  |  /ws/stream : client-facing broadcast                |                  |
|  |  EventBus   : internal pub/sub for agent coordination|                  |
|  +------------------------------+-----------------------+                  |
|                                 |                                          |
|  +------ Multi-Agent Pipeline --v--------------------------+              |
|  |                                                          |              |
|  |  [1] MarketAgent (market_agent.py)                       |              |
|  |      HMM regime + 14 sector ETF momentum                |              |
|  |      -> regime, top_sectors, sector_scores               |              |
|  |                                                          |              |
|  |  [2] QuantAgent v8 (quant_agent.py)                      |              |
|  |      13 factors -> z-score -> XGBoost P(top-quartile)    |              |
|  |      -> ranked tickers with p_win, z-scored features     |              |
|  |                                                          |              |
|  |  [3] RL Ensemble (ensemble_agent.py)                     |              |
|  |      PPO + A2C + SAC, Sharpe-weighted voting             |              |
|  |      -> action signals per ticker                        |              |
|  |                                                          |              |
|  |  [4] FinBERT Sentiment (sentiment_finbert.py)            |              |
|  |      ProsusAI/finbert transformer                        |              |
|  |      -> sentiment scores per ticker                      |              |
|  |                                                          |              |
|  |  [5] Synthesizer (synthesizer.py)                        |              |
|  |      6-signal weighted voting -> buy/sell/hold           |              |
|  |      market 0.20 | quant 0.30 | rl 0.15                  |              |
|  |      sentiment 0.10 | momentum 0.15 | regime 0.10        |              |
|  |      + Position-aware EXIT management (orphan detection) |              |
|  |      + SELL-before-BUY ordering for buying power         |              |
|  |                                                          |              |
|  |  [6] RiskManager (risk_manager.py)                       |              |
|  |      Position sizing (quarter-Kelly), exposure limits    |              |
|  |      TP/SL, trailing stop, circuit breakers              |              |
|  +----------------------------+-----------------------------+              |
|                               |                                            |
|  +----------------------------v----+                                       |
|  |  Broker (broker_alpaca.py)      |                                       |
|  |  Alpaca paper/live execution    |                                       |
|  +---------------------------------+                                       |
+----------------------------------------------------------------------------+
         |                    |                    |
    +----v----+         +----v----+          +----v----+
    | yfinance|         | Models  |          | Neo4j   |
    | (OHLCV) |         | (disk)  |          | Graphiti|
    +---------+         +---------+          +---------+
```

## Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| **Plugin** | `index.ts` | OpenClaw plugin lifecycle, 15 tools, cron, hooks, WS bridge |
| **Bridge** | `src/python-bridge.ts` | Node->Python HTTP calls, process management, WS client |
| **Server** | `server.py` | FastAPI (15 REST + 1 WS endpoint), cron scheduler, startup |
| **EventBus** | `event_bus.py` | Async pub/sub: agents publish events, WS layer broadcasts |
| **WS Stream** | `ws_stream.py` | WebSocket connection manager, client broadcast |
| **MarketAgent** | `agents/market_agent.py` | HMM regime + sector ETF scan |
| **QuantAgent** | `agents/quant_agent.py` | **Core signal**: 13-factor z-score XGBoost (v8) |
| **Synthesizer** | `agents/synthesizer.py` | Weighted voting + EXIT management for held positions |
| **RL Ensemble** | `ensemble_agent.py` | PPO+A2C+SAC with Sharpe-weighted averaging |
| **FinBERT** | `sentiment_finbert.py` | ProsusAI/finbert news sentiment |
| **Auto Trader** | `auto_trader.py` | v2 legacy decision engine (fallback via `/decide`) |
| **Regime** | `regime_detector.py` | HMM 2-state market regime detection |
| **Sector** | `sector_scanner.py` | 14 sector ETF momentum scan + stock selection |
| **Risk** | `risk_manager.py` | Position sizing, limits, TP/SL, circuit breakers |
| **Broker** | `broker_alpaca.py` | Alpaca paper/live order execution |
| **Data** | `data_processor.py` | OHLCV + 19 technical indicators |
| **Config** | `config.py` | SECTOR_MAP, 64-stock universe, scan config, SWING_EXIT_CONFIG |
| **Pipeline BT** | `backtest_pipeline.py` | Full v3 pipeline backtest (scan→rank→synthesize→execute) |

## Data Flow — v3 Multi-Agent Decision (`/decide/v3`)

```
1. MarketAgent.analyze()
   ├── regime_detector.detect() -> {regime: "low_vol", confidence: 0.92}
   ├── sector_scanner.scan_sectors() -> 14 sector ETFs ranked
   └── Output: MarketView {regime, top_sectors, sector_scores}

2. QuantAgent.rank_stocks(candidates, sector_scores)
   ├── Download 600 days OHLCV for candidates + ETFs
   ├── calc_features() -> 13 raw technical factors per stock
   ├── compute_zscore_features() -> cross-sectional z-scores (winsorized +-3)
   ├── Train/load XGBoost (18-month rolling, scale_pos_weight)
   ├── predict_proba() -> P(top-quartile) per stock
   └── Output: [{ticker, p_outperform, rank_features, confidence}]

3. RL Ensemble (if models loaded)
   ├── ensemble_agent.predict(tickers)
   ├── PPO, A2C, SAC each produce actions
   └── Sharpe-weighted average -> rl_signal per ticker

4. FinBERT Sentiment
   ├── sentiment_finbert.score(ticker) for each candidate
   └── Output: sentiment scores [-1, +1]

5. Synthesizer.synthesize(market_view, quant_rankings, rl, sentiment, current_positions)
   ├── For each candidate with p_outperform > 0.40:
   │   ├── sector_signal = sector composite score (from MarketAgent)
   │   ├── quant_signal = p_outperform (from QuantAgent v8)
   │   ├── rl_signal = ensemble action (from RL)
   │   ├── sentiment = finbert score
   │   ├── momentum = z-scored momentum_21d / 3.0
   │   ├── regime_bias = +0.5 (low_vol) or -0.3 (high_vol)
   │   └── weighted_sum -> final_score
   ├── BUY if final_score > 0.35
   ├── SELL if final_score < -0.20
   ├── EXIT management for held positions:
   │   ├── Orphan detection (held but not in scan candidates)
   │   ├── Stop loss (-3%), Take profit (+5%)
   │   ├── Consecutive scan miss (3x) → SELL
   │   └── SELL-before-BUY ordering (free buying power first)
   └── Output: [{ticker, action, confidence, size, reasons}]

6. RiskManager.filter()
   ├── Check portfolio exposure < 80%
   ├── Check position count < max_positions
   ├── Apply quarter-Kelly sizing
   └── Output: filtered decisions

7. Executor (if LIVE_TRADING=true)
   ├── broker_alpaca.execute(decisions)
   └── EventBus.publish("order.filled", ...)

8. RiskManager cron (every 5 min, independent)
   ├── broker_alpaca.get_positions_detail()
   ├── For each position: check TP/SL/trailing stop (RISK_CONFIG thresholds)
   └── Auto-SELL on trigger via broker_alpaca.execute_decisions()
```

## QuantAgent v8 — Core Algorithm

The most important signal (weight 0.30, validated by backtest).

### 13 Factors
| Factor | Description | Category |
|--------|-------------|----------|
| momentum_5d | 5-day price change | Momentum |
| momentum_21d | 21-day price change | Momentum |
| momentum_63d | 63-day (3-month) price change | Momentum |
| rsi_14 | Relative Strength Index | Mean-reversion |
| mfi | Money Flow Index | Volume |
| obv_slope | On-Balance Volume 20-day slope | Volume |
| gk_vol | Garman-Klass OHLC volatility | Volatility |
| amihud | Amihud illiquidity ratio | Liquidity |
| adx | Average Directional Index | Trend |
| volume_ratio | Volume vs 20d SMA | Volume |
| bb_width | Bollinger Band width | Volatility |
| high_52w_pct | Distance from 52-week high | Strength |
| sector_momentum | Sector ETF weighted momentum | Sector |

### Z-Score vs Percentile (why v8 works)
```
Percentile ranking (v6/v7):
  NVDA momentum = 99th percentile -> 0.98
  AMD  momentum = 95th percentile -> 0.94
  Difference: 0.04 (XGBoost can't distinguish)

Z-score normalization (v8):
  NVDA momentum = +2.3 sigma
  AMD  momentum = +1.1 sigma
  Difference: 1.2 (XGBoost learns meaningful splits)
```

### XGBoost Training
- **Labels**: Top 25% absolute 21-day return = 1, rest = 0
- **Window**: 18-month rolling (walk-forward)
- **Purge**: 5-day gap between train/predict (no leakage)
- **Params**: max_depth=4, n_estimators=300, learning_rate=0.03
- **Imbalance**: scale_pos_weight = n_negative / n_positive (~3.0)
- **Model file**: `models/quant_xgboost_v8.pkl`

## Cron Schedules

| Schedule | Job | Description |
|----------|-----|-------------|
| Every 15 min (market hours) | `cron_decide` | Position-aware v3 pipeline + execute decisions |
| Every 5 min (market hours) | `cron_risk_check` | TP/SL/trailing stop check for all held positions |
| Daily 09:00 ET | `cron_regime` | Regime detection update |
| Weekly Sunday 02:00 ET | `cron_weekly_retrain` | XGBoost 18-month retrain + RL dynamic ticker retrain |

## Model Files

| File | Size | Description |
|------|------|-------------|
| `quant_xgboost_v8.pkl` | ~2MB | Current XGBoost v8 model (13 features, z-score) |
| `quant_xgboost.pkl` | ~1MB | Legacy v6 model (8 features, percentile) |
| `ppo_model.zip` | ~15MB | PPO RL model |
| `a2c_model.zip` | ~10MB | A2C RL model |
| `sac_model.zip` | ~15MB | SAC RL model |
| `stock_ranker.pkl` | ~1MB | v1 stock ranker (legacy) |
| `ensemble_meta.pkl` | ~1KB | Ensemble metadata (Sharpe weights) |

## Key Design Decisions

1. **Python <-> TypeScript via HTTP**: Clean separation. Python handles ML compute, TypeScript handles OpenClaw SDK integration.

2. **Z-score over percentile ranking**: Preserves how far a stock deviates from peers. Percentile ranking normalizes everything to uniform [0,1], destroying the signal that XGBoost needs for meaningful splits.

3. **Top-quartile labeling over SPY-relative**: "Is this stock in the top 25% of returns?" is a more stable signal than "Does this stock beat SPY?" across different market regimes.

4. **Multi-agent weighted voting**: No single signal is reliable enough. Combining 6 signals (quant, market, RL, sentiment, momentum, regime) with validated weights reduces single-point-of-failure risk.

5. **Quarter-Kelly sizing**: Full Kelly is too aggressive (~25% of portfolio per position). Quarter-Kelly balances growth with drawdown protection.

6. **Paper-first**: All defaults use Alpaca paper trading. Live trading requires explicit `LIVE_TRADING=true`.

7. **Walk-forward validation**: Models are never evaluated on training data. 18-month train, 21-day predict, 5-day purge gap.

8. **Position-aware cron**: `cron_decide` fetches current Alpaca positions before each decision cycle. Synthesizer evaluates orphan positions (held but not in scan candidates) for exit.

9. **Layered risk defense**: Two independent loops — Synthesizer EXIT rules (SWING_EXIT_CONFIG: -3% SL, +5% TP) at 15-min and RiskManager cron (RISK_CONFIG: -2.5% SL, +4% TP) at 5-min. Tighter thresholds fire first.

10. **Dynamic RL retraining**: Weekly retrain uses top 15 tickers from recent scan results instead of fixed TRAIN_CONFIG tickers. Ensures RL signals are relevant to actual trading candidates.

11. **SELL-before-BUY ordering**: Synthesizer sorts decisions with SELLs first, then BUYs by confidence. This frees buying power before submitting new purchases.
