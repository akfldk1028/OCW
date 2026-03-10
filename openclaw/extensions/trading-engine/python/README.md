# OpenClaw Trading Engine

## What This Is

Event-driven algorithmic trading engine for **crypto** (Binance Futures/Spot) and **US stocks** (Alpaca).
**Claude (Sonnet 4.6) is the PRIMARY decision maker — a TRUE AGENT, not just an LLM.**
Claude has **7 MCP tools** to independently analyze markets (technical analysis, volume, derivatives, positions, trade history, TS weights, OHLCV data) and makes ALL trading decisions: **BUY, SELL, HOLD, ADD** (DCA/averaging down), position sizing, and exit timing.

**Philosophy**: "OpenClaw이 직접 판단하고, 엔진은 서포트한다. 위기는 기회다."
Rule-based pipeline exists for backtesting only. In production, Claude IS the trader.
Claude uses tools to gather data, analyzes it independently, and decides — including **averaging down** when a position dips but thesis remains intact.
Inspired by IQC 2025 winner (김민겸, UNIST) — stability via macro regime awareness beats pure return chasing.

---

## How It Works (Data Flow)

```
 1. MARKET DATA IN                    2. ADAPTIVE GATE                3. CLAUDE AGENT DECIDES
 ─────────────────                    ──────────────                  ──────────────────────
 Binance WebSocket ──┐                AdaptiveGate (4-layer)          ClaudeAgent.decide()
   every tick        │   EventBus       G1: candle close? → pass      Claude Sonnet 4.6 via SDK
   kline updates     ├──(pub/sub)──→    G2: Claude timer expired?     ┌─ 7 MCP Tools ──────┐
                     │                  G3: z-score outlier?   ──→    │ technical_analysis  │
 DerivativesMonitor ─┘                  G4: wake condition met?       │ volume_analysis     │
   Funding rate                         cooldown: min 60s             │ get_derivatives     │
   Open Interest                        ~90% ticks SKIPPED            │ check_position      │
   Taker Delta (CVD)                                                  │ search_trades       │
   Long/Short Ratio                                                   │ get_ts_weights      │
                                                                      │ get_ohlcv           │
                                                                      └─────────────────────┘
                                                                      BUY / SELL / HOLD / ADD
                                                                      position sizing
                                                                      confidence + reasoning

 4. SNAPSHOT BUILD                    5. EXECUTION + MONITORING       6. LEARNING
 ──────────────────                   ────────────────────────        ─────────────
 MarketSnapshot                       Broker (Binance)                HierarchicalOnlineLearner
   OHLCV + returns                      Spot or Futures ←── exec      Hierarchical TS v3
   Derivatives context                  verify success before track   28 signals × 6 groups
   Regime (crypto+macro)              PositionTracker                 per-regime Beta(α,β)
   TS posteriors (RL)                   30s: safety (SL/trailing)     group_discount=0.98
   Recent trades (learning)             3min: agent re-eval           signal_discount=0.95
   Open positions + portfolio         Watchdog (WS disconnect)        auto-save to JSON
```

**Agent-Autonomous Architecture:** Claude is a TRUE AGENT with tools, not just an LLM.
Every WebSocket tick passes through the AdaptiveGate. ~90% are skipped (no Claude call).
When Claude wakes, it **independently calls 5-9 tools** to analyze the market before deciding.
Claude wakes only when:
- 4h candle closes (always)
- Claude's own timer expires (`next_check_seconds`)
- Z-score outlier detected (|z| >= 2.5 on price/volume)
- Claude's wake conditions met (e.g. "wake me if BTC < $80K")
- WS disconnect watchdog fires (5min no ticks → force check)

---

## Key Design Decisions (WHY, not just WHAT)

### 1. Hierarchical Thompson Sampling v3, not Deep RL
167-paper meta-analysis (arXiv:2512.10913) found algorithm choice explains only 8% of trading performance.
Implementation quality explains 31%. Deep RL (PPO/SAC) needs 950+ episodes (~2 years of data) to converge.
Thompson Sampling works with 10-20 trades. We have ~15-30 trades/month. Decision: keep it simple.

**Hierarchical structure** (v3): 28 signals organized into 6 groups:
```
Group           Signals                                        group_discount
─────────────   ─────────────────────────────────────────────  ──────────────
momentum (5)    rsi, macd, ema_cross, stoch_rsi, adx          0.98
market (5)      regime, crypto_regime, fear_greed, trend, vix  0.98
quant (5)       bb_squeeze, volume_profile, atr, obv, vwap    0.98
sentiment (3)   news, social, whale_alert                      0.98
funding_rate(5) funding, oi_change, liquidations, basis, cvd   0.98
macro (5)       yield_curve, dxy, gold_corr, m2_supply, cpi   0.98
```
Each signal has its own Beta(α,β) distribution, updated per trade with `signal_discount=0.95`.

### 2. Regime-Aware Weights
Different signals perform differently in different markets:
- Momentum signals excel in trending markets
- Mean-reversion signals work in ranging markets

Solution: maintain separate Beta distributions **per regime** for each signal.
If a regime has <5 trades, fall back to global (all-regime) posterior.
Trades held < 5 minutes are excluded from posteriors (position-restore artifacts).

### 3. All-Weather Macro Regime (FRED API)
Inspired by Bridgewater All-Weather and IQC 2025 winner.
4-quadrant classification based on growth + inflation direction:

```
              Inflation Rising      Inflation Falling
           ┌──────────────────┬──────────────────┐
Growth     │   REFLATION      │   GOLDILOCKS     │
Rising     │   neutral         │   risk-on (1.2x) │
           ├──────────────────┼──────────────────┤
Growth     │   STAGFLATION    │   DEFLATION      │
Falling    │   risk-off (0.6x)│   cautious (0.9x)│
           └──────────────────┴──────────────────┘
```

Each regime scales `position_pct` and tightens/loosens trailing stops.

### 4. Event-Driven, Not Cron
Markets don't move on a schedule. A cron that runs every 15min will miss a 5% flash crash at minute 3,
then make a stale decision at minute 15. Instead:
- `MarketListener` watches Binance WebSocket 24/7
- `DerivativesMonitor` adaptively polls (5min base → 1min when extreme)
- Decisions triggered by events, not clocks

### 5. Agent-Autonomous Position Management (완벽한 Agent화)
- **Layer 1 (Catastrophic Safety, 30s)**: Only fires at -5% SL or +5% TP — flash crash protection ONLY
- **Layer 2 (Agent, 3min)**: Claude evaluates ALL positions — decides HOLD/EXIT/ADD autonomously

**Philosophy**: Agent makes ALL exit decisions. No hardcoded SL/TP/trailing/time-stop overrides.
Claude learns optimal exit timing through H-TS + Trading Diary (self-reflection RL).
Safety layer is catastrophic-only — it should never fire in normal operation.

### 6. Derivatives as Alpha (Not Just Risk)
Most systems use funding rate for risk only. We use 4 microstructure signals as **directional alpha**:
- **Funding Rate** (weight 0.35): Extreme positive = longs crowded = bearish
- **Taker Delta/CVD** (weight 0.30): Buy/sell volume imbalance = order flow direction
- **Long/Short Ratio** (weight 0.25): Top trader positioning, contrarian signal
- **Open Interest** (weight 0.10): Momentum confirmation

---

## Directory Structure

```
python/
├── config.py                  # All parameters (see "Configuration" section below)
├── server.py                  # FastAPI REST API (1750 lines, 30+ endpoints)
├── requirements.txt           # Dependencies
│
├── core/                      # Event-driven infrastructure
│   ├── event_bus.py           # Async pub/sub: publish("market.candle_close", {...})
│   ├── market_listener.py     # Binance WS: 1m/5m/15m/1h klines (24/7 stream)
│   ├── ohlcv_store.py         # Thread-safe WS-fed OHLCV bar accumulator
│   ├── derivatives_monitor.py # Adaptive poll: funding, OI, CVD, L/S ratio
│   ├── pipeline.py            # n8n-style node pipeline (15 nodes)
│   ├── position_tracker.py    # 2-layer position monitoring (30s/3min)
│   ├── agent_evaluator.py     # 5-signal position scoring
│   ├── online_learner.py      # Hierarchical Thompson Sampling v3 (28 signals, 6 groups)
│   ├── risk_manager.py        # Drawdown guards, exposure limits
│   ├── claude_auth.py         # Claude auth (API key or OAuth)
│   ├── claude_agent.py        # PRIMARY decision maker (Claude Sonnet 4.6 via SDK, 7 MCP tools, pre-computed TA)
│   ├── agent_tools.py         # 7 MCP tools for Claude Agent (TA, volume, derivatives, etc.)
│   ├── zscore_gate.py         # 4-layer adaptive gate (Welford z-score + wake conditions)
│   ├── trading_diary.py       # 3-layer self-reflection memory (7 papers, Ebbinghaus decay)
│   └── ws_stream.py           # WebSocket broadcast to frontend/Claude
│
├── agents/                    # Signal generators
│   ├── market_agent.py        # Regime + sector momentum → signal [-1, 1]
│   ├── quant_agent.py         # XGBoost cross-sectional ranking (equities)
│   ├── quant_agent_crypto.py  # XGBoost for crypto
│   └── synthesizer.py         # Weighted voting: agents × weights → BUY/SELL/HOLD
│
├── analysis/                  # Market analysis modules
│   ├── regime_detector.py     # HMM 2-state volatility regime (SPY daily)
│   ├── regime_detector_crypto.py  # Crypto regime (BTC/ETH vol + funding)
│   ├── macro_regime.py        # FRED 4-quadrant macro (growth × inflation)
│   ├── data_processor.py      # OHLCV loading, feature engineering
│   ├── sector_scanner.py      # Moskowitz momentum sector rotation
│   ├── stock_ranker.py        # XGBoost v8 quantitative ranking
│   ├── sentiment_finbert.py   # FinBERT news sentiment scoring
│   ├── sentiment_scorer.py    # Sentiment aggregation
│   ├── ensemble_agent.py      # PPO/A2C/SAC ensemble (equities)
│   ├── auto_trader.py         # Automated execution orchestrator
│   └── lgbm_signal.py         # DEPRECATED: LightGBM 15m (AUC 0.54, useless)
│
├── brokers/                   # Multi-platform execution
│   ├── base.py                # Abstract: connect, execute, get_positions
│   ├── binance.py             # Binance spot/futures via ccxt
│   ├── alpaca.py              # Alpaca REST API (US stocks)
│   ├── kis.py                 # 한국투자증권 via mojito2
│   └── factory.py             # BrokerRegistry pattern
│
├── backtests/                 # Historical validation
│   ├── backtest_swing.py      # 4h swing: +12.7%, Sharpe 0.72, MDD -19.1%
│   ├── backtest_v2.py         # Daily regime blend
│   ├── backtest_pipeline.py   # Full pipeline test
│   ├── backtest_optimize.py   # Parameter grid search
│   └── run_full_analysis.py   # Run all backtests
│
├── binance/                   # Crypto-specific runner
│   ├── main.py                # CLI: --testnet --futures --leverage 3
│   ├── runner.py              # CryptoRunner: 6 concurrent async loops
│   ├── api.py                 # Dashboard API (FastAPI, port 8080) + MCP server
│   ├── crypto_config.py       # SWING vs DAILY mode selection
│   ├── tests/                 # Test suite + backtest reports
│   ├── data/                  # OHLCV cache + runner_state.json
│   │   └── diary/             # Trading Diary (L1: reflections.jsonl, L2: daily/*.json, L3: semantic_lessons.json)
│   ├── models/                # Trained models + online_learner.json
│   └── logs/                  # Runtime logs
│       ├── trades.csv         # 거래 이력 (타임스탬프/가격/수량/PnL)
│       ├── decisions.jsonl    # Claude 판단 기록 (JSON, 기계용)
│       ├── agent_decisions.txt # Claude 판단 상세 (사람이 읽는 용)
│       └── trading.log        # 전체 엔진 로그
│
└── templates/
    └── chart.html             # TradingView Lightweight Charts v5 dashboard
```

---

## Configuration (config.py)

### Safety Parameters — Agent-Autonomous (2026-02-27)

**Philosophy**: Claude agent가 RL(Thompson Sampling)로 모든 entry/exit/sizing을 학습. Safety layer는 **catastrophic-only** (flash crash 방어).

| Parameter | Value | Role |
|-----------|-------|------|
| `stop_loss_pct` | **-5%** | Catastrophic SL — flash crash protection only. Agent exits earlier via RL |
| `take_profit_pct` | **+5%** | Catastrophic TP — agent takes profit before this fires |
| `max_hold_hours` | **4h** | Maximum hold time — agent learns optimal exit timing |
| `max_position_pct` | 0.15 | 15% max single position (portfolio protection) |
| `max_exposure_pct` | 0.60 | 60% max total exposure (diversification floor) |
| `kelly_fraction` | 0.15 | Conservative Kelly for crypto volatility |
| `scalp_trail_activation` | +2% | Trailing activates at +2% peak (effectively disabled for scalps) |
| `scalp_trail_width` | 1% | 1% trail width — only catches big runners |
| `profit_protect_peak` | +3% | Effectively disabled — agent protects profits via RL |
| `dca_stop_loss_pct` | -5% | DCA catastrophic SL (same as base) |
| `dca_max_hold_hours` | 4h | DCA hold limit (same as base) |

**Removed hardcoded overrides** (agent learns these):
- ~~BUY confidence filter~~ (was adaptive 0.30→0.55) — agent decides conviction
- ~~EXIT confidence gate~~ (was >0.3 required) — agent decides all exits
- ~~Position size clamp to 20%~~ → uses config `max_position_pct` (15%)
- ~~120s re-entry cooldown~~ → 30s (minimum rate limiter only)

### Agent Weights (synthesizer.py, default before TS adaptation)

| Agent | Weight | What It Does |
|-------|--------|-------------|
| quant | 0.35 | XGBoost P(outperform) ranking — primary signal |
| market | 0.24 | Regime detection + sector momentum |
| momentum | 0.18 | Raw price momentum (14d, 30d) |
| sentiment | 0.12 | FinBERT news sentiment |
| regime | 0.11 | Defensive/offensive bias from HMM state |
| rl | 0.00 | Disabled — Thompson Sampling replaces this |

These are **starting weights**. After 5+ trades, OnlineLearner samples from Beta posteriors instead.

### Event Configuration (binance/crypto_config.py)

| Param | Value | Meaning |
|-------|-------|---------|
| `kline_intervals` | 1m/5m/15m/1h | WS candle timeframes (1m=gate, 5m/15m/1h=analysis) |
| `significant_move_pct` | 0.015 | 1.5% BTC move triggers re-evaluation |
| `min_decision_gap` | 60s | Minimum 60s between decisions |
| `funding_extreme_threshold` | 0.0005 | 0.05%/8h funding = extreme |
| `oi_spike_threshold` | 0.10 | 10% OI change = spike |

### REST vs WebSocket 전략 (2026-02-25)

| 컴포넌트 | REST calls | 언제 |
|----------|-----------|------|
| `connect()` | 1 (fetch_balance) | 시작 시 1회 |
| OHLCV bootstrap | 12 (4 tickers × 3 TFs: 5m/15m/1h) | 시작 시 1회 |
| Derivatives monitor | ~28/cycle | 15분마다 (testnet) |
| Decision loop | **0** | dry_run에서 절대 안 함 |
| Price fetcher | **0** | WS tick → OHLCV store |
| **분당 평균** | **~3.3 weight** | Binance 한도: 6000/분 |

IP ban 방지: `broker._ip_ban_until`이 모든 REST를 자동 차단.
ban 중 재시도 → ban 에스컬레이션 (10분→30분→50분). **절대 하면 안 됨.**

---

## Backtest Results

### 4h Swing Mode (11 months, BTC/ETH/SOL)
```
Return:     +12.7%   (vs BTC B&H: -6.0%)
Alpha:      +18.7%
Sharpe:     0.72
MDD:        -19.1%
Win Rate:   44%
Avg Win:    +20.3%
Avg Loss:   -10.5%
Risk Exits: 48%      (was 81% before parameter tuning)
Trades:     25        (~2.3/month)
```

### Daily Mode (1 year, BTC/ETH/SOL)
```
Return:     +14.0%   (vs BTC B&H: -21.9%)
Alpha:      +35.9%
Sharpe:     0.57
MDD:        -20.2%
```

### Claude Agent Backtest (2 weeks, BTC/ETH/SOL, 2025-03-13~27)
```
Return:     -0.5%    (vs BTC B&H: -4.7%)
Alpha:      +4.2%
MDD:        -0.9%
Trades:     9         (Claude SDK 14/14 calls successful)
Mode:       24h rebalance, derivatives data limited (testnet)
Note:       Claude correctly stayed conservative with missing CVD/L-S data
```

### LightGBM 15min Experiment (FAILED)
```
AUC:        0.54     (barely above random)
Return:     -6.9%
Sharpe:     -0.30
Verdict:    Rule-based regime blend >> ML for crypto short-term
```

---

## Trading Logic (Entry / Exit / Sizing)

### Production: Claude Agent with 7 MCP Tools
Claude receives the MarketSnapshot, then **independently calls tools** to analyze deeper.
No hard-coded entry/exit rules in production. Claude decides everything: BUY, SELL, HOLD, ADD (DCA).

**Agent Workflow per Decision (max_turns=3):**
1. Read MarketSnapshot — includes **pre-computed TA** for 5m/15m/1h (RSI, MACD, BB, EMA, StochRSI, ATR, VWAP)
2. If pre-computed data sufficient → proceed directly to decision (0 tool calls)
3. Call tools ONLY for: custom periods, volume profile, derivatives deep dive, position check
4. Synthesize all data → output JSON with decisions

**Speed optimization:** TA indicators are pre-computed in the snapshot build (~15ms).
This eliminates 2-3 `technical_analysis` tool calls that previously added 10-20s of Claude thinking time.
Typical decision time: **21-29s** (was 33-42s with Haiku + 5 tool calls).

**ADD Action (DCA / 평단가 낮추기):**
When a position is losing but thesis is intact (위기 = 기회), Claude can issue ADD:
- Only allowed on existing positions
- Calculates weighted average entry price
- Subject to 60% total portfolio exposure limit
- **Converts position to DCA mode**: safety SL widens from -0.7% to -1.5%, hold from 45min to 2h
- trailing_high resets to max(new avg entry, current price)
- H-TS tracks ADD outcomes separately for learning

### Backtesting Fallback: Rule-Based Pipeline (RegimeBlendSignalNode)
```
IF regime == TRENDING:
    BUY when: RSI_14 > 50 AND momentum_14d > 2%
    SELL when: momentum_14d < -5%

IF regime == RANGING:
    BUY when: price < BB_lower * 1.02 AND RSI_14 < 40
    SELL when: price > BB_upper * 0.98
```
Each signal is weighted by agents (quant 0.35, market 0.24, momentum 0.18, sentiment 0.12, regime 0.11).
Thompson Sampling dynamically adjusts these weights per regime after 5+ trades.

### Exit Rules (2 layers, 6 checks — position-type-aware)
```
Two position types with different safety rules:

SCALP (default BUY):        DCA (after ADD):
  SL:    -0.7%                SL:    -1.5%
  Trail: +0.4% act / 0.2% w  Trail: +0.8% act / 0.4% w
  Hold:  45min                Hold:  2h
  TP:    +0.8% (same)         TP:    +0.8% (same)

Layer 1 — Safety (every 30 seconds):
    1. Hard SL:        EXIT if PnL <= SL (scalp: -0.7%, DCA: -1.5%)
    2. Take Profit:    EXIT if PnL >= +0.8%
    3. Mode Trail:     EXIT if peak >= trail_act AND price drops trail_width from peak
    4. Swing Trail:    EXIT if peak >= +4% AND price drops 6% from peak (rare big runners)
    5. Profit Protect: EXIT if peak >= +0.3% AND current PnL < +0.1%
    6. Time Stop:      EXIT if held > max_hold (scalp: 45min, DCA: 2h)

Layer 2 — Claude Re-evaluation (every 3 minutes):
    ClaudeAgent.evaluate_position() → HOLD / EXIT / ADD with confidence
    DCA mode info is passed to Claude for context-aware re-evaluation
    Fallback (if Claude unavailable): 5-signal weighted scoring

Position type transition:
    BUY → tracked as SCALP (tight stops)
    ADD on existing position → converts to DCA (wider stops, longer hold)
    Entry price recalculated as weighted average after ADD
    trailing_high reset to max(new_avg_entry, current_price)
```

### Position Sizing (All-Weather)
```
base_size = portfolio_value * position_pct (default 30%)

Macro regime scaling:
    GOLDILOCKS  → base_size * 1.2  (risk-on: growth↑ inflation↓)
    REFLATION   → base_size * 1.0  (neutral)
    STAGFLATION → base_size * 0.6  (risk-off: growth↓ inflation↑)
    DEFLATION   → base_size * 0.9  (cautious)
```

### Strategy Goal
**NOT return maximization.** Risk-adjusted return (Sharpe) maximization.
- IQC 2025 winner insight: stability across all macro regimes > chasing returns
- Target: Sharpe > 0.5, MDD < 25%, positive alpha vs BTC B&H
- Backtest validated: Sharpe 0.72, alpha +18.7% vs BTC

### Traded Assets

| Ticker | Type | Mode | Description |
|--------|------|------|-------------|
| BTC/USDT | Crypto | Spot/Futures | Bitcoin — 주력 |
| ETH/USDT | Crypto | Spot/Futures | Ethereum |
| SOL/USDT | Crypto | Spot/Futures | Solana |
| PAXG/USDT | Gold RWA | Spot | 1 PAXG = 1 troy oz 금. 비상관 자산 (올웨더) |
| XAU/USDT:USDT | Gold Perp | Futures (TODO) | 금 선물 퍼프, 최대 50x |
| XAG/USDT:USDT | Silver Perp | Futures (TODO) | 은 선물 퍼프, 최대 50x |

PAXG는 크립토 하락장 헤지용 — 금은 BTC와 상관관계 낮아 포트폴리오 분산 효과.
XAU/XAG는 선물 모드 안정화 후 추가 예정.

---

## Scalping Setups (Claude 참고용 — H-TS 데이터가 이를 오버라이드)

### Setup A: Mean Reversion Scalp (횡보/저변동 레짐)
**핵심 시그널 (더 많은 confluence = 높은 confidence):**
1. StochRSI < 0.10 on 5m (극과매도)
2. Price at/below lower Bollinger Band (20, 2σ)
3. Price below VWAP

**보너스 (확신도 상승):** RSI divergence, CVD divergence, Volume spike > 1.5x
**주의:** 5m EMA(9) < EMA(21) AND 15m MACD histogram < 0일 때 약화됨

### Setup B: Momentum Breakout Scalp (추세 레짐)
**3가지 필수 조건:**
1. EMA(9) crosses above EMA(21) on 5m
2. RSI(14) > 50 (과매수 아닌 상태)
3. Price above VWAP

**보너스:** Rising CVD, OI 증가 + positive funding (but not extreme)

### Setup C: Liquidation Flush Bounce (폭락장)
- Liquidation cascade 감지 (1h liquidation > 3x normal)
- Cascade 소진 후 CVD divergence + StochRSI < 0.10에서 진입

**Sources:** opofinance.com (70-75% win rate backtest), Bookmap CVD Guide, Cryptowisser Fib+VWAP+EMA

---

## Trading Diary (3-Layer Self-Reflection Memory)

7개 논문 기반 자기성찰 시스템. 매 거래 후 자동 기록, 매일 Claude가 1회 분석.

```
Layer 1: Per-Trade Reflection (JSONL, template, no LLM)
    → binance/data/diary/reflections.jsonl
    Papers: Reflexion (Shinn 2023), Chain of Hindsight (Liu 2023), ECHO (Hu 2025), LLM Regret (Park 2024)

Layer 2: Daily Digest (Claude 1x/day, MD + JSON)
    → binance/data/diary/daily/YYYY-MM-DD.json (기계용)
    → ~/Documents/trading-diary/YYYY-MM-DD.md (사람이 읽는 용)
    분석 항목: Win/Loss asymmetry, tiny wins, per-ticker breakdown, profit_protect 효과
    Paper: FinMem (Yu 2024)

Layer 3: Weekly Semantic Distillation (no LLM, Ebbinghaus decay)
    → binance/data/diary/semantic_lessons.json (max 20 lessons)
    Papers: SAGE (Wei 2024), TradingGroup (Tian 2025)

Prompt Injection: MarketSnapshot.diary_context → ~300 tokens
    현재 레짐에 맞는 교훈 + 최근 3거래 결과를 Claude에게 전달
```

---

## Agent 전체 로직 플로우 (한눈에 보기)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT LIFECYCLE (매 틱마다)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SENSE (데이터 수집 — 실시간)                                       │
│  ├── Binance WebSocket → 1m/5m/15m kline 가격/볼륨 (매초)             │
│  ├── DerivativesMonitor → Funding/OI/CVD/L-S Ratio (5분 폴링)        │
│  ├── Fear & Greed Index (alternative.me, 1h 캐시)                    │
│  ├── Stablecoin Supply (DeFiLlama, 4h 캐시)                          │
│  ├── Liquidation Cascade (Binance REST, 1h 윈도우)                    │
│  ├── CoinGecko Trending (시장 버즈, 1h 캐시)                          │
│  └── FRED Macro Regime (성장×인플레 4분면, 6h 캐시)                     │
│                                                                     │
│  2. FILTER (AdaptiveGate — z-score 항상 활성)                         │
│  ├── Gate 1: 캔들 마감? → 쿨다운만 체크, 타이머 무시                      │
│  ├── Gate 2: Claude 타이머 만료? (루틴 체크용)                           │
│  ├── Gate 3: Z-score 이상치? (|z| >= 1.5~2.5) ← 항상 활성!             │
│  │   └── 급변 시 타이머/블로킹 무시, 즉시 Claude 호출                     │
│  ├── Gate 4: Claude wake_conditions 충족?                            │
│  └── 쿨다운: min 60s (유일한 호출 제한)                                  │
│                                                                     │
│  3. THINK (Claude Haiku 4.5 — TRUE AGENT, 7 MCP 도구)                │
│  ├── MarketSnapshot 구축 (모든 데이터 한 곳에)                          │
│  │   ├── 가격 + 수익률 + 볼륨                                         │
│  │   ├── 파생상품 (funding, CVD, L/S, OI)                             │
│  │   ├── 레짐 (crypto HMM + FRED macro)                              │
│  │   ├── H-TS v3 사후확률 (28 시그널 × 6 그룹, 레짐별)                  │
│  │   ├── 최근 10 거래 (학습 맥락)                                      │
│  │   ├── 보유 포지션 + 포트폴리오 상태                                   │
│  │   ├── 센티먼트 (F&G, 뉴스, 청산, 스테이블코인)                        │
│  │   ├── 에이전트 메모리 (이전 판단 메모)                                │
│  │   └── Trading Diary (최근 3거래 + Ebbinghaus-filtered 교훈)        │
│  ├── Claude가 자체적으로 도구 호출 (5-9회 per decision):                 │
│  │   ├── technical_analysis: RSI, MACD, EMA, BB, StochRSI, ATR, VWAP │
│  │   ├── volume_analysis: 볼륨 프로파일, 매수/매도 압력, 다이버전스       │
│  │   ├── get_derivatives: 펀딩레이트, OI, L/S 비율, 청산                │
│  │   ├── check_position: 보유 포지션 상세 (PnL, 트레일링스톱)            │
│  │   ├── search_trades: 과거 거래 검색 (필터 가능)                      │
│  │   ├── get_ts_weights: H-TS 사후확률 (어떤 시그널이 신뢰할만한지)       │
│  │   └── get_ohlcv: 원시 캔들 데이터 (타임프레임별)                      │
│  ├── 분석 완료 후 → JSON 응답                                         │
│  │   ├── decisions: BUY/SELL/HOLD/ADD (티커별)                        │
│  │   ├── position_pct: 포지션 크기 (최대 20%)                          │
│  │   ├── confidence: 신뢰도 (0~1)                                    │
│  │   ├── reasoning: 판단 이유 (상세)                                   │
│  │   ├── signal_weights: 사용한 시그널 가중치 (TS 학습용)                │
│  │   ├── next_check_seconds: 다음 깨울 시간 (자기 스케줄링)              │
│  │   ├── wake_conditions: 조건부 깨우기 (예: BTC<61K)                  │
│  │   └── memory_update: 다음 깨울 때 참고할 메모                        │
│  └── Fallback: Claude 불가 시 → 규칙 기반 파이프라인 (백테스트용)          │
│                                                                     │
│  4. ACT (주문 실행)                                                   │
│  ├── BUY: Binance 브로커 → 주문 전송 (dry_run or LIVE)                 │
│  ├── SELL: 포지션 정리 → PnL 계산 → TS 학습 피드백                      │
│  ├── ADD (DCA): 기존 포지션에 추가 매수 → 가중평균 진입가 + DCA모드 전환   │
│  │   └── new_avg = (old_entry×old_qty + new_price×add_qty) / total     │
│  ├── 주문 성공 확인 후에만 포지션 추적 시작                               │
│  ├── trades.csv 기록 (타임스탬프/가격/수량/PnL/이유/레짐)                 │
│  └── decisions.jsonl 기록 (Claude 전체 응답 + 도구 호출 로그)             │
│                                                                     │
│  5. MONITOR (PositionTracker — 2레이어)                               │
│  ├── Layer 1 Safety (30초): SL -5%, 트레일링스톱 -6%@+4% (SWING)     │
│  ├── Layer 2 Agent (3분): Claude 재평가 → HOLD/EXIT/ADD               │
│  └── 위험 퇴출 → position.exit 이벤트 발행                              │
│                                                                     │
│  6. LEARN (Hierarchical Thompson Sampling v3)                         │
│  ├── position.exit 수신 → record_trade()                              │
│  ├── 5분 미만 보유 거래 → 포스테리어 업데이트 제외 (복원 아티팩트)          │
│  ├── Claude signal_weights (1순위) or keyword fallback (2순위)         │
│  ├── 28개 시그널 × 6 그룹 Beta 분포 업데이트                             │
│  │   ├── signal_discount=0.95 (시그널별)                               │
│  │   └── group_discount=0.98 (그룹별)                                  │
│  ├── 레짐별 별도 Beta (trending에서 momentum 잘 맞으면 ↑)               │
│  ├── < 5 거래 레짐 → 글로벌 fallback                                   │
│  ├── 다음 결정 시 sample_weights(regime) → 적응된 가중치                 │
│  └── models/online_learner.json 자동 저장                             │
│                                                                     │
│  7. SCHEDULE (자기 모니터링)                                           │
│  ├── AdaptiveGate.update_from_claude()                               │
│  ├── 다음 타이머 설정 (quiet: 1-4h, active: 5-30min)                   │
│  ├── Wake conditions 설정 (가격 레벨, 변동폭 등)                        │
│  └── 에이전트 메모리 저장 (500자, runner_state.json)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 강화학습 동작 확인 체크리스트

| 단계 | 이벤트 | 확인 방법 |
|------|--------|-----------|
| BUY 실행 | `position_tracker.track()` | 로그: `[tracker] Tracking BTC/USDT` |
| ADD (DCA) 실행 | 가중평균 진입가 계산 | 로그: `[runner] ADD BTC/USDT: avg entry $X → $Y` |
| 포지션 감시 | 30s safety + 3min agent | 로그: `[tracker] Safety check`, `[tracker] Agent eval` |
| EXIT 발생 | `position.exit` 이벤트 | 로그: `[runner] Position exit: BTC/USDT pnl=+X%` |
| H-TS 업데이트 | `record_trade()` | 로그: `[h-ts] Trade #N: BTC/USDT pnl=+X%` |
| 5분 미만 필터 | 아티팩트 제외 | 로그: `[h-ts] Trade #N: skipped (held < 5 min)` |
| Beta 저장 | `online_learner.save()` | 파일: `models/online_learner.json` |
| 가중치 적응 | `sample_weights(regime)` | 로그: H-TS posteriors in MarketSnapshot |

---

## CryptoRunner: 6 Concurrent Loops (binance/runner.py)

```python
async def run(self):
    await asyncio.gather(
        self._market_listener.run(),      # Loop 1: WebSocket klines
        self._derivatives_monitor.run(),   # Loop 2: Adaptive polling
        self._position_tracker.run(),      # Loop 3: 30s/3min position checks
        self._heartbeat_loop(),            # Loop 4: 30s Docker health
        self._watchdog_loop(),             # Loop 5: WS disconnect fallback
        start_api_server(self),            # Loop 6: Dashboard API (port 8080)
    )
```

**Decision flow (tick → gate → Claude Agent → execute):**
1. MarketListener receives WS tick → publishes `market.tick`
2. CryptoRunner._handle_tick() → _compute_tick_features() → AdaptiveGate.evaluate()
3. Gate decides: skip (90%+) or wake Claude
4. If wake: Build `MarketSnapshot` → **Claude Agent starts** (max_turns=5)
5. Claude calls 5-9 MCP tools: technical_analysis, volume_analysis, get_derivatives, etc.
6. Claude synthesizes tool results → JSON response (BUY/SELL/HOLD/ADD)
7. Response includes `signal_weights`, `next_check_seconds`, `wake_conditions`, `memory_update`
8. AdaptiveGate.update_from_claude() → sets timer + conditions for next wake
9. Execute BUY/SELL/ADD (only if broker confirms success)
10. ADD: weighted average entry price calculation → existing position updated
11. Track positions in PositionTracker / publish RL events
12. Position close → H-TS.record_trade(regime, signal_weights) → 28-signal Beta update

**MarketSnapshot contains initial context (Claude uses tools for deeper analysis):**
- Price action (OHLCV, returns, volume per ticker)
- Derivatives (funding rate, CVD, L/S ratio)
- Regime (crypto HMM + FRED macro 4-quadrant)
- H-TS posteriors (28 signals × 6 groups, per-regime)
- Recent trades (last 10 with PnL, for learning context)
- Open positions and portfolio state
- Agent memory (500 chars, persisted across restarts)

---

## Pipeline Nodes (core/pipeline.py)

| Node | Reads | Writes | Logic |
|------|-------|--------|-------|
| `RegimeBlendDetectNode` | crypto_close, eval_date | regime, volatility_state | HMM on BTC/ETH volatility |
| `DerivativesSignalNode` | derivatives_context | derivatives_signal [-1,1] | 4-signal weighted composite |
| `RegimeBlendSignalNode` | regime, crypto_close, candidates | scored (BUY/SELL/HOLD per ticker) | RSI/momentum in trending, BB in ranging |
| `RegimeBlendExitNode` | positions, scored, crypto_close | sells, updated positions, cash | DD/trail/time exit rules |
| `RegimeBlendEntryNode` | scored, positions, cash | new positions, updated cash | Position sizing with regime awareness |

---

## Hierarchical Thompson Sampling v3 (core/online_learner.py)

### Architecture: HierarchicalOnlineLearner
**28 signals** organized into **6 groups**, each with its own Beta(α,β) distribution.
Two-level discount: group_discount=0.98 (slow adaptation), signal_discount=0.95 (fast adaptation).

```
Group           │ Signals                                    │ Purpose
────────────────┼────────────────────────────────────────────┼──────────────────────
momentum (5)    │ rsi, macd, ema_cross, stoch_rsi, adx       │ 추세 추종 지표
market (5)      │ regime, crypto_regime, fear_greed, trend,   │ 시장 상태 지표
                │ vix                                        │
quant (5)       │ bb_squeeze, volume_profile, atr, obv, vwap │ 정량적 기술 지표
sentiment (3)   │ news, social, whale_alert                  │ 감성/소셜 지표
funding_rate(5) │ funding, oi_change, liquidations, basis,   │ 파생상품 미시구조
                │ cvd                                        │
macro (5)       │ yield_curve, dxy, gold_corr, m2_supply,    │ 거시경제 지표
                │ cpi                                        │
```

### How It Works
After every closed trade (held ≥ 5 minutes):

1. **Signal source priority:**
   - 1순위: Claude JSON의 `signal_weights` 필드 (Claude가 어떤 시그널을 사용했는지 직접 명시)
   - 2순위: Keyword fallback (`_extract_signals_from_reasoning()`)
2. Map PnL to reward via sigmoid: `reward = 1 / (1 + exp(-pnl * 100))`
   - +1% PnL → reward 0.731, -1% → 0.269
   - +0.5% → 0.622, -0.5% → 0.378
3. Update each signal's Beta: aligned → `alpha += reward`, else `beta += (1-reward)`
4. Discount: `signal α,β *= 0.95`, `group α,β *= 0.98`
5. To get weights: sample from each signal's Beta → group-level aggregation → normalize

### Regime-Aware Extension
```
Regime "trending_goldilocks":  momentum.rsi  Beta(3.2, 1.8) → mean 0.64 (strong)
Regime "ranging_stagflation":  momentum.rsi  Beta(1.5, 2.9) → mean 0.34 (weak)
```
If a regime has <5 trades → fall back to global posterior.

### Artifact Filtering
- Trades held < 5 minutes are excluded from posterior updates (position-restore artifacts)
- On load: auto-detects short trades and reprocesses all posteriors
- PnL 0% trades skipped during reprocessing

### Persistence
Saved to `models/online_learner.json` after every trade. Loaded on startup.
Contains: version, trades list, signal posteriors, group posteriors, per-regime posteriors.

---

## Macro Regime (analysis/macro_regime.py)

### FRED Series Used
| Indicator | FRED ID | Frequency | Purpose |
|-----------|---------|-----------|---------|
| Yield Curve 10Y-2Y | T10Y2Y | Daily | Growth direction (steepening = improving) |
| 10Y Breakeven Inflation | T10YIE | Daily | Inflation expectations |
| Unemployment Rate | UNRATE | Monthly | Growth confirmation (falling = good) |
| CPI All Items | CPIAUCSL | Monthly | Inflation confirmation |

### Output
```python
detector = MacroRegimeDetector()  # needs FRED_API_KEY in .env
regime = detector.detect()        # → MacroRegime.GOLDILOCKS
scale = detector.exposure_scale   # → 1.2 (aggressive)
trail = detector.trail_multiplier # → 1.0 (normal)
```
Cached for 6 hours (FRED data doesn't change intraday).

---

## OpenClaw Integration

### Architecture: Claude IS the Trader (True Agent with Tools)

```
Market Events → Data Collection → MarketSnapshot → ClaudeAgent.decide() → Execute
                                       ↑              ↓ (max_turns=5)         ↓
                                       │         ┌─ 7 MCP Tools ─────────┐    ↓
                                       │         │ technical_analysis     │    ↓
                                       │         │ volume_analysis        │    ↓
                                       │         │ get_derivatives        │    ↓
                                       │         │ check_position         │    ↓
                                       │         │ search_trades          │    ↓
                                       │         │ get_ts_weights         │    ↓
                                       │         │ get_ohlcv              │    ↓
                                       │         └────────────────────────┘    ↓
                                       └──── H-TS v3 feedback (28 signals) ←──┘
```

**Claude (Haiku 4.5) is a TRUE AGENT, not just an LLM.** The engine collects data, builds a snapshot,
Claude calls tools to independently analyze, then decides. Rule-based pipeline is for backtesting only.

- Model: `claude-haiku-4-5-20251001` (Haiku 4.5 — 빠른 응답, OAuth 비용 무관)
- Tools: 7 MCP tools via `core/agent_tools.py` (in-process, same Python process)
- max_turns: 5 (Claude can call tools up to 5 rounds before outputting decision)
- Timeout: 180s (tool calls need more time)
- Auth: `ANTHROPIC_API_KEY` (server/cloud) or Claude Code OAuth (local dev)
- SDK: `claude-agent-sdk` via `ClaudeAgentOptions(env=sdk_env, mcp_servers=..., allowed_tools=...)`
- Fallback: If no auth available, rule-based pipeline runs (backtesting mode)

### Authentication (core/claude_auth.py)

```python
# configure_sdk_authentication() — one-time auth setup
# get_sdk_env_vars()              — collect env vars for SDK subprocess
# ClaudeAgentOptions(env=...)     — pass env to SDK
#
# Auth priority:
# 1. ANTHROPIC_API_KEY env var (recommended for server/cloud — no expiry)
# 2. API profile (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)
# 3. CLAUDE_CODE_OAUTH_TOKEN env var (local dev)
# 4. Credentials file (~/.claude/.credentials.json) with auto-refresh
# 5. System credential store (macOS Keychain, Windows creds)
```

**Server/Cloud**: Use `ANTHROPIC_API_KEY` (from https://console.anthropic.com/settings/keys).
OAuth token refresh is blocked from datacenter IPs (Cloudflare 403 error 1010).

**Local dev**: OAuth auto-configured from Keychain/credentials. Run `claude login` to authenticate.

### Claude Agent (`core/claude_agent.py` + `core/agent_tools.py`)

| Method | When Called | Tools? | What It Does |
|--------|------------|--------|-------------|
| `decide(snapshot)` | Every market event | Yes (7 MCP tools) | BUY/SELL/HOLD/ADD decisions + position sizing |
| `evaluate_position()` | Position re-eval (volatile) | No (fast eval) | HOLD/EXIT/ADD verdict with confidence |

**7 MCP Tools** (in-process, `core/agent_tools.py`):
| Tool | Input | Returns |
|------|-------|---------|
| `technical_analysis` | ticker, timeframe | RSI, MACD, EMA(9/21), BB, StochRSI, ATR, VWAP |
| `volume_analysis` | ticker, timeframe | Volume profile, buy/sell ratio, divergence |
| `get_derivatives` | ticker | Funding rate, OI, L/S ratio, liquidations |
| `check_position` | ticker | Entry price, PnL, qty, trailing high, held hours |
| `search_trades` | ticker, action, limit | Past trade history with PnL |
| `get_ts_weights` | regime | H-TS posteriors (28 signals × 6 groups) |
| `get_ohlcv` | ticker, timeframe, limit | Raw OHLCV candle data |

Claude receives a full `MarketSnapshot`, calls tools independently, then returns:
```json
{
  "decisions": [{"ticker": "BTC/USDT:USDT", "action": "BUY", "position_pct": 0.15, "confidence": 0.85, "reasoning": "..."}],
  "signal_weights": {"rsi": 0.25, "macd": 0.20, "funding": 0.15, "volume_profile": 0.15, ...},
  "market_assessment": "one sentence overall market view",
  "regime_agreement": true,
  "learning_note": "what I learned from recent trade history",
  "next_check_seconds": 3600,
  "wake_conditions": [{"metric": "btc_price", "operator": "lt", "threshold": 80000, "reason": "support break"}],
  "memory_update": "BTC testing 80K support, watching for breakdown"
}
```

**`signal_weights`** field is NEW — Claude explicitly reports which signals it used and their relative importance. This feeds directly into H-TS for accurate per-signal learning (1순위). Keyword fallback is 2순위.

**Circuit breaker** (half-open pattern):
- 3 consecutive failures → OPEN (block all calls)
- After 15 min cooldown → HALF-OPEN (retry with forced re-auth for token refresh)
- Success → CLOSED (normal). Failure → OPEN again (restart 15 min cooldown)

Returns `None` if no auth available → caller uses rule-based fallback.

### MCP Tools (두 가지 레벨)

**Level 1: Agent Tools (in-process, `core/agent_tools.py`)**
Claude Agent가 의사결정 중 호출하는 7개 도구. 같은 Python 프로세스에서 실행.
`claude-agent-sdk`의 `@tool` 데코레이터 + `create_sdk_mcp_server()` 사용.

**Level 2: Dashboard API MCP (`binance/api.py`)**
`binance/api.py`가 `fastapi-mcp`로 모든 REST 엔드포인트를 MCP 도구로 노출.
Claude.ai에서 원격으로 엔진 상태를 조회 가능:

```
Claude.ai: "현재 레짐과 포지션 확인해줘"
→ MCP tool call: GET /api/status → {"portfolio_value": 74680, "positions": {...}}
→ MCP tool call: GET /api/ts-weights → {"mean_weights": {...}}
→ MCP tool call: GET /api/daily-pnl → {"days": [...], "summary": {...}}
```

Requires `pip install fastapi-mcp`.

---

## API Endpoints

### Dashboard API (`binance/api.py`, port 8080)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Portfolio, positions, regime, gate state, tick prices |
| GET | `/api/decisions` | Claude decision history (JSONL, latest N) |
| GET | `/api/trades` | Trade history (CSV) with summary stats |
| GET | `/api/ts-weights` | H-TS weights, group weights, regime info |
| GET | `/api/price-history` | OHLCV candle data from ohlcv_store |
| GET | `/api/equity-curve` | Portfolio value over time |
| GET | `/api/daily-pnl` | Daily PnL breakdown (wins/losses/cumulative per day) |
| GET | `/api/mcp/snapshot` | Full MarketSnapshot for MCP clients |
| GET | `/api/mcp/explain-trade/{idx}` | Explain specific trade by index |
| WS | `/ws/live` | Real-time state push (10s interval) |

### Server API (`server.py`, port 8787)

### Health & Status
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| GET | `/status` | Full system status |
| GET | `/broker/status` | Broker connection state |
| GET | `/tracker/status` | Position tracker state |
| GET | `/regime` | Current market regime |
| GET | `/learner/status` | Thompson Sampling posteriors per agent |
| GET | `/learner/weights` | Current adapted weights |

### Trading
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single-ticker prediction |
| POST | `/decide` | Multi-agent trading decision |
| POST | `/decide/v3` | v3 pipeline-based decision |
| POST | `/execute` | Execute order list |
| POST | `/risk/check` | Risk assessment |

### Analysis
| Method | Path | Description |
|--------|------|-------------|
| POST | `/backtest` | Run backtest |
| POST | `/backtest/pipeline` | Pipeline-specific backtest |
| POST | `/scan` | Sector scan |
| POST | `/quant/rank` | Quantitative ranking |
| POST | `/quant/retrain` | Retrain XGBoost model |

### Real-time
| Method | Path | Description |
|--------|------|-------------|
| WS | `/ws/stream` | WebSocket event broadcast |
| GET | `/chart` | TradingView dashboard |
| GET | `/events/recent` | Recent EventBus events |
| GET | `/api/signals` | Current signal state |

---

## Setup & Run

### Prerequisites
- Python 3.11+
- Binance API key (testnet: https://testnet.binance.vision/)
- Optional: FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)

### .env
```bash
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET_KEY=your_testnet_secret
BINANCE_PAPER=true        # Use testnet
LIVE_TRADING=false         # Safety: no real orders
BLEND_MODE=swing           # "swing" (4h) or "daily" (1d)
ANTHROPIC_API_KEY=sk-ant-api03-...  # Claude Agent (server/cloud)
FRED_API_KEY=your_key      # Optional: macro regime detection
```

### Install & Run
```bash
cd openclaw/extensions/trading-engine/python
pip install -r requirements.txt

# Crypto (event-driven, 24/7)
python binance/main.py --testnet

# Crypto with Futures leverage
python binance/main.py --testnet --futures --leverage 3

# API server (equities + dashboard)
python server.py
# → http://127.0.0.1:8787/chart (TradingView dashboard)
# → http://127.0.0.1:8787/docs  (Swagger API docs)
```

### Docker (Cloud / Mac Mini)
```bash
docker build -f binance/Dockerfile -t trading-engine .
docker run -d --name trader --restart=always \
  --env-file .env \
  -v trader-data:/app/binance/data \
  -v trader-models:/app/binance/models \
  -v trader-logs:/app/binance/logs \
  trading-engine

# Cloud deployment (Oracle Cloud Free Tier 추천)
# 1순위: Oracle Cloud ARM (무료, 전용 IP, 24GB RAM)
# 2순위: Fly.io ($3-7/월, 전용 static IP)
# 3순위: Railway ($5/월, 배포 편의성 최고)
```

### Cloud 환경변수 (Docker / Fly.io)
```bash
# Claude Agent (API key — 만료 없음, 리프레시 불필요)
ANTHROPIC_API_KEY=sk-ant-api03-...

# NOTE: OAuth (CLAUDE_CODE_OAUTH_TOKEN) 는 클라우드에서 사용 불가
# Cloudflare가 데이터센터 IP에서 토큰 리프레시 차단 (403 error 1010)
```

### Fly.io 배포
```bash
# 앱: fly.toml 참조 (nrt = Tokyo region)
# 머신: shared-cpu-1x, 1024MB RAM, 1GB volume
# 예상 비용: ~$3.34/월

fly auth login
fly secrets set BINANCE_API_KEY="$(grep BINANCE_API_KEY binance/.env | cut -d= -f2)"
fly secrets set BINANCE_SECRET_KEY="$(grep BINANCE_SECRET_KEY binance/.env | cut -d= -f2)"
fly secrets set ANTHROPIC_API_KEY="sk-ant-api03-..."
fly deploy
fly logs
```

---

## Honest Status (2026-02-26)

### 실제로 검증된 것
- **Claude = True Agent**: 7 MCP tools 연결, 의사결정당 5-9회 도구 호출 확인
- **Agent tool calls 검증**: `technical_analysis×3, volume_analysis×3, check_position, get_derivatives, get_ts_weights` 실제 호출 로그 확인
- 테스트넷 연속 가동: 11 trades 완료 (8 valid, 3 artifacts excluded)
- PnL: 총 -4.02% (PAXG -2.97% 버그 트레이드 제외 시 ~break-even)
- H-TS v3 학습 작동: 8 valid trades로 포스테리어 갱신 중
- Claude SDK (Haiku 4.5) 호출 성공, 구체적 판단 근거 + 도구 기반 분석
- AdaptiveGate z-score 트리거 + timer 만료 작동
- SELL → RL 피드백 루프: 통합테스트 4/4 PASS
- 포지션 재시작 복원: trailing_high 포함 영속화
- 5분 미만 아티팩트 거래 자동 감지 + 포스테리어 재처리
- **REST→WS 전면 전환**: 분당 REST weight ~3.3
- Dashboard API: `/api/daily-pnl`, `/api/status`, `/ws/live` 모두 작동
- MCP 서버: Dashboard API 전체 MCP 노출 (`fastapi-mcp`)
- ADD 액션 (DCA/평단가 낮추기) 구현 완료

### 검증 안 된 것
- **수익률**: 아직 통계적으로 무의미 (8 valid trades). 40+ trades 필요
- **ADD (DCA) 실행**: 코드 완료, Claude가 ADD를 아직 발동하지 않음
- **CVD / L-S Ratio**: 테스트넷 미지원 → N/A
- **FRED Macro**: API 키 없음 → regime="unknown"
- **Skills / Hooks / Graphiti**: 설계만, 연결 안 됨
- **장기 안정성**: 장기간 연속 운용 미확인

### 핵심 한계
- **Agent 메모리 한계**: 매 세션 새로 태어남. 500자 메모가 유일한 컨텍스트
- **인터넷/논문 검색 안 함**: 도구로 실시간 시장 데이터는 분석하지만 외부 리서치는 못함
- **OpenClaw 기능 미활용**: Skills, Hooks, Memory(Graphiti) 미연결
- **로컬 PC IP ban 문제**: Binance IP ban 에스컬레이션으로 로컬 테스트 제한적
- **FRED API key 필요**: 없으면 macro regime = "unknown" (중립)

---

## What's Done vs What's Next

### Fully Implemented
- [x] Event-driven runner (6 concurrent loops incl. watchdog + API)
- [x] **Claude = True Agent**: 7 MCP tools (TA, volume, derivatives, position, trades, TS, OHLCV)
- [x] **ADD action (DCA)**: 위기 = 기회, 기존 포지션에 추가매수 + 가중평균 진입가
- [x] AdaptiveGate: 4-layer gatekeeper (candle/timer/z-score/wake-conditions)
- [x] Claude self-scheduling (next_check_seconds + wake_conditions + memory)
- [x] **Hierarchical Thompson Sampling v3**: 28 signals × 6 groups, per-regime Beta
- [x] **5분 미만 아티팩트 필터**: 포지션 복원 거래 자동 제외
- [x] **signal_weights 우선**: Claude JSON → keyword fallback 순서
- [x] Derivatives signals (funding, OI, CVD, L/S ratio)
- [x] FRED macro regime detector (4-quadrant, all-weather)
- [x] 2-layer position tracking (safety 30s + agent 3min)
- [x] 5-signal agent evaluator (regime/momentum/correlation/PnL/time)
- [x] Pipeline architecture (15 nodes, n8n-style)
- [x] Multi-broker support (Binance, Alpaca, KIS)
- [x] TradingView dashboard (/chart)
- [x] **Dashboard API** (`binance/api.py`): status, trades, decisions, daily-pnl, equity-curve, ts-weights, WS live
- [x] **MCP 서버 (2레벨)**: Agent tools (in-process) + Dashboard API (fastapi-mcp)
- [x] Claude Agent as primary decision maker (claude_auth + claude_agent via claude-agent-sdk)
- [x] Circuit breaker with half-open recovery (15 min cooldown + forced re-auth)
- [x] Parameter tuning validated on 1-year backtest
- [x] **Atomic state write**: write-to-tmp + rename (crash 안전)
- [x] **Portfolio exposure limit**: 60% max enforced in `_convert_claude_decisions()`
- [x] **Trailing stop 영속화**: restart 시 trailing_high 복원
- [x] **OHLCVStore**: WS-fed OHLCV bar accumulator, REST bootstrap 1회 후 WS only
- [x] **REST→WS 전면 전환**: decision cycle 0 REST
- [x] **IP ban detection**: broker `_ip_ban_until` + `_detect_ip_ban()`
- [x] **Position restore**: `_restore_positions()` trailing_high 포함
- [x] **Docker**: Dockerfile + .dockerignore + healthcheck

### Not Yet Implemented (P1-P2)
- [x] Liquidation cascade detection (REST polling in DerivativesMonitor)
- [x] Fear & Greed index filter (alternative.me API, cached 1h)
- [x] Stablecoin supply as buying power proxy (DeFiLlama API, cached 4h)
- [x] CoinGecko trending coins as market buzz (cached 1h)
- [x] PAXG/USDT (금 RWA 토큰) 스팟 추가
- [x] Trade logging (trades.csv + decisions.jsonl)
- [x] Basis spread signal (perp-spot premium, annualized)
- [x] BTC ETF daily flow as institutional sentiment (CoinGlass)
- [x] PnL 분석 도구 (`binance/tests/analyze_trades.py`)
- [ ] **거시경제 유동성 지표**: Fed Net Liquidity (`WALCL - WTREGEN - RRPONTSYD`), DXY, HY Spread
- [ ] **Global M2**: US+EU+JP+CN M2 합산 (90일 lag) — BTC와 0.6~0.9 상관
- [ ] Exchange net flow / whale tracking (CryptoQuant, $29/mo)
- [ ] XAU/XAG 선물 모드 추가 (선물 안정화 후)
- [ ] End-to-end testnet soak test (1주일 연속 운용)
- [ ] OpenClaw Skills / Hooks / Graphiti 연결

### Known Issues / Tech Debt
- **TS signal differentiation**: ~~Architecture change needed~~ FIXED — Claude now reports `signal_weights` in JSON output (1순위), keyword fallback (2순위). Per-signal learning now works.
- **online_learner load**: `_trades` list restored from JSON. 5분 미만 아티팩트 자동 감지 + 재처리.
- **Process restart**: Fixed — `runner_state.json` persists entry_prices + agent_memory + trailing_highs, `_restore_positions()` recovers on startup.
- **Gate wake condition**: Claude가 `threshold: 'activated'` (문자열) 보낼 수 있음 → Gate가 gracefully 무시하지만, 바람직하지 않음. Claude 프롬프트에서 제한 필요.
- `lgbm_signal.py` is dead code (AUC 0.54). Kept for reference, never loaded.
- Equity cron scheduler in server.py still uses fixed timers (not event-driven like crypto).
- `FRED_API_KEY` is optional — without it, macro regime returns UNKNOWN (neutral exposure).
- `fastapi-mcp` is optional — server works without it, just no MCP tool exposure.
- `claude-agent-sdk` is required for agent mode — without it, Claude falls back to rule-based pipeline.

### Bug Fixes (2026-02-24~26 Code Review, 총 50+건)

**2026-02-26 (종합 리뷰)**
| ID | Severity | Fix | File |
|----|----------|-----|------|
| E1 | CRITICAL | PnL always 0% — `_build_snapshot()` used OHLCV close instead of WS prices | `runner.py` |
| E2 | CRITICAL | TS sigmoid scale 10→100 (old: +1%=0.525, new: +1%=0.731) | `online_learner.py` |
| E3 | CRITICAL | Portfolio exposure limit 60% enforced in `_convert_claude_decisions()` | `runner.py` |
| E4 | CRITICAL | Trailing stop state persisted across restarts | `runner.py` |
| E5 | CRITICAL | Atomic state write (write-to-tmp + rename) | `runner.py`, `online_learner.py` |
| E6 | HIGH | HTTP calls parallelized in `_build_snapshot()` (5 sequential→gather) | `runner.py` |
| E7 | HIGH | Per-ticker candle close debounce (was global, blocking other tickers) | `market_listener.py` |
| E8 | HIGH | Welford z-score (fixes float cancellation on BTC $90K+) | `zscore_gate.py` |
| E9 | HIGH | Safety config dynamic in prompt (was hardcoded, config mismatch) | `claude_agent.py` |
| E10 | HIGH | `_on_exit` lock protection (fire-and-forget EventBus race condition) | `runner.py` |
| E11 | MEDIUM | Hardcoded balance fallbacks removed (327K, 74K) → always use broker | `runner.py` |
| E12 | MEDIUM | Claude output validation (position_pct try/except) | `runner.py` |
| E13 | MEDIUM | EventBus.publish() fire-and-forget (was blocking WS) | `event_bus.py` |
| E14 | MEDIUM | MultiTFAggregator maxlen 20→50 (EMA21 needs 21 bars) | `multi_tf_aggregator.py` |
| E15 | MEDIUM | Safety prompt 8%/12% → 4%/6% (SWING mode mismatch) | `claude_agent.py` |
| E16 | MEDIUM | next_check_seconds prompt 60-3600 → 60-1800 (config max) | `claude_agent.py` |
| E17 | MEDIUM | Stale TS posteriors auto-detect + reprocess on load | `online_learner.py` |
| E18 | MEDIUM | Anti-churn 10min cooldown after selling | `runner.py` |
| E19 | MEDIUM | Price fetch failure: debug→warning in position_tracker | `position_tracker.py` |
| E20 | MEDIUM | OHLCV df fetch wrapped in asyncio.to_thread | `runner.py` |
| E21 | LOW | MCP server mounted on Dashboard API | `api.py` |
| E22 | CRITICAL | Claude = True Agent (7 MCP tools, max_turns=5, model Haiku 4.5) | `claude_agent.py`, `agent_tools.py` |
| E23 | CRITICAL | ADD action (DCA/averaging down) 지원 | `runner.py`, `claude_agent.py` |
| E24 | HIGH | H-TS v3: 28 signals × 6 groups, hierarchical posteriors | `online_learner.py` |
| E25 | HIGH | 5분 미만 아티팩트 거래 TS 포스테리어 제외 | `online_learner.py` |
| E26 | HIGH | _MIN_REGIME_TRADES 3→5 (insufficient data guard) | `online_learner.py` |
| E27 | HIGH | signal_weights 1순위, keyword fallback 2순위 | `runner.py` |
| E28 | MEDIUM | Daily PnL dashboard API (`/api/daily-pnl`) | `api.py` |

**2026-02-25 (이전 세션)**
| ID | Severity | Fix | File |
|----|----------|-----|------|
| D1 | CRITICAL | `broker.exchange` AttributeError → property 추가, 파생상품 5/7 소스 복구 | `brokers/binance.py` |
| D2 | CRITICAL | Dockerfile `core/`, `analysis/`, `brokers/` COPY 누락 → 컨테이너 즉시 크래시 | `binance/Dockerfile` |
| D3 | CRITICAL | `_liq_seen_ts` set 무한 증가 (메모리 누수) → maxlen + periodic sync | `derivatives_monitor.py` |
| D4 | IMPORTANT | EventBus `_event_log` list 재할당 churn → `deque(maxlen=500)` | `event_bus.py` |
| D5 | IMPORTANT | `.dockerignore` 없음 → .env/로그/캐시 이미지 유출 방지 | `.dockerignore` |
| D6 | CRITICAL | dry_run SELL → `_get_position_qty()` REST 호출 → tracker qty 사용 | `runner.py`, `brokers/binance.py` |
| D7 | CRITICAL | price_fetcher REST fallback → WS tick + OHLCV store (REST 0) | `runner.py` |
| D8 | CRITICAL | `get_account_status()` startup REST 제거 → connect 결과 사용 | `main.py`, `runner.py` |
| D9 | MAJOR | IP ban detection + 자동 REST 차단 | `brokers/binance.py` |
| D10 | MAJOR | derivatives_monitor ban check + testnet poll 간격 5→15min | `derivatives_monitor.py` |

**2026-02-24 (이전 세션)**
| ID | Severity | Fix | File |
|----|----------|-----|------|
| C1 | CRITICAL | `position.exit` 이중 발행 방지 — safety exit 후 Claude SELL skip | `runner.py` |
| C2 | CRITICAL | Circuit breaker half-open 복구 — 15분 후 재시도 + 강제 re-auth | `claude_agent.py` |
| M1 | MAJOR | BUY 주문 실패 시 phantom position 방지 — exec 결과 확인 후 track | `runner.py` |
| M3 | MAJOR | Safety trailing stop 임계값 통일 — config trail_pct(12%) 사용 | `position_tracker.py` |
| M4 | MAJOR | WS 끊김 watchdog — 5분 무응답 시 강제 decision | `runner.py` |
| M5 | MAJOR | Regime 라벨 불일치 — combined_regime에서 crypto 부분 추출 비교 | `agent_evaluator.py` |
| prev1 | CRITICAL | `await` 누락 — decision.signal publish coroutine 미실행 | `runner.py` |
| prev2 | CRITICAL | SELL 시 untrack() 미호출 — double-sell 위험 | `runner.py` |
| prev3 | CRITICAL | SELL 시 position.exit 미발행 — RL 피드백 루프 끊김 | `runner.py` |
| prev4 | MAJOR | 티커 정규화 — BTC/USDT:USDT → BTC/USDT | `runner.py` |
| prev5 | MAJOR | AdaptiveGate 초기 타이머 0 → 즉시 Claude 호출 방지 | `zscore_gate.py` |

---

## Research References

| Topic | Source | Key Finding |
|-------|--------|-------------|
| RL Meta-Analysis | [arXiv:2512.10913](https://arxiv.org/abs/2512.10913) | Algorithm = 8% of performance, implementation = 31% |
| Adaptive TS | [arXiv:2410.04217](https://arxiv.org/abs/2410.04217) | CADTS: Sharpe 1.76 vs vanilla TS 1.35 |
| IQC 2025 Winner | [UNIST News](https://news.unist.ac.kr/kor/20251014-2/) | All-weather macro > pure return optimization |
| FinRL Contest | [arXiv:2504.02281](https://arxiv.org/abs/2504.02281) | Most RL underperforms simple benchmarks in practice |
| MacroHFT | [arXiv:2406.14537](https://arxiv.org/abs/2406.14537) | Memory-augmented macro RL for HFT — crypto SOTA (KDD 2024) |
| Dynamic TS | [Kwon et al.](https://daheekwon.github.io/pdfs/dynamicTS.pdf) | Exponential filtering for time-varying rewards |
| BTC×M2 Correlation | Coinbase Institutional Research | BTC-Global M2 corr 0.6~0.9, 70-107 day lag |
| Fed Net Liquidity | FRED (WALCL-WTREGEN-RRPONTSYD) | 가장 강력한 BTC 선행 지표 |
| Full Research | `trading-research-2026-v2.md` | Comprehensive signal/RL/OpenClaw analysis |

### 매크로 경제 지표 (추가 구현 필요)

**Tier 1 — 유동성 (가장 중요, BTC = 유동성 자산)**
| 지표 | FRED ID | 설명 | 빈도 |
|------|---------|------|------|
| Fed Balance Sheet | `WALCL` | Fed 총자산 (QE/QT 직접 측정) | 주간 |
| Treasury General Account | `WTREGEN` | 재무부 현금잔고 (유동성 드레인) | 주간 |
| Overnight Reverse Repo | `RRPONTSYD` | RRP 잔고 (유동성 드레인) | 일간 |
| US M2 Money Supply | `M2SL` | 미국 통화량 | 월간 |
| **Net Liquidity** | `WALCL - WTREGEN - RRPONTSYD` | **핵심 선행 지표** | 계산 |

**Tier 2 — 금융 조건**
| 지표 | FRED ID / Source | 설명 |
|------|-----------------|------|
| Fed Funds Rate | `FEDFUNDS` | 기준금리 방향 |
| NFCI | `NFCI` | National Financial Conditions (음수=완화) |
| HY Spread | `BAMLH0A0HYM2` | 하이일드 스프레드 (리스크 선호도) |
| 10Y Real Rate | `REAINTRATREARAT10Y` | BTC 기회비용 |
| DXY | yfinance `DX-Y.NYB` | 달러 인덱스 (BTC -0.4~-0.8 역상관) |

**Tier 3 — 선행 지표**
| 지표 | FRED ID | 설명 |
|------|---------|------|
| Initial Jobless Claims | `ICSA` | 가장 빠른 노동시장 지표 (주간) |
| Consumer Sentiment | `UMCSENT` | 미시건대 소비자심리 |
| Industrial Production | `INDPRO` | 산업생산 지수 |
