# OpenClaw Trading Engine

## What This Is

Event-driven algorithmic trading engine for **crypto** (Binance Futures/Spot) and **US stocks** (Alpaca).
**Claude (Sonnet) is the PRIMARY decision maker.** The engine collects market data, builds a `MarketSnapshot`,
and Claude makes ALL trading decisions: BUY, SELL, HOLD, position sizing, and exit timing.

**Philosophy**: "OpenClaw이 직접 판단하고, 엔진은 서포트한다."
Rule-based pipeline exists for backtesting only. In production, Claude IS the trader.
Inspired by IQC 2025 winner (김민겸, UNIST) — stability via macro regime awareness beats pure return chasing.

---

## How It Works (Data Flow)

```
 1. MARKET DATA IN                    2. ADAPTIVE GATE                3. CLAUDE DECIDES
 ─────────────────                    ──────────────                  ──────────────
 Binance WebSocket ──┐                AdaptiveGate (4-layer)          ClaudeAgent.decide()
   every tick        │   EventBus       G1: candle close? → pass      Claude Sonnet via SDK
   kline updates     ├──(pub/sub)──→    G2: Claude timer expired?     BUY / SELL / HOLD
                     │                  G3: z-score outlier?   ──→    position sizing
 DerivativesMonitor ─┘                  G4: wake condition met?       confidence + reasoning
   Funding rate                         cooldown: min 60s             next_check_seconds
   Open Interest                        ~90% ticks SKIPPED            wake_conditions
   Taker Delta (CVD)                                                  memory_update
   Long/Short Ratio

 4. SNAPSHOT BUILD                    5. EXECUTION + MONITORING       6. LEARNING
 ──────────────────                   ────────────────────────        ─────────────
 MarketSnapshot                       Broker (Binance)                OnlineLearner
   OHLCV + returns                      Spot or Futures ←── exec      Thompson Sampling
   Derivatives context                  verify success before track   per-regime Beta(α,β)
   Regime (crypto+macro)              PositionTracker                 discount=0.95
   TS posteriors (RL)                   30s: safety (SL/trailing)     auto-save to JSON
   Recent trades (learning)             3min: agent re-eval
   Open positions + portfolio         Watchdog (WS disconnect)
```

**Agent-Autonomous Architecture:** Claude controls its own monitoring schedule.
Every WebSocket tick passes through the AdaptiveGate. ~90% are skipped (no Claude call).
Claude wakes only when:
- 4h candle closes (always)
- Claude's own timer expires (`next_check_seconds`)
- Z-score outlier detected (|z| >= 2.5 on price/volume)
- Claude's wake conditions met (e.g. "wake me if BTC < $80K")
- WS disconnect watchdog fires (5min no ticks → force check)

---

## Key Design Decisions (WHY, not just WHAT)

### 1. Thompson Sampling, not Deep RL
167-paper meta-analysis (arXiv:2512.10913) found algorithm choice explains only 8% of trading performance.
Implementation quality explains 31%. Deep RL (PPO/SAC) needs 950+ episodes (~2 years of data) to converge.
Thompson Sampling works with 10-20 trades. We have ~15-30 trades/month. Decision: keep it simple.

### 2. Regime-Aware Weights
Different agents perform differently in different markets:
- Momentum agents excel in trending markets
- Mean-reversion signals work in ranging markets

Solution: maintain separate Beta distributions **per regime** for each agent.
If a regime has <3 trades, fall back to global (all-regime) posterior.

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

### 5. Two-Layer Position Monitoring
- **Layer 1 (Safety, 30s)**: Hard rules — drawdown >15%, trailing stop hit, time decay
- **Layer 2 (Agent, 3min)**: 5-signal weighted evaluation — regime change, momentum reversal,
  correlation risk, PnL trajectory, time decay

Layer 1 catches catastrophic moves fast. Layer 2 makes intelligent hold/exit/add decisions.

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
│   ├── market_listener.py     # Binance WS: 15m/1h/4h klines (24/7 stream)
│   ├── ohlcv_store.py         # Thread-safe WS-fed OHLCV bar accumulator
│   ├── derivatives_monitor.py # Adaptive poll: funding, OI, CVD, L/S ratio
│   ├── pipeline.py            # n8n-style node pipeline (15 nodes)
│   ├── position_tracker.py    # 2-layer position monitoring (30s/3min)
│   ├── agent_evaluator.py     # 5-signal position scoring
│   ├── online_learner.py      # Regime-Aware Thompson Sampling RL
│   ├── risk_manager.py        # Drawdown guards, exposure limits
│   ├── claude_auth.py         # Claude auth (API key or OAuth)
│   ├── claude_agent.py        # PRIMARY decision maker (Claude Sonnet via SDK)
│   ├── zscore_gate.py         # 4-layer adaptive gate (z-score + wake conditions)
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
│   ├── runner.py              # CryptoRunner: 4 concurrent async loops
│   ├── crypto_config.py       # SWING vs DAILY mode selection
│   ├── tests/                 # Test suite + backtest reports
│   ├── data/                  # OHLCV cache
│   ├── models/                # Trained models + online_learner.json
│   └── logs/                  # Runtime logs
│
└── templates/
    └── chart.html             # TradingView Lightweight Charts v5 dashboard
```

---

## Configuration (config.py)

### Trading Parameters (Tuned 2026-02-23)

| Parameter | Value | Why |
|-----------|-------|-----|
| `dd_trigger` | 0.15 | Single position max 15% drawdown. Was 8%, raised because crypto volatility caused 81% premature exits |
| `trail_activation_pct` | 0.08 | Trailing stop activates after 8% profit. Lets winners run |
| `trail_pct` | 0.12 | Exit at -12% from peak. Balances profit capture vs noise |
| `position_pct` | 0.30 | Max 30% of portfolio per position. Concentrated but not all-in |
| `portfolio_dd_trigger` | 0.12 | Total portfolio -12% → halt new entries |

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
| `kline_interval` | "4h" | Candle timeframe for pipeline |
| `significant_move_pct` | 0.02 | 2% BTC move triggers re-evaluation |
| `min_decision_gap` | 300s | Minimum 5min between pipeline runs |
| `funding_extreme_threshold` | 0.0005 | 0.05%/8h funding = extreme |
| `oi_spike_threshold` | 0.10 | 10% OI change = spike |

### REST vs WebSocket 전략 (2026-02-25)

| 컴포넌트 | REST calls | 언제 |
|----------|-----------|------|
| `connect()` | 1 (fetch_balance) | 시작 시 1회 |
| OHLCV bootstrap | 4 (4 tickers × 1h) | 시작 시 1회 |
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

### Production: Claude Agent Decides Everything
Claude receives the full MarketSnapshot and makes all decisions. No hard-coded entry/exit rules in production.
The system prompt constrains Claude: max 30% per position, respect macro exposure_scale, trust TS posteriors.

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

### Exit Rules (2 layers)
```
Layer 1 — Safety (every 30 seconds):
    EXIT if: drawdown > 4% (hard stop loss — flash crash protection)
    EXIT if: trailing stop hit (activates at +8% profit, trails at -12% from peak)
    NOTE: TP removed from safety — Claude manages take-profit timing

Layer 2 — Claude Re-evaluation (every 3 minutes):
    ClaudeAgent.evaluate_position() → HOLD / EXIT / ADD with confidence
    Fallback (if Claude unavailable): 5-signal weighted scoring:
        Score = 0.25 * regime_change      (low→high vol = -0.8)
               + 0.25 * momentum_reversal (direction flip = -0.7)
               + 0.20 * correlation_risk  (BTC -5% in 1h = -0.9)
               + 0.15 * pnl_trajectory    (profit evaporating = -0.6)
               + 0.15 * time_decay        (stale + low profit = -0.5)
        EXIT if: score < -0.3
        ADD  if: score > +0.4
        HOLD otherwise
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

## Agent 전체 로직 플로우 (한눈에 보기)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT LIFECYCLE (매 틱마다)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SENSE (데이터 수집)                                               │
│  ├── Binance WebSocket → 4h kline 가격/볼륨                          │
│  ├── DerivativesMonitor → Funding/OI/CVD/L-S Ratio (5분 폴링)        │
│  ├── Fear & Greed Index (alternative.me, 1h 캐시)                    │
│  ├── Stablecoin Supply (DeFiLlama, 4h 캐시)                          │
│  ├── Liquidation Cascade (Binance REST, 1h 윈도우)                    │
│  ├── CoinGecko Trending (시장 버즈, 1h 캐시)                          │
│  └── FRED Macro Regime (성장×인플레 4분면, 6h 캐시)                     │
│                                                                     │
│  2. FILTER (AdaptiveGate — 90% 틱 스킵)                              │
│  ├── Gate 1: 4h 캔들 마감? → 항상 통과                                 │
│  ├── Gate 2: Claude 타이머 만료? (next_check_seconds)                 │
│  ├── Gate 3: Z-score 이상치? (|z| >= 2.5 가격/볼륨)                    │
│  ├── Gate 4: Claude wake_conditions 충족?                            │
│  └── 쿨다운: min 60s (Gate 통과해도 스팸 방지)                          │
│                                                                     │
│  3. THINK (Claude Sonnet — PRIMARY 트레이더)                          │
│  ├── MarketSnapshot 구축 (모든 데이터 한 곳에)                          │
│  │   ├── 가격 + 수익률 + 볼륨                                         │
│  │   ├── 파생상품 (funding, CVD, L/S, OI)                             │
│  │   ├── 레짐 (crypto HMM + FRED macro)                              │
│  │   ├── TS 강화학습 사후확률 (레짐별 시그널 신뢰도)                       │
│  │   ├── 최근 10 거래 (학습 맥락)                                      │
│  │   ├── 보유 포지션 + 포트폴리오 상태                                   │
│  │   ├── 센티먼트 (F&G, 뉴스, 청산, 스테이블코인)                        │
│  │   └── 에이전트 메모리 (이전 판단 메모)                                │
│  ├── Claude 호출 → JSON 응답                                         │
│  │   ├── decisions: BUY/SELL/HOLD (티커별)                            │
│  │   ├── position_pct: 포지션 크기 (최대 30%)                          │
│  │   ├── confidence: 신뢰도 (0~1)                                    │
│  │   ├── reasoning: 판단 이유 (상세)                                   │
│  │   ├── next_check_seconds: 다음 깨울 시간 (자기 스케줄링)              │
│  │   ├── wake_conditions: 조건부 깨우기 (예: BTC<61K)                  │
│  │   └── memory_update: 다음 깨울 때 참고할 메모                        │
│  └── Fallback: Claude 불가 시 → 규칙 기반 파이프라인 (백테스트용)          │
│                                                                     │
│  4. ACT (주문 실행)                                                   │
│  ├── Binance 브로커 → 주문 전송 (dry_run or LIVE)                      │
│  ├── 주문 성공 확인 후에만 포지션 추적 시작                               │
│  ├── trades.csv 기록 (타임스탬프/가격/수량/PnL/이유/레짐)                 │
│  └── decisions.jsonl 기록 (Claude 전체 응답 + 맥락)                     │
│                                                                     │
│  5. MONITOR (PositionTracker — 2레이어)                               │
│  ├── Layer 1 Safety (30초): SL -4%, 트레일링스톱 -12%@+8%             │
│  ├── Layer 2 Agent (3분): Claude 재평가 → HOLD/EXIT/ADD               │
│  └── 위험 퇴출 → position.exit 이벤트 발행                              │
│                                                                     │
│  6. LEARN (Thompson Sampling 강화학습)                                 │
│  ├── position.exit 수신 → record_trade()                              │
│  ├── 시그널 정렬도 계산 (agent signal ↔ 수익 방향)                       │
│  ├── Beta 분포 업데이트 (alpha/beta, discount=0.95)                    │
│  ├── 레짐별 별도 Beta (trending에서 momentum 잘 맞으면 ↑)               │
│  ├── < 3 거래 레짐 → 글로벌 fallback                                   │
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
| 포지션 감시 | 30s safety + 3min agent | 로그: `[tracker] Safety check`, `[tracker] Agent eval` |
| EXIT 발생 | `position.exit` 이벤트 | 로그: `[runner] Position exit: BTC/USDT pnl=+X%` |
| TS 업데이트 | `online_learner.record_trade()` | 로그: `[online_rl] Trade #N: BTC/USDT pnl=+X%` |
| Beta 저장 | `online_learner.save()` | 파일: `models/online_learner.json` |
| 가중치 적응 | `sample_weights(regime)` | 로그: TS posteriors in MarketSnapshot |

---

## CryptoRunner: 5 Concurrent Loops (binance/runner.py)

```python
async def run(self):
    await asyncio.gather(
        self._market_listener.run(),      # Loop 1: WebSocket klines
        self._derivatives_monitor.run(),   # Loop 2: Adaptive polling
        self._position_tracker.run(),      # Loop 3: 30s/3min position checks
        self._heartbeat_loop(),            # Loop 4: 30s Docker health
        self._watchdog_loop(),             # Loop 5: WS disconnect fallback
    )
```

**Decision flow (tick → gate → Claude → execute):**
1. MarketListener receives WS tick → publishes `market.tick`
2. CryptoRunner._handle_tick() → _compute_tick_features() → AdaptiveGate.evaluate()
3. Gate decides: skip (90%+) or wake Claude
4. If wake: Build `MarketSnapshot` → **Claude decides** → JSON response
5. Claude response includes `next_check_seconds`, `wake_conditions`, `memory_update`
6. AdaptiveGate.update_from_claude() → sets timer + conditions for next wake
7. Execute BUY/SELL (only if broker confirms success)
8. Track positions in PositionTracker / publish RL events
9. Position close → OnlineLearner.record_trade(regime="trending_goldilocks") → Beta update

**MarketSnapshot contains everything Claude needs:**
- Price action (OHLCV, returns, volume per ticker)
- Derivatives (funding rate, CVD, L/S ratio)
- Regime (crypto HMM + FRED macro 4-quadrant)
- TS posteriors (which signal categories are reliable in current regime)
- Recent trades (last 10 with PnL, for learning context)
- Open positions and portfolio state

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

## Thompson Sampling RL (core/online_learner.py)

### How It Works
Each agent has a Beta(alpha, beta) distribution representing its "quality".
After every closed trade:

1. Check if agent's signal was **aligned** with the outcome
   - Agent said BUY (+0.7) and trade was profitable → aligned ✓
   - Agent said BUY (+0.7) and trade lost money → not aligned ✗
2. Map PnL to reward via sigmoid: `reward = 1 / (1 + exp(-|pnl| * 10))`
3. If aligned: `alpha += reward`, else: `beta += (1 - reward)`
4. Discount old beliefs: `alpha *= 0.95`, `beta *= 0.95`
5. To get weights: sample from each agent's Beta → normalize

### Regime-Aware Extension
```
Regime "trending":  momentum Beta(4.2, 1.5) → mean 0.74 (strong)
Regime "ranging":   momentum Beta(1.8, 3.1) → mean 0.37 (weak)
```
If a regime has <3 trades → fall back to global posterior.

### Persistence
Saved to `models/online_learner.json` after every trade. Loaded on startup.

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

### Architecture: Claude IS the Trader

```
Market Events → Data Collection → MarketSnapshot → ClaudeAgent.decide() → Execute
                                       ↑                                      ↓
                                       └──── Thompson Sampling feedback ←─────┘
```

**Claude (Sonnet) makes ALL trading decisions.** The engine collects data, builds a snapshot,
and asks Claude what to do. Rule-based pipeline is for backtesting only.

- Model: `claude-sonnet-4-6` (Sonnet, not Haiku — quality > speed for trading)
- Auth: `ANTHROPIC_API_KEY` (server/cloud) or Claude Code OAuth (local dev)
- SDK: `claude-agent-sdk` via `ClaudeAgentOptions(env=sdk_env)`
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

### Claude Agent (`core/claude_agent.py`)

| Method | When Called | What It Does |
|--------|------------|-------------|
| `decide(snapshot)` | Every market event | Makes BUY/SELL/HOLD decisions with position sizing |
| `evaluate_position()` | Position re-eval (volatile) | HOLD/EXIT/ADD verdict with confidence |

Claude receives a full `MarketSnapshot` and returns structured JSON:
```json
{
  "decisions": [{"ticker": "BTC/USDT:USDT", "action": "BUY", "position_pct": 0.20, "confidence": 0.85, "reasoning": "..."}],
  "market_assessment": "one sentence overall market view",
  "regime_agreement": true,
  "learning_note": "what I learned from recent trade history",
  "next_check_seconds": 3600,
  "wake_conditions": [{"metric": "btc_price", "operator": "lt", "threshold": 80000, "reason": "support break"}],
  "memory_update": "BTC testing 80K support, watching for breakdown"
}
```

**Circuit breaker** (half-open pattern):
- 3 consecutive failures → OPEN (block all calls)
- After 15 min cooldown → HALF-OPEN (retry with forced re-auth for token refresh)
- Success → CLOSED (normal). Failure → OPEN again (restart 15 min cooldown)

Returns `None` if no auth available → caller uses rule-based fallback.

### MCP Tools (fastapi-mcp)

`server.py` mounts all FastAPI endpoints as MCP tools via `fastapi-mcp`.
User can connect Claude.ai to the engine as a remote MCP server:

```
Claude.ai: "Check current regime and positions"
→ MCP tool call: GET /regime → {"regime": "trending", "confidence": 0.82}
→ MCP tool call: GET /tracker/status → {"positions": [...]}
→ MCP tool call: GET /claude/status → {"agent_available": true, "mode": "primary_decision_maker"}
```

Requires `pip install fastapi-mcp`.

---

## API Endpoints (30+)

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

## Honest Status (2026-02-25)

### 실제로 검증된 것
- 테스트넷 연속 가동 (WARP VPN): BUY 3건, SELL 1건 (BTC +0.35%)
- Claude SDK 호출 성공, 구체적 판단 근거 + 포지션 사이징 작동
- AdaptiveGate z-score 트리거 + timer 만료 작동
- SELL → RL 피드백 루프: 통합테스트 4/4 PASS (safety SL, RL 기록, TS 가중치 변동, 영속성)
- 포지션 재시작 복원: `Restored 1 positions (dry_run)` 성공
- z-score 쿨다운: `z-score blocked for 600s` 작동
- **REST→WS 전면 전환**: 분당 REST weight ~3.3 (Binance 한도 6000의 0.06%)
- OHLCVStore 단위테스트 6/6 PASS
- trades.csv / decisions.jsonl 로깅 작동
- 에이전트 메모리 (500자) 전달 작동

### 검증 안 된 것 (코드는 있지만 충분히 실행되지 않음)
- **수익률**: SELL 1건 (+0.35%) — 통계적으로 무의미. 장기 검증 필요
- **CVD / L-S Ratio**: 테스트넷 미지원 → N/A (파생상품 시그널 5/7만 작동)
- **FRED Macro**: API 키 없음 → regime="unknown"
- **Basis Spread / ETF Flow**: 코드 추가됨, 실전 데이터 미확인
- **Skills / Hooks / Graphiti**: 설계만, 연결 안 됨
- **장기 안정성**: 1시간 이상 연속 운용 미확인 (메모리/WS 안정성)

### 핵심 한계
- **Agent가 "학습"하지 않음**: 매번 새로 태어남. 500자 메모 외에 이전 맥락 없음
- **인터넷/논문/새 전략 탐색 안 함**: 받은 MarketSnapshot만 보고 판단
- **거래 빈도 낮음**: 4h 캔들 기준 월 2-3회 (단타/스윙에 부적합)
- **OpenClaw 기능 미활용**: Skills, Hooks, Memory(Graphiti) 전부 미연결
- **로컬 PC IP ban 문제**: Binance IP ban 에스컬레이션으로 로컬 테스트 제한적
- **WARP VPN 부적합**: Cloudflare WARP shared IP pool → 다른 유저 트래픽과 합산, Binance rate limit/ban

---

## What's Done vs What's Next

### Fully Implemented
- [x] Event-driven runner (5 concurrent loops incl. watchdog)
- [x] AdaptiveGate: 4-layer gatekeeper (candle/timer/z-score/wake-conditions)
- [x] Claude self-scheduling (next_check_seconds + wake_conditions + memory)
- [x] Regime-Aware Thompson Sampling RL (per-regime Beta distributions)
- [x] Derivatives signals (funding, OI, CVD, L/S ratio)
- [x] FRED macro regime detector (4-quadrant, all-weather)
- [x] 2-layer position tracking (safety 30s + agent 3min)
- [x] 5-signal agent evaluator (regime/momentum/correlation/PnL/time)
- [x] Pipeline architecture (15 nodes, n8n-style)
- [x] Multi-broker support (Binance, Alpaca, KIS)
- [x] TradingView dashboard (/chart)
- [x] fastapi-mcp integration (Claude as MCP client)
- [x] Claude Agent as primary decision maker (claude_auth + claude_agent via claude-agent-sdk)
- [x] Circuit breaker with half-open recovery (15 min cooldown + forced re-auth)
- [x] Parameter tuning validated on 1-year backtest
- [x] Claude Agent 2-week backtest (alpha +4.2% vs BTC)
- [x] **OHLCVStore**: WS-fed OHLCV bar accumulator, REST bootstrap 1회 후 WS only
- [x] **REST→WS 전면 전환**: decision cycle 0 REST, price_fetcher WS→OHLCV (REST fallback 삭제)
- [x] **IP ban detection**: broker `_ip_ban_until` + `_detect_ip_ban()` → 모든 REST 자동 차단
- [x] **SELL tracker qty**: SELL 결정에 position tracker qty 포함 → dry_run REST 제거
- [x] **Position restore**: `_restore_positions()` dry_run 모드 지원
- [x] **Docker**: Dockerfile + .dockerignore + healthcheck

### Not Yet Implemented (P1-P2)
- [x] Liquidation cascade detection (REST polling in DerivativesMonitor, $5M/1h threshold)
- [x] Fear & Greed index filter (alternative.me API, cached 1h)
- [x] Stablecoin supply as buying power proxy (DeFiLlama API, cached 4h)
- [x] CoinGecko trending coins as market buzz (cached 1h)
- [x] PAXG/USDT (금 RWA 토큰) 스팟 추가
- [x] Trade logging (trades.csv + decisions.jsonl)
- [x] Basis spread signal (perp-spot premium, annualized, `_check_basis_spread()`)
- [x] BTC ETF daily flow as institutional sentiment (CoinGlass, `_fetch_etf_flow()`)
- [x] TS 시그널 분화 — Claude `signal_weights` 필드 + keyword fallback
- [x] PnL 분석 도구 (`binance/tests/analyze_trades.py`)
- [ ] Exchange net flow / whale tracking (CryptoQuant, $29/mo)
- [ ] XAU/XAG 선물 모드 추가 (선물 안정화 후)
- [ ] End-to-end testnet soak test (1주일 연속 운용)

### Known Issues / Tech Debt
- **TS signal differentiation**: All signal categories get the same confidence value from `_extract_signals_from_reasoning()` → TS learns "did Claude profit?" rather than "which signal type is reliable". Architecture change needed for true per-signal learning.
- **online_learner load**: After process restart, `_trades` list is not restored → `has_enough_data=False` until 5 new trades. Beta distributions are restored correctly, but static weights are used initially.
- **Process restart**: ~~Lost on restart~~ Fixed — `runner_state.json` persists entry_prices + agent_memory, `_restore_positions()` recovers on startup.
- `lgbm_signal.py` is dead code (AUC 0.54). Kept for reference, never loaded.
- Equity cron scheduler in server.py still uses fixed timers (not event-driven like crypto).
- `FRED_API_KEY` is optional — without it, macro regime returns UNKNOWN (neutral exposure).
- `fastapi-mcp` is optional — server works without it, just no MCP tool exposure.
- `claude-agent-sdk` is optional — without it, Claude agent falls back to rule-based pipeline (backtesting mode).

### Bug Fixes (2026-02-24~25 Code Review, 총 40+건)

**2026-02-25 (이 세션)**
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
| Full Research | `trading-research-2026-v2.md` | Comprehensive signal/RL/OpenClaw analysis |
