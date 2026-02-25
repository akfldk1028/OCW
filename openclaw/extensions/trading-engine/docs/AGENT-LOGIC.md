# Agent 전체 로직 흐름

> 이 문서는 트레이딩 엔진의 데이터 흐름, 의사결정 과정, 학습 루프를 코드 레벨에서 설명한다.

---

## 1. 시스템 구조 (한눈에 보기)

```
                    ┌─────────────────────────────────────┐
                    │         Binance WebSocket            │
                    │  15m/1h/4h kline (3 tickers × 3 TF) │
                    └──────────────┬──────────────────────┘
                                   │ every tick (~250ms)
                                   ▼
                    ┌──────────────────────────────────────┐
                    │      MarketListener (ws_stream.py)    │
                    │  parse kline → publish "market.tick"  │
                    └──────────────┬───────────────────────┘
                                   │ EventBus pub/sub
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   CryptoRunner._handle_tick()                    │
│                                                                  │
│  1. MultiTFAggregator.update() — 캔들 업데이트                    │
│  2. _compute_tick_features() — 가격변화%, 볼륨, 파생상품          │
│  3. AdaptiveGate.evaluate(features) — 4-layer 필터               │
│     ├── Gate 1: 캔들 마감? → 항상 통과                            │
│     ├── Gate 2: Claude 타이머 만료? (next_check_seconds)         │
│     ├── Gate 3: Z-score 이상치? (|z| >= 2.0)                    │
│     ├── Gate 4: Claude wake_conditions 충족?                     │
│     └── 쿨다운: min 120s                                        │
│  4. 통과 시 → _run_decision() 호출                               │
│     불통 시 → 다음 틱 대기 (90%+ 여기서 종료)                     │
└──────────────────┬───────────────────────────────────────────────┘
                   │ Gate 통과 (~10%)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    _run_decision(trigger)                         │
│                                                                  │
│  1. _fetch_ohlcv_df() — Binance REST, 4h 봉 180일분              │
│  2. broker.get_positions_detail() — 현재 포지션                   │
│  3. broker.get_portfolio_value() — 포트폴리오 가치                │
│  4. _build_snapshot() — MarketSnapshot 구축 (아래 상세)           │
│  5. ClaudeAgent.decide(snapshot) — Claude Sonnet 호출            │
│     ├── 성공 → JSON 응답 (decisions + scheduling)                │
│     └── 실패 → _fallback_pipeline() (rule-based)                │
│  6. _convert_claude_decisions() — 티커 정규화, 가격 설정          │
│  7. broker.execute_decisions() — 주문 실행                       │
│  8. 결과 처리 (BUY → track, SELL → untrack + RL)                 │
│  9. AdaptiveGate.update_from_claude() — 다음 스케줄              │
│ 10. _save_state() — runner_state.json 저장                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. MarketSnapshot 구성 (Claude가 보는 데이터)

`_build_snapshot()` (runner.py:634)에서 구축. Claude는 이 데이터만 보고 결정한다.

| 필드 | 소스 | 설명 |
|------|------|------|
| `ticker_prices` | OHLCV close | BTC, ETH, SOL, PAXG 현재가 |
| `ticker_returns_4h` | OHLCV | 직전 4h 수익률 |
| `ticker_returns_24h` | OHLCV | 24h 수익률 (6봉) |
| `ticker_volumes` | OHLCV volume | 직전 봉 거래량 |
| `btc_price`, `btc_change_1h/24h` | OHLCV + WS cache | BTC 기준 가격 |
| `funding_rates` | DerivativesMonitor | 바이낸스 8h 펀딩레이트 |
| `open_interest` | DerivativesMonitor | OI 변화율 |
| `taker_delta` | DerivativesMonitor | CVD buy/sell ratio |
| `long_short_ratio` | DerivativesMonitor | 탑 트레이더 롱/숏 비율 |
| `liquidation_context` | DerivativesMonitor | 1h 청산 총액, cascade 여부 |
| `basis_spread` | DerivativesMonitor | 선물-현물 프리미엄 (연율화) |
| `stablecoin_supply` | DeFiLlama (4h cache) | USDT+USDC 총 공급량 |
| `fear_greed_index` | alternative.me (1h) | 공포-탐욕 지수 (0-100) |
| `news_summary` | CoinGecko trending (1h) | 트렌딩 코인 목록 |
| `etf_flow` | CoinGlass (4h) | BTC ETF 일간 유입액 |
| `crypto_regime` | HMM 2-3 state | low_vol / medium_vol / high_vol |
| `macro_regime` | FRED 4사분면 | GOLDILOCKS / REFLATION / STAGFLATION / DEFLATION |
| `combined_regime` | crypto + macro | "low_volatility_goldilocks" 형태 |
| `exposure_scale` | 매크로 레짐 | 1.2(골디) / 1.0(리플) / 0.6(스태그) / 0.9(디플) |
| `ts_posteriors` | OnlineLearner | 레짐별 에이전트 Beta 분포 (신뢰도) |
| `recent_trades` | OnlineLearner | 최근 10거래 (PnL, 레짐, 시그널) |
| `open_positions` | PositionTracker | 보유 포지션 (가격, PnL, 보유시간) |
| `portfolio_value`, `cash` | Broker | 총 가치, 현금 |
| `gate_wake_reasons` | AdaptiveGate | 왜 깨어났는지 (z-score, timer 등) |
| `agent_memory` | runner_state.json | 이전 Claude 메모 (500자) |
| `historical_insights` | Neo4j TradingMemory | 레짐별 과거 인사이트 |
| `multi_tf_summary` | MultiTFAggregator | 15m/1h/4h 각 TF의 트렌드/BB |

---

## 3. Claude 의사결정 과정

### 3.1 입력 (System Prompt + MarketSnapshot)

`claude_agent.py` SYSTEM_PROMPT:
- "You are an autonomous crypto trader"
- position_pct: 0.05~0.20 범위
- TS posteriors 신뢰하라
- macro exposure_scale 준수하라
- 반드시 JSON으로 응답하라

### 3.2 출력 (JSON)

```json
{
  "decisions": [
    {
      "ticker": "BTC/USDT:USDT",
      "action": "BUY",
      "position_pct": 0.15,
      "confidence": 0.82,
      "reasoning": "Low vol regime, funding negative (contrarian bullish)...",
      "signal_weights": {
        "momentum": 0.7, "funding_rate": -0.3, "macro": 0.5
      }
    }
  ],
  "market_assessment": "Cautious bullish, funding negative supports long",
  "regime_agreement": true,
  "learning_note": "Previous SOL loss due to high vol entry — avoid",
  "next_check_seconds": 3600,
  "wake_conditions": [
    {"metric": "btc_price", "operator": "lt", "threshold": 60000, "reason": "support break"}
  ],
  "memory_update": "BTC range-bound 62-65K. Funding negative = longs unwinding."
}
```

### 3.3 응답 처리 흐름

```
Claude JSON → _convert_claude_decisions()
  ├── ticker 정규화: BTC/USDT:USDT → BTC/USDT
  ├── price: 현재가 설정
  ├── signal_weights → agent_signals (RL용)
  │   └── 없으면 reasoning 키워드 매칭으로 추출
  └── position_pct: min(응답값, 0.20) 으로 cap

decisions → broker.execute_decisions()
  ├── BUY: limit order (maker fee 0.02%)
  │   ├── 성공 확인 → _entry_prices 기록
  │   ├── position_tracker.track() → 2-layer 감시 시작
  │   └── event_bus.publish("decision.signal") → RL용 시그널 기록
  └── SELL: market/limit order
      ├── position_tracker.get_position() → None이면 skip (safety 선행)
      ├── position_tracker.untrack()
      ├── event_bus.publish("position.exit") → RL 피드백
      └── trades.csv + decisions.jsonl 기록
```

---

## 4. 포지션 감시 (2-Layer)

### Layer 1: Safety (매 30초)

`position_tracker.py _check_safety_rules()`

```
현재 PnL 계산 (current_price vs entry_price)
  ├── PnL <= -10% → STOP LOSS (무조건 퇴출)
  └── trailing_high 갱신 후:
      ├── peak_pnl >= 8% (trail_activation) 이면:
      │   └── current_price <= trailing_high * (1 - 12%) → TRAILING STOP
      └── 미활성 → pass
```

### Layer 2: Agent 재평가 (매 3분)

`position_tracker.py _agent_check()` → `agent_evaluator.py`

Claude 불가 시 fallback:
```
score = 0.25 * regime_change_signal      (low→high vol = -0.8)
      + 0.25 * momentum_reversal_signal  (방향 전환 = -0.7)
      + 0.20 * correlation_risk_signal   (BTC -5%/1h = -0.9)
      + 0.15 * pnl_trajectory_signal     (수익 증발 = -0.6)
      + 0.15 * time_decay_signal         (장기보유+저수익 = -0.5)

score < -0.3 → EXIT
score > +0.4 → ADD
else → HOLD
```

---

## 5. 강화학습 피드백 루프

### Thompson Sampling (online_learner.py)

```
position.exit 이벤트 수신
  → _on_exit() 핸들러 (runner.py:355)
  → agent_signals 추출 (Claude signal_weights 또는 키워드 매칭)
  → combined_regime = "low_volatility_goldilocks"
  → online_learner.record_trade(pnl, signals, regime)
      │
      ├── 각 signal에 대해:
      │   ├── signal > 0 이고 profitable → aligned ✓
      │   ├── signal < 0 이고 loss → aligned ✓ (숏 시그널이 맞음)
      │   └── else → not aligned
      │
      ├── effective_pnl = |pnl| if aligned, -|pnl| if not
      ├── reward = sigmoid(effective_pnl * 10) → (0, 1)
      │
      ├── Global Beta 업데이트:
      │   alpha *= 0.95 (discount)
      │   beta *= 0.95
      │   alpha += reward (if aligned)
      │   beta += (1-reward) (if not aligned)
      │
      ├── Regime-specific Beta 업데이트 (동일)
      └── save to models/online_learner.json

다음 결정 시:
  → sample_weights(regime)
  → 레짐에 3+ 거래 있으면: 레짐별 Beta에서 샘플
  → 아니면: Global Beta에서 샘플
  → 정규화 → MarketSnapshot.ts_posteriors에 포함
  → Claude가 "신뢰도" 참고하여 의사결정
```

### Sliding Window (신규, ADTS 논문 기반)
- `_trades` 리스트를 최대 50건으로 제한
- 50건 초과 시 가장 오래된 거래 삭제
- discount=0.95 + window=50 조합으로 최근 성과에 집중

---

## 6. 자기 스케줄링 (Agent-Autonomous)

```
Claude 응답:
  next_check_seconds: 3600    (1시간 후 다시 깨워라)
  wake_conditions: [           (이 조건 중 하나라도 만족하면 즉시 깨워라)
    {metric: "btc_price", operator: "lt", threshold: 60000},
    {metric: "funding_rate", operator: "abs_gt", threshold: 0.001}
  ]
  memory_update: "BTC range-bound..."

  → AdaptiveGate.update_from_claude()
    ├── _next_check_at = now + clamp(next_check_seconds, 60, 3600)
    ├── _wake_conditions = [WakeCondition(...), ...]
    └── 다음 틱부터 새 조건으로 평가

매 틱마다:
  Gate 2: time.time() >= _next_check_at ? → wake
  Gate 4: any(cond.evaluate(features) for cond in _wake_conditions) ? → wake
```

---

## 7. 5개 비동기 루프 (runner.py)

```python
await asyncio.gather(
    self._market_listener.run(),      # Loop 1: WS kline 수신 → market.tick
    self._derivatives_monitor.run(),   # Loop 2: 5분 적응형 폴링 (funding/OI/CVD/LS)
    self.position_tracker.run(),       # Loop 3: 30s safety + 3min agent
    self._heartbeat_loop(),            # Loop 4: 30s heartbeat (Docker health)
    self._watchdog_loop(),             # Loop 5: 5min WS 끊김 감지 → 강제 decision
)
```

### 각 루프 역할:

| 루프 | 주기 | 역할 |
|------|------|------|
| MarketListener | 매 틱 (~250ms) | WS kline → `market.tick` 이벤트 |
| DerivativesMonitor | 5min (극단시 1min) | funding/OI/CVD/LS/liquidation/basis/stablecoin |
| PositionTracker | 30s / 3min | safety SL/trail + agent 재평가 |
| Heartbeat | 30s | 파일 touch (Docker healthcheck) |
| Watchdog | 5min | WS 끊김 시 강제 _run_decision("watchdog") |

---

## 8. 파생상품 시그널 계산 (DerivativesMonitor)

### 적응형 폴링

```
기본 주기: 5분
극단 감지 시 (funding > 0.05% 또는 OI > 10% 변화):
  → 주기 1분으로 가속
  → 15분 후 자동 복원
```

### 시그널 가중합 (DerivativesSignalNode in pipeline.py)

```
signal = 0.0

# Funding Rate (0.35) — 비선형 tanh 변환
normalized = funding_rate / 0.0005  # >1 = extreme
signal -= tanh(normalized) * 0.35

# CVD Taker Delta (0.30) — buy/sell ratio
signal += tanh((ratio - 1.0) * 2.0) * 0.30

# Long/Short Ratio (0.25) — contrarian, z-score
if |z| > 1.0:
  signal -= tanh(z * 0.15) * 0.25

# Multi-ticker 평균화 (합산 아님)
signal /= n_tickers

# Clamp [-1, 1]
```

---

## 9. HMM 레짐 감지 (regime_detector_crypto.py)

### BIC 자동 state 수 선택

```
for n in [2, 3, 4]:
  model = GaussianHMM(n_components=n)
  model.fit(X)  # X = [returns, realised_vol, vol_of_vol]
  bic = -2 * log_likelihood + k * log(N)
  → 최소 BIC 선택

State 매핑 (vol 기준 정렬):
  0 = low_volatility   → exposure 1.0
  1 = medium_volatility → exposure 0.8
  2 = high_volatility   → exposure 0.6
```

---

## 10. Circuit Breaker (claude_agent.py)

```
정상 → 3회 연속 실패 → OPEN (모든 호출 차단)
  → 15분 대기 → HALF-OPEN (1회 시도 + 강제 re-auth)
    ├── 성공 → CLOSED (정상)
    └── 실패 → OPEN (15분 재대기)
```

---

## 11. 데이터 영속성

| 파일 | 내용 | 저장 시점 |
|------|------|----------|
| `runner_state.json` | entry_prices, entry_times, agent_memory | 매 거래 후 |
| `models/online_learner.json` | Beta 분포, 거래 이력, 레짐 카운트 | 매 거래 클로즈 |
| `logs/trades.csv` | 모든 BUY/SELL 기록 | 매 거래 |
| `logs/decisions.jsonl` | Claude 전체 응답 + context | 매 결정 |
| Neo4j (Trade/Insight 노드) | 거래 기록 + 레짐 인사이트 | 매 거래 클로즈 |

---

## 12. Fallback 경로

```
Claude 호출 → 실패 (timeout / JSON 파싱 / circuit breaker open)
  → _fallback_pipeline()
    1. OHLCV → RegimeBlendDetectNode (HMM)
    2. derivatives_context → DerivativesSignalNode (가중합)
    3. regime + prices → RegimeBlendSignalNode (RSI/momentum/BB)
    4. positions → RegimeBlendExitNode (DD/trail)
    5. signals → RegimeBlendEntryNode (position sizing)
    → decisions = [{action, ticker, price, qty}]
```
