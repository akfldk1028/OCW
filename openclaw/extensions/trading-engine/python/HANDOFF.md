# Trading Engine HANDOFF

> 마지막 업데이트: 2026-02-24
> 상태: Phase 1 (Agent-Autonomous Gate) 구현 완료 + 코드 리뷰 11개 버그 수정 완료

---

## 현재 상태 요약

**Claude (Sonnet 4.6) = PRIMARY 트레이더.** 엔진이 시장 데이터를 수집하고, Claude가 모든 매매 결정(BUY/SELL/HOLD + 포지션 사이즈 + 다음 체크 시간)을 내린다. Rule-based pipeline은 Claude 불가 시 fallback용으로만 존재.

### 핵심 아키텍처

```
WS tick → AdaptiveGate (4-layer) → 통과 시에만 Claude 호출 (~10%)
                                    → OHLCV/Derivatives/Regime/TS 수집
                                    → MarketSnapshot 구성
                                    → ClaudeAgent.decide(snapshot)
                                    → 주문 실행 + RL 피드백
```

### 5개 비동기 루프 (runner.py)
1. `market_listener.run()` — Binance WS kline 수신, `market.tick` 이벤트 발행
2. `derivatives_monitor.run()` — Funding/OI/CVD/L-S Ratio 적응형 폴링
3. `position_tracker.run()` — 30s safety(SL/trailing) + 180s agent 재평가
4. `_heartbeat_loop()` — 30s heartbeat 파일 갱신 (Docker healthcheck)
5. `_watchdog_loop()` — WS 끊김 시 5분 후 강제 decision

---

## 파일별 역할 (데이터 흐름 순서)

| 파일 | 역할 | 한줄 설명 |
|------|------|----------|
| `core/market_listener.py` | WS 수신 | Binance WS → `market.tick` 이벤트 (매 틱) |
| `core/zscore_gate.py` | 게이트 | 4-layer: candle_close/timer/z-score/wake_condition |
| `binance/runner.py` | 오케스트레이터 | tick→gate→snapshot→Claude→execute→RL |
| `core/claude_agent.py` | 의사결정 | ClaudeSDK로 Sonnet 호출, JSON 응답 파싱 |
| `core/claude_auth.py` | 인증 | OAuth 토큰 해석, SDK env 구성 |
| `core/online_learner.py` | RL | Regime-Aware Thompson Sampling, Beta(α,β) 업데이트 |
| `core/position_tracker.py` | 포지션 감시 | 2-layer: safety(30s) + agent(180s) |
| `core/agent_evaluator.py` | 포지션 재평가 | 5-signal weighted scoring (fallback) |
| `core/event_bus.py` | 이벤트 | async pub/sub, 500 이벤트 로그 |
| `core/derivatives_monitor.py` | 파생상품 | Funding/OI/CVD/L-S 적응형 폴링 |
| `analysis/regime_detector_crypto.py` | 레짐 | BTC HMM 2-state (low_vol/high_vol) |
| `analysis/macro_regime.py` | 매크로 | FRED 4사분면 (성장×인플레) |
| `brokers/binance.py` | 브로커 | ccxt Binance Spot/Futures 주문 |
| `binance/crypto_config.py` | 설정 | SWING/DAILY 모드, EVENT_CONFIG, gate 설정 |
| `config.py` | 전역 설정 | tickers, risk params, agent weights |

---

## 핵심 데이터 흐름

### BUY 플로우
```
Claude decide → action="BUY", position_pct=0.20
  → _convert_claude_decisions(): ticker 정규화, price 설정, signal 추출
  → broker.execute_decisions(): limit order (maker 0.02%)
  → exec 결과 확인: submitted/dry_run만 진행
  → _entry_prices[ticker] = price
  → position_tracker.track(ticker, entry_price, qty, regime)
  → event_bus.publish("decision.signal", {ticker, signals})  ← RL용
```

### SELL 플로우
```
Claude decide → action="SELL"
  → exec 결과 확인: submitted/dry_run만 진행
  → position_tracker.get_position(ticker)
    → None이면 skip (safety가 먼저 퇴출한 경우)
  → position_tracker.untrack(ticker)
  → event_bus.publish("position.exit", {ticker, pnl, held_hours, agent_signals})
    → online_learner.record_trade() 호출됨 (RL 피드백)
```

### RL 피드백 루프
```
position.exit 이벤트 → _on_exit() 핸들러
  → agent_signals 추출 (Claude reasoning에서 키워드 매칭)
  → regime = crypto_regime + "_" + macro_regime
  → online_learner.record_trade(pnl, signals, regime)
    → 모든 signal에 대해 Beta 업데이트 (global + regime-specific)
    → save to models/online_learner.json
```

### Claude 자기 스케줄링
```
Claude 응답: {next_check_seconds: 3600, wake_conditions: [...], memory_update: "..."}
  → adaptive_gate.update_from_claude()
    → _next_check_at = now + clamped(next_check_seconds)
    → _wake_conditions = parsed WakeCondition list
  → _agent_memory = memory_update[:500]
  → 다음 tick부터 gate가 새 조건으로 평가
```

---

## 수정 이력 (2026-02-24)

### CRITICAL 버그 (수정 완료)
1. `await` 누락: `event_bus.publish("decision.signal")` coroutine 미실행 → TS 시그널 유실
2. SELL 시 `untrack()` 미호출 → double-sell 위험
3. SELL 시 `position.exit` 미발행 → RL 피드백 루프 끊김
4. **position.exit 이중 발행**: safety exit + Claude SELL 동시 발생 시 RL 2회 업데이트 → runner에서 tracker 상태 확인 후 skip
5. **circuit breaker 영구 열림**: 3회 실패 → 복구 불가 → 15분 half-open 패턴 + 강제 re-auth 추가

### MAJOR 버그 (수정 완료)
6. 티커 정규화: `BTC/USDT:USDT` → `BTC/USDT` (Claude Futures 형식 → 데이터 Spot 형식)
7. AdaptiveGate 초기 타이머 0 → `time.time() + max_check_seconds` (즉시 호출 방지)
8. BUY 주문 실패 시 phantom position → exec 결과 확인 후에만 track()
9. Safety trailing stop 2.5% → config trail_pct 12% 통일 + trail_activation 적용
10. WS 끊김 watchdog 추가 (5분 무응답 → 강제 decision)
11. Regime 라벨 불일치 → combined_regime에서 crypto 부분 추출 비교

### 공식문서 기반 수정 (2026-02-24, 2차)
12. **Claude SDK `break` 제거**: `receive_response()`는 `ResultMessage`에서 자동 종료 — `break` 사용 시 asyncio 정리 문제 (공식문서)
13. **Claude SDK `msg.error` 체크 추가**: rate limit 등 API 에러가 `AssistantMessage.error` 필드로 오는데 감지 안 함 (Issue #472)
14. **Claude timeout 60s → 120s**: SDK subprocess spawn + auth + API call에 60초 부족
15. **`analysis/__init__.py` lazy import**: eager import가 gymnasium/transformers/torch 전부 로드 → `__getattr__` 패턴으로 변경
16. **`crypto_regime_detector.detect()` → `asyncio.to_thread()`**: yf.download() blocking → event loop 차단 → WS ping timeout
17. **Futures ticker 정규화**: `_normalize_symbol()`에 `market == "future"` 시 `:USDT` suffix 추가
18. **ccxt demo trading 준비**: `BINANCE_USE_DEMO=true` 환경변수로 `enable_demo_trading(True)` 전환 가능 (현재는 legacy testnet 사용)

---

## Testnet 실행 검증 결과 (2026-02-24)

| 항목 | 결과 |
|------|------|
| WS 연결 (3 tickers) | 성공 |
| AdaptiveGate z-score 트리거 | 성공 (ETH price +2.81, BTC volume +2.68) |
| OHLCV 수집 | 성공 |
| HMM regime detection | 성공 (365일 데이터, 2-state) |
| Claude SDK 호출 | 성공 (Sonnet, ~66초) |
| Claude 결정 | HOLD (캐시 유지), 합리적 reasoning |
| Claude 자기 스케줄링 | 1200-1800초 + 3 wake conditions |
| Fallback pipeline | Claude timeout 시 정상 작동 |
| WS 자동 재연결 | 1초 내 복구 |

**알려진 이슈**:
- `rate_limit_event`: SDK가 파싱 못함 → partial response로 정상 처리 (Issue #472)
- WS disconnect: Claude SDK subprocess 실행 중 SSL 프로토콜 레벨 이슈 (CPython 버그) → 자동 재연결로 복구, 기능 영향 없음

---

## Known Limitations (미수정)

| 항목 | 영향 | 이유 |
|------|------|------|
| TS signal 차별화 불가 | 모든 signal이 동일 confidence | architecture 변경 필요 |
| online_learner load 후 has_enough_data=False | 재시작 후 5거래까지 static weights | _trades 복원 로직 추가 필요 |
| 프로세스 재시작 시 포지션 미복원 | _entry_prices, tracker positions 소실 | broker.get_positions_detail()에서 복원 로직 필요 |
| funding_rate 키 ticker-scoped 아님 | 마지막 ticker 값만 남음 | prefix 추가 필요 |
| Claude에게 safety layer 임계값 미전달 | Claude가 SL 4%/trailing 12% 모름 | MarketSnapshot.to_prompt()에 추가 필요 |
| WS disconnect during Claude call | 매 결정마다 WS 끊김 후 재연결 | CPython SSL bug, 기능 영향 없음 |
| Binance legacy testnet 종료 예정 | demo-api.binance.com으로 이전 필요 | `BINANCE_USE_DEMO=true` + demo API 키 발급 |

---

## 즉시 실행 가능한 명령어

### Testnet 실행 (dry_run, 주문 안 나감)
```bash
cd openclaw/extensions/trading-engine/python
python3 binance/main.py --testnet
```

### Testnet Futures (3x 레버리지)
```bash
python3 binance/main.py --testnet --futures --leverage 3
```

### 테스트
```bash
python3 -m pytest binance/tests/test_zscore_gate.py -v  # 18개 테스트
```

### 백테스트 (Claude SDK 필요)
```bash
cd backtests/backtest_claude
python3 simulator.py  # 2주 시뮬레이션, reports/ 에 결과 저장
```

### .env 필수 변수
```
BINANCE_API_KEY=testnet_key
BINANCE_SECRET_KEY=testnet_secret
BINANCE_PAPER=true
LIVE_TRADING=false
CLAUDE_CONFIG_DIR=D:\DevCache\claude-data  # Windows 전용
```

---

## 다음 TODO (우선순위 순)

1. **Testnet 연속 운영 테스트** — `main.py --testnet` 24시간 이상 돌려서 gate/watchdog/circuit-breaker 검증
2. **실거래 API 키 발급** — Binance 실거래 API 발급 → `.env` 설정 → `LIVE_TRADING=true`
3. **포지션 복원** — 프로세스 재시작 시 `broker.get_positions_detail()`에서 기존 포지션 복원
4. **TS signal 차별화** — Claude 응답 스키마에 per-signal confidence 추가
5. **Phase 2** — Graphiti 메모리 + Redis SemanticCache + trade-reflect skill
