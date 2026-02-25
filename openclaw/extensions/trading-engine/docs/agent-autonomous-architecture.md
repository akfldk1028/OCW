# Agent-Autonomous Trading Architecture

> "OpenClaw이 직접 판단하고, 엔진은 서포트한다."

## 문제: 지금은 에이전트가 아니라 cron이다

현재 구조:
```
WebSocket → 4h 캔들 마감 → Claude 호출 → 결정
             ↑ 하드코딩된 임계값 (2% BTC 변동, OI 10% 스파이크)
```

이건 `cron + API call`이지 **에이전트**가 아님.
- "언제 판단할지"를 **규칙이 결정** (4h 타이머, 하드코딩된 %threshold)
- Claude는 "호출당하는" 수동적 존재
- OpenClaw의 도구 사용, 메모리, 훅, MCP 등 **전혀 활용 안 함**

---

## 목표: 에이전트가 스스로 판단하는 구조

```
WebSocket (24/7 실시간)
    │
    ▼
[로컬 이상치 탐지] ──→ "평소랑 다른가?"
    │                        │
    │  아님 ──→ [에이전트가 설정한 Wake Conditions]
    │                        │
    │  해당없음 ──→ [에이전트가 정한 next_check 타이머]
    │                        │
    │  아직 ──→ SKIP (Claude 호출 없음)
    │
    ▼ (뭔가 변했을 때만)
[Semantic Cache] ──→ "이 상황 전에 본 적 있나?"
    │
    │  있음 ──→ 캐시된 결정 재사용
    │
    ▼ (진짜 새로운 상황)
Claude Sonnet 호출
    │
    ▼ 리턴:
    ├── 매매 결정 (BUY/SELL/HOLD + 포지션 사이징)
    ├── wake_conditions (에이전트가 직접 설정)
    │     "BTC 94000 깨지면 깨워줘"
    │     "펀딩 0.05% 넘으면 깨워줘"
    ├── next_check_seconds (에이전트가 직접 결정)
    │     조용한 시장: 1800초 (30분)
    │     변동성 높은 시장: 60초 (1분)
    │     포지션 진입 직후: 120초 (2분)
    ├── reasoning (판단 근거)
    └── learning_note (메모리에 저장)
```

**핵심**: Claude가 "언제 다시 봐야 하는지"를 **스스로 결정**한다.

---

## 4-Layer Gatekeeper 상세

### Layer 1: ZScoreGate (매 틱, 마이크로초, 로컬)

```python
from collections import deque
import math

class ZScoreGate:
    """실시간 스트리밍 이상치 탐지. 매 틱마다 실행, LLM 비용 0."""

    def __init__(self, window=100, threshold=2.5):
        self.window = window
        self.threshold = threshold
        self.buffers: dict[str, deque] = {}

    def is_anomalous(self, features: dict) -> tuple[bool, dict]:
        scores = {}
        triggered = False
        for key, value in features.items():
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.window)
            buf = self.buffers[key]
            if len(buf) >= 20:
                mean = sum(buf) / len(buf)
                std = math.sqrt(sum((x - mean)**2 for x in buf) / len(buf))
                if std > 0:
                    z = abs(value - mean) / std
                    scores[key] = z
                    if z > self.threshold:
                        triggered = True
            buf.append(value)
        return triggered, scores
```

감시 대상 features:
- `price_change_pct`: 가격 변동률
- `volume_zscore`: 거래량 이상
- `funding_rate`: 펀딩 레이트
- `oi_change_pct`: 미결제약정 변동
- `liquidation_volume`: 청산량

논문 근거: Gatekeeper (arXiv:2502.19335) — 40% LLM 호출 감소, 품질 유지

**고도화**: River 라이브러리의 `HalfSpaceTrees` (스트리밍 Isolation Forest)
```python
from river import anomaly, preprocessing
detector = anomaly.HalfSpaceTrees(n_trees=10, height=6, window_size=100)
# 매 틱마다 score_one() + learn_one() — 자동으로 concept drift 적응
```

### Layer 2: Agent Wake Conditions (매 틱, 마이크로초, 로컬)

Claude가 마지막 판단 시 설정한 조건들을 체크:

```python
@dataclass
class WakeCondition:
    metric: str        # "price", "funding_rate", "volume_zscore", "rsi"
    operator: str      # ">", "<", "crosses_above", "crosses_below"
    threshold: float
    reason: str        # "지지선 이탈 감시", "펀딩 극단값 주의"

    def evaluate(self, features: dict) -> bool:
        value = features.get(self.metric)
        if value is None:
            return False
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        return False
```

**에이전트가 스스로 설정**하므로 하드코딩 없음.

### Layer 3: Agent-Emitted Timer (비용 0)

Claude의 응답에 `next_check_seconds` 포함:

```
시스템 프롬프트 지침:
- 조용한 시장, 포지션 없음: 900-1800초 (15-30분)
- 추세장, 포지션 있음: 180-300초 (3-5분)
- 고변동성, 스탑 근접: 60-120초 (1-2분)
- 포지션 진입 직후: 120초 (2분, 진입 확인)
- 극단적 이벤트: 30초
```

### Layer 4: Semantic Cache (밀리초)

```python
# Redis SemanticCache — TTL + 코사인 유사도
from redisvl.extensions.cache.llm import SemanticCache

cache = SemanticCache(
    name="trading_decisions",
    distance_threshold=0.1,  # 낮을수록 엄격
)
cache.set_ttl(300)  # 5분 후 만료

# 또는 간단한 State Fingerprint 방식:
# regime + volatility_bucket + momentum_bucket + funding_bucket
# → 이산화된 상태 벡터 간 거리로 캐시 히트 판단
```

논문 근거:
- GPT Semantic Cache (arXiv:2411.05276) — 200-300x 속도 향상
- Domain-Specific Embeddings (arXiv:2504.02268) — 도메인 특화 임베딩이 캐시 히트율 향상
- Plan Reuse (arXiv:2512.21309) — 반복 상황에서 밀리초 응답

### 예상 효과

| 현재 | Layer 1-4 적용 후 |
|------|-------------------|
| 하루 6회 고정 (4h) | 하루 5-15회 적응적 |
| 매번 Sonnet 호출 | 80-90% 로컬 처리 |
| 시장 무시 4시간 | 매 틱 감시, 이상 시 즉시 반응 |
| 하드코딩 임계값 | 에이전트가 조건 설정 |

---

## OpenClaw 기능 최대 활용

OpenClaw은 단순 API 래퍼가 아님. **컴퓨터를 지지고 볶는 프로젝트**. 활용해야 할 기능:

### 1. Skills (이미 존재)

| Skill | 파일 | 용도 |
|-------|------|------|
| `trade-scan` | `skills/trade-scan.md` | 자율 매매 판단 (레짐→스캔→랭킹→결정) |
| `trade-execute` | `skills/trade-execute.md` | 주문 실행 + TP/SL 설정 |
| `trade-status` | `skills/trade-status.md` | 포트폴리오 상태 조회 |
| `trade-backtest` | `skills/trade-backtest.md` | 백테스트 실행 |

**추가해야 할 Skill**:
- `trade-reflect`: 최근 N 거래 복기 → 프롬프트 자기 최적화 (ATLAS 패턴)
- `trade-watch`: Wake Conditions 설정 → "이 조건 감시해줘"
- `market-brief`: 현재 시장 요약 (레짐, 주요 지표, 포지션)

### 2. Hooks (이미 존재)

| Hook | 이벤트 | 용도 |
|------|--------|------|
| `trade-alert` | `trade:signal` | 시그널 감지 시 텔레그램/디스코드 알림 |
| `risk-guardian` | `trade:executed`, `cron:tick` | 실시간 SL/TP/일일 손실 한도 감시 |

**추가해야 할 Hook**:
- `agent-wakeup`: Layer 1-3 트리거 시 에이전트 세션 시작
- `trade-journal`: 거래 실행 후 Graphiti + daily memory에 자동 기록
- `regime-shift`: 레짐 변경 감지 시 에이전트에게 알림

### 3. Memory (Graphiti + MEMORY.md)

```
Graphiti (Neo4j 기반 그래프 메모리):
├── group: "trading-portfolio"  ← 거래 이력, 포지션
├── group: "trading-patterns"   ← 학습된 패턴 (이 레짐에서 뭐가 통하는지)
└── group: "trading-research"   ← 리서치 인사이트

MEMORY.md (세션 간 영구 메모리):
├── 트레이딩 규칙 (dd_trigger, position_pct 등)
├── 학습된 교훈 ("trending_goldilocks에서 펀딩 높을 때 롱 피하라")
└── 현재 전략 상태 (에이전트의 자기 평가)

memory/YYYY-MM-DD.md (일일 로그):
├── 매 판단마다 자동 기록
├── 에이전트의 reasoning + confidence
└── 거래 결과 (entry/exit/PnL)
```

**FinMem 논문 패턴** (arXiv:2311.13743):
- Working Memory = 현재 MarketSnapshot
- Episodic Memory = 최근 거래 이력 (Graphiti trading-portfolio)
- Semantic Memory = 학습된 패턴 (Graphiti trading-patterns)
- Memory Scoring = `recency * relevancy * importance` → 유사 상황 검색

### 4. MCP Tools

이미 설정된 것:
- **Graphiti**: 메모리 저장/검색
- **arxiv-mcp**: 논문 검색 (에이전트가 스스로 리서치)

추가 가능:
- **Alpaca MCP**: 브로커 직접 호출 (에이전트가 직접 주문)
- **Alpha Vantage**: 실시간 데이터
- **Financial Datasets**: 뉴스 + 펀더멘털
- **Maverick MCP**: 29개 기술 분석 도구 + VectorBT 백테스트

### 5. Tool Use (에이전트가 직접 실행)

OpenClaw 에이전트는 **도구를 직접 사용**할 수 있음:
- `bash`: Python 스크립트 실행, API 호출
- `read/write`: 설정 파일, 로그 읽기/쓰기
- `message`: 텔레그램/디스코드로 알림 전송
- MCP tools: Graphiti 메모리, 브로커 API, 데이터 API

즉 에이전트가:
```
1. Graphiti에서 최근 거래 패턴 검색
2. Python으로 현재 시장 데이터 수집
3. 자체 판단으로 매매 결정
4. 브로커 API로 직접 주문 실행
5. 결과를 메모리에 저장
6. 텔레그램으로 알림 전송
7. Wake Conditions 설정하고 잠들기
```

이 모든 걸 **하나의 에이전트 세션** 안에서 자율적으로 수행.

---

## 새로운 Decision Response 스키마

```json
{
  "decisions": [
    {
      "ticker": "BTC/USDT:USDT",
      "action": "BUY",
      "position_pct": 0.20,
      "confidence": 0.85,
      "reasoning": "trending_goldilocks 레짐에서 RSI 저점 반등..."
    }
  ],
  "market_assessment": "BTC 94K 지지선 테스트 중, 변동성 확대 조짐",
  "regime_agreement": true,
  "learning_note": "최근 3거래 중 2건이 trail stop에 걸림 — 활성화 기준 상향 고려",

  "next_check_seconds": 300,
  "wake_conditions": [
    {"metric": "price", "operator": "<", "threshold": 93500, "reason": "지지선 이탈"},
    {"metric": "funding_rate", "operator": ">", "threshold": 0.05, "reason": "극단적 롱 포지셔닝"},
    {"metric": "volume_zscore", "operator": ">", "threshold": 3.0, "reason": "거래량 폭증"}
  ],
  "memory_update": "BTC 94K 지지 확인, 0.2 포지션 진입. 다음 확인 5분 후."
}
```

---

## Adaptive Agent Loop 구현

```python
class AdaptiveAgentLoop:
    """에이전트 중심 적응형 루프.

    에이전트가 판단 주기, 감시 조건을 모두 스스로 결정.
    로컬 게이트키퍼가 매 틱 필터링 → 진짜 필요할 때만 Claude 호출.
    """

    def __init__(self, agent, gate, cache):
        self.agent = agent          # ClaudeAgent
        self.gate = gate            # ZScoreGate
        self.cache = cache          # SemanticCache (or None)
        self.wake_conditions = []   # 에이전트가 설정한 조건
        self.next_check_at = 0      # 에이전트가 설정한 타이머
        self.last_decision = None

    async def on_market_tick(self, features: dict, timestamp: float):
        """매 틱마다 호출. 대부분 마이크로초 안에 리턴."""

        # Layer 1: 이상치 탐지
        is_anomalous, scores = self.gate.is_anomalous(features)

        # Layer 2: 에이전트 Wake Conditions
        wake_triggered = any(
            wc.evaluate(features) for wc in self.wake_conditions
        )

        # Layer 3: 에이전트 타이머
        time_to_check = timestamp >= self.next_check_at

        # 3개 Layer 모두 통과 못하면 SKIP
        if not (is_anomalous or wake_triggered or time_to_check):
            return  # Claude 호출 없음

        # Layer 4: Semantic Cache
        if self.cache and not is_anomalous:
            cached = self.cache.check(self._state_key(features))
            if cached:
                return  # 캐시된 결정 재사용

        # 여기까지 온 것만 Claude 호출
        snapshot = await self._build_snapshot(features)
        decision = await self.agent.decide(snapshot)

        if decision:
            # 에이전트의 자기 스케줄 반영
            self.next_check_at = timestamp + decision.get("next_check_seconds", 300)
            self.wake_conditions = self._parse_wake_conditions(decision)

            # 캐시 저장
            if self.cache:
                self.cache.store(self._state_key(features), decision)

            # 메모리 저장 (Graphiti)
            if decision.get("memory_update"):
                await self._save_to_memory(decision["memory_update"])

            # 매매 실행
            await self._execute(decision)
```

---

## 논문 근거 요약

### Agent 자율성 관련

| 논문 | arXiv | 핵심 아이디어 | 활용 |
|------|-------|---------------|------|
| Talker-Reasoner | 2410.08328 | System 1(빠른 로컬) + System 2(느린 LLM) 분리 | 로컬 게이트 + Claude 판단 |
| Gatekeeper | 2502.19335 | 소형 모델이 대형 모델 호출 필요성 판단 | ZScoreGate / HalfSpaceTrees |
| RouteLLM | 2406.18665 | 경량 라우터가 약한/강한 모델 동적 선택 | 레짐별 Claude vs 캐시 라우팅 |
| AdaptEvolve | 2602.11931 | 단계별 모델 에스컬레이션 + 효과 비평가 | 로컬→캐시→Claude 계단식 |

### 트레이딩 LLM 에이전트

| 논문 | arXiv | 핵심 아이디어 | 활용 |
|------|-------|---------------|------|
| FinMem | 2311.13743 | 3-layer 메모리 (working/episodic/semantic) + 점수 기반 검색 | Graphiti 메모리 구조 |
| FinRS | 2511.12599 | 계층적 매크로/마이크로 + 멀티타임스케일 보상 | 레짐별 판단 주기 차별화 |
| TradingAgents | 2412.20138 | 2-tier 모델 (빠른+느린) + 다중 에이전트 | 로컬 분석 + Claude 결정 |
| CryptoTrade | 2407.09546 | 반성(Reflection) 메커니즘 + 온체인/오프체인 융합 | trade-reflect skill |
| ATLAS | 2510.15949 | Adaptive-OPRO: 거래 성과 기반 프롬프트 자기 최적화 | 주기적 시스템 프롬프트 개선 |
| Adaptive BTC | 2510.08068 | 일/주간 verbal feedback, 가중치 업데이트 없이 행동 조정 | Graphiti 메모리 기반 반성 |
| QuantAgent | 2509.09995 | 가격 기반 다중 에이전트, 최종 결정만 LLM | 구조화된 시그널 + Claude 결정 |
| QuantAgents | 2510.04643 | 리스크 점수 > 0.75일 때 긴급 회의 트리거 | 리스크 기반 Claude 호출 |

### 캐싱 & 효율

| 논문/도구 | 출처 | 핵심 아이디어 | 활용 |
|-----------|------|---------------|------|
| GPT Semantic Cache | arXiv:2411.05276 | 임베딩 유사도 기반 LLM 응답 캐싱 | 유사 시장상태 결정 재사용 |
| Redis SemanticCache | redis.io | 프로덕션급 시맨틱 캐시 + TTL | 실시간 캐시 인프라 |
| Plan Reuse | arXiv:2512.21309 | 기존 계획 재사용으로 밀리초 응답 | 반복 패턴 즉시 대응 |
| Domain Embeddings | arXiv:2504.02268 | 도메인 특화 임베딩이 캐시 히트율 향상 | 트레이딩 특화 임베딩 |
| Async Verified Cache | arXiv:2602.13165 | 2-tier 캐시 (정적+동적) + 비동기 검증 | 오프라인 패턴 + 온라인 결정 |
| River | github.com/online-ml/river | 스트리밍 ML (HalfSpaceTrees) | Layer 1 이상치 탐지 고도화 |

### 벤치마크 (주의사항)

| 논문 | arXiv | 발견 |
|------|-------|------|
| AI-Trader | 2512.10971 | 대부분 LLM 에이전트가 실시간 시장에서 수익 부진, 리스크 관리 약함 |
| CryptoBench | 2512.00417 | 특정 난이도에서 거의 전멸. 검색 성능 ≠ 예측 성능 |
| Agent Market Arena | 2510.11695 | 실시간 환경에서 LLM 에이전트 적응력 부족 |

→ 결론: LLM만으로는 부족. **로컬 정량 분석 + LLM 판단 + 메모리 학습**의 조합이 필수.

---

## 구현 우선순위

### Phase 1: 즉시 (1-2일)
1. `ZScoreGate` 구현 (30줄, 의존성 없음)
2. Claude 응답에 `next_check_seconds` + `wake_conditions` 추가
3. `AdaptiveAgentLoop`로 runner.py 교체
4. 시스템 프롬프트에 자기 스케줄링 지침 추가

### Phase 2: 단기 (1주)
5. State Fingerprint 기반 캐시 (regime+vol+momentum 이산화)
6. Graphiti 메모리 통합 (거래 기록 자동 저장/검색)
7. `trade-reflect` skill 추가 (주간 반성 루프)
8. Wake Condition 모니터 → Hook 연동

### Phase 3: 중기 (2-3주)
9. River `HalfSpaceTrees`로 Layer 1 고도화
10. Redis SemanticCache 도입
11. ATLAS 패턴: 거래 성과 기반 시스템 프롬프트 자동 최적화
12. 텔레그램/디스코드 알림 Hook 연동

### Phase 4: 장기
13. Liquidation cascade detection (Binance forceOrder 스트림)
14. 온체인 데이터 통합 (active wallets, gas)
15. Multi-agent debate (bull/bear 에이전트 토론)
