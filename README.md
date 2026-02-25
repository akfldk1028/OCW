# OCW — Agent-vs-Agent Crypto Trading Engine

> Claude(Sonnet)가 직접 트레이딩하는 자율 매매 엔진.
> Rule-based pipeline은 fallback 전용. **Claude = PRIMARY trader.**

## Quick Start (새 머신에서)

```bash
git clone https://github.com/akfldk1028/OCW.git
cd OCW/openclaw/extensions/trading-engine/python

python3 -m venv .venv && source .venv/bin/activate
pip install -r binance/requirements.txt

cp binance/.env.example binance/.env
# binance/.env 에 API 키 입력 (아래 참조)

python3 binance/main.py --testnet
```

### 필수 환경변수 (`binance/.env`)

```env
BINANCE_API_KEY=testnet_api_key        # https://testnet.binance.vision
BINANCE_SECRET_KEY=testnet_secret_key
BINANCE_PAPER=true
LIVE_TRADING=false
ANTHROPIC_API_KEY=sk-ant-api03-...     # https://console.anthropic.com/settings/keys
```

---

## Repo 구조

```
OCW/
├── README.md                ← 지금 읽는 파일
├── HANDOFF.md               ← 아키텍처 + 상태 + TODO 상세
├── .gitignore
├── docs/
│   └── MAC-MINI-SETUP.md   ← Mac Mini 24/7 운영 가이드
│
├── dashboard/               ← Next.js 모니터링 대시보드
│   ├── app/page.tsx         ← 메인 페이지 (7개 컴포넌트)
│   ├── lib/hooks.ts         ← SWR 데이터 fetching
│   └── components/          ← portfolio, positions, decisions, trades, regime, ts-weights
│
└── openclaw/
    └── extensions/trading-engine/python/   ← ★ 트레이딩 엔진 루트
        ├── fly.toml              ← Fly.io 배포 설정 (ocw-trader, nrt)
        ├── README.md             ← 엔진 상세 문서 (845줄)
        ├── HANDOFF.md            ← 엔진별 핸드오프
        ├── config.py             ← 전역 파라미터
        ├── binance/
        │   ├── main.py           ← CLI 진입점
        │   ├── runner.py         ← 6개 async 루프 오케스트레이터
        │   ├── api.py            ← 대시보드 FastAPI (8080 포트)
        │   ├── crypto_config.py  ← 트레이딩 설정값
        │   ├── Dockerfile
        │   ├── .env.example
        │   └── tests/
        ├── core/                 ← 핵심 모듈
        │   ├── claude_agent.py   ← Claude SDK 연동 (PRIMARY)
        │   ├── claude_auth.py    ← 인증 (API_KEY > OAuth)
        │   ├── zscore_gate.py    ← 4-layer 적응형 게이트
        │   ├── online_learner.py ← Regime-Aware Thompson Sampling
        │   ├── position_tracker.py ← 2-layer 포지션 감시
        │   ├── market_listener.py  ← Binance WS kline
        │   └── derivatives_monitor.py ← Funding/OI/CVD/L-S
        ├── analysis/             ← 레짐 감지
        │   ├── regime_detector_crypto.py ← HMM BIC
        │   └── macro_regime.py   ← FRED 4사분면
        └── brokers/
            └── binance.py        ← ccxt 주문 실행
```

---

## 아키텍처 (30초 요약)

```
WS tick → AdaptiveGate (4-layer, ~90% skip) → MarketSnapshot → Claude decides
                                                                    ↓
                                                              BUY/SELL/HOLD
                                                                    ↓
                                                         Broker execute → RL feedback
                                                                    ↓
                                                     Thompson Sampling Beta update
```

**6개 비동기 루프** (runner.py):
1. MarketListener — Binance WS kline (15m/1h/4h)
2. DerivativesMonitor — Funding/OI/CVD/L-S 폴링
3. PositionTracker — 30s safety + 180s agent 재평가
4. Heartbeat — 30s 헬스체크
5. Watchdog — WS 끊김 5분 후 강제 decision
6. API Server — FastAPI 대시보드 (포트 8080)

---

## 현재 상태 (2026-02-25)

| 항목 | 상태 |
|------|------|
| 코드 | Phase 1 완료, 40+ 버그 수정, 통합테스트 PASS |
| 거래 | **0건** (testnet 검증만, 실거래 없음) |
| 포지션 | 캐시 보유 (Beta(2,2) uninformative prior) |
| Fly.io | `ocw-trader` (nrt, 2048MB) — 서브시스템 작동 중 |
| Claude Agent | **ANTHROPIC_API_KEY 필요** (현재 fallback 모드) |
| 대시보드 | API 구현 완료, 프론트엔드 빌드 완료, Vercel 미배포 |

---

## 배포

### Fly.io (클라우드)
```bash
cd openclaw/extensions/trading-engine/python
fly secrets set BINANCE_API_KEY=... BINANCE_SECRET_KEY=... ANTHROPIC_API_KEY=...
fly deploy
fly logs --app ocw-trader
```

### 대시보드 (Vercel)
```bash
cd dashboard
npm install
NEXT_PUBLIC_API_URL=https://ocw-trader.fly.dev vercel --prod
```

### Mac Mini 24/7
→ `docs/MAC-MINI-SETUP.md` 참조 (launchd 자동 재시작 포함)

---

## TODO (우선순위)

1. **ANTHROPIC_API_KEY 발급 → Fly.io secret → 재배포** ← 최우선
2. Vercel 대시보드 배포
3. Walk-forward 백테스트
4. 3-6개월 testnet paper trading (24/7)
5. Phase 2: Graphiti 메모리 + SemanticCache

---

## 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| Thompson Sampling > Deep RL | 10-20 거래로 작동, GPU 불필요, 해석 가능 |
| Regime-aware TS | 레짐별 시그널 신뢰도가 다름 |
| tanh 비선형 | 이상치 포화로 안정적 파생상품 합성 |
| REST → WS 전환 | Binance IP ban 방지 (분당 REST 3.3 = 0.06%) |
| AdaptiveGate ~90% skip | Claude 호출 $0.03/건, 월 $5-10 목표 |
| Claude SDK (CLI 래퍼) | Raw API보다 추론 품질 우수 |

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| `HANDOFF.md` | 프로젝트 정체, 환경변수, 배포, 파일맵 |
| `docs/MAC-MINI-SETUP.md` | Mac Mini 셋업 + launchd |
| `openclaw/.../python/README.md` | 엔진 상세 (845줄): 설계, 파이프라인, 백테스트, 버그 이력 |
| `openclaw/.../python/HANDOFF.md` | 엔진 핸드오프: 데이터 흐름, BUY/SELL/RL 플로우 |
