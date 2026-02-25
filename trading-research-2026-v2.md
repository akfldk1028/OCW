# Trading Engine Research v2 (2026-02-23)

이전 리서치: `ai-trading-research.md` (2026-02-12)

## 1. 김민겸 / IQC 2025 올웨더 전략

**대회**: WorldQuant IQC 2025 (제5회 국제 퀀트 챔피언십)
- 142개국, 8만명 참가, 26.3만개 모델 제출
- 한국인 최초 글로벌 1위, 총상금 $23,000

**핵심 인사이트**:
- 다른 참가자: 수익률/알고리즘 완성도에 집중
- 김민겸: **거시경제 움직임 + 안정성** 중심
- "모든 거시경제 상황에 대비하는 올웨더 포트폴리오 전략 풀"
- 기준금리 변동 등 매크로 지표 분석이 차별화 포인트

**우리 엔진 적용**:
- FRED 거시경제 레짐 4분면 분류 구현 (성장/인플레 상승·하강)
- 레짐별 최적 전략 풀 자동 선택
- 수익률보다 MDD/Sharpe 안정성 우선

Sources:
- https://news.unist.ac.kr/kor/20251014-2/
- https://www.fnnews.com/news/202510131612292993
- https://zdnet.co.kr/view/?no=20251013174556

---

## 2. 새로운 시그널 (우선순위순)

### P0: 무료, Binance API 직접 가능
| 시그널 | API | 핵심 로직 |
|--------|-----|----------|
| **Taker Buy/Sell Delta (CVD)** | `fapiPublicGetFuturesDataTakerBuySellVol` | 매수·매도 공격량 차이. CVD 하락 + 가격 상승 = 반전 임박 |
| **Long/Short Ratio (Top Traders)** | `fapiPublicGetFuturesDataTopLongShortPositionRatio` | 상위 20% 트레이더 포지션. 극단값 = 스퀴즈 가능성 (역방향) |
| **Basis Spread** | `fapiPublicGetPremiumIndex` | 선물-현물 괴리. >30% 연환산 = 과열, <0 = 항복 |

### P1: 무료, 추가 구현 필요
| 시그널 | API | 핵심 로직 |
|--------|-----|----------|
| **Liquidation Cascade** | Binance WS `forceOrder` | 4h 청산량 2σ 초과 = 방향 확인/가속 |
| **Fear & Greed Index** | alternative.me (무료) | <25 극단공포 = 매수필터, >80 극단탐욕 = 매도필터 |
| **Stablecoin Supply** | DeFiLlama (무료) | 7일 스테이블코인 공급 변화 = 매수력 프록시 |

### P2: 유료 또는 저빈도
| 시그널 | API | 비용 |
|--------|-----|------|
| Exchange Net Flow | CryptoQuant | $29/월 |
| BTC ETF Daily Flow | CoinGlass | 무료(일간) |
| MVRV/SOPR | CryptoQuant/Glassnode | $29-799/월 |

---

## 3. RL 분석: Thompson Sampling 유지 + 개선

### 결론: 알고리즘 바꾸지 마라
- 167편 메타분석: 알고리즘 선택 = 성과의 8% (p=0.640, 통계적 유의성 없음)
- 구현 품질 = 31% (가장 큰 요인)
- EXP3 (적대적 밴딧): 시장은 적대적이 아님 (우리는 price-taker)
- UCB: 소표본(월 15-30회)에서 TS보다 불리

### 개선 1: Regime-Aware TS (최우선)
```python
# 기존: 에이전트별 단일 Beta
{"momentum": Beta(3.5, 1.8)}

# 개선: 레짐별 Beta
{"momentum": {"trending": Beta(3.5, 1.8), "ranging": Beta(1.5, 2.8)}}
```
- 레짐 감지기 이미 있음 → 바로 연결
- 8 에이전트 x 2-3 레짐 = 16-24 분포
- 계층적 fallback: 레짐별 데이터 <3건이면 글로벌 posterior 사용

### 개선 2: Gaussian TS (선택)
- 현재 sigmoid 매핑이 큰 수익 정보를 압축 (+30% → 0.95, +10% → 0.73)
- Normal-Normal conjugate로 연속 PnL 직접 모델링
- 단 Beta 모델이 충분히 잘 작동하면 불필요

### gamma=0.95는 적절
- 20거래 후 유효 가중치 36% (≈1개월)
- 60거래 후 5% (≈3개월) → 실질적으로 잊힘
- 크립토 레짐 지속기간(수주~수개월)과 잘 맞음
- CADTS 논문은 gamma=0.9 사용 → 더 공격적 망각, 크립토에 적합할 수 있음

Sources:
- CADTS: https://arxiv.org/abs/2410.04217
- 167편 메타분석: https://arxiv.org/abs/2512.10913
- POW-dTS: Springer AI Review 2025
- FinRL Contest: https://arxiv.org/abs/2504.02281

---

## 4. OpenClaw 통합 방안

### 핵심: fastapi-mcp로 MCP 도구 노출
```python
# server.py에 추가
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()  # 모든 FastAPI 엔드포인트를 MCP 도구로 노출
```
- Claude가 직접 시장 조회, 거래 실행, 파라미터 조정 가능
- "openclaw가 직접 판단" 철학에 부합

### Hook 연결
- `pre-trade`: 거래 전 리스크 검증 → Claude에게 확인 요청
- `post-trade`: 거래 후 메모리에 기록
- `market-alert`: 시장 이벤트 → Claude 알림

### Memory 활용
- 거래 이력을 Claude memory에 저장
- 에이전트 성과 패턴 학습
- 크로스세션 컨텍스트 유지

---

## 5. FRED 매크로 레짐 설계 (올웨더)

### 4분면 레짐 분류 (Bridgewater All-Weather 기반)

```
              인플레 상승          인플레 하강
           ┌─────────────┬─────────────┐
성장 상승  │  REFLATION   │   GOLDILOCKS │
           │  원자재/TIPS  │   주식/크립토 │
           ├─────────────┼─────────────┤
성장 하강  │  STAGFLATION │   DEFLATION  │
           │  현금/금      │   채권/달러   │
           └─────────────┴─────────────┘
```

### FRED 지표
| 지표 | Series ID | 빈도 | 용도 |
|------|-----------|------|------|
| Fed Funds Rate | FEDFUNDS | 월간 | 금리 방향 |
| 10Y Treasury | DGS10 | 일간 | 장기 금리 |
| 2Y-10Y Spread | T10Y2Y | 일간 | 경기 선행 |
| CPI YoY | CPIAUCSL | 월간 | 인플레이션 |
| Core PCE | PCEPILFE | 월간 | Fed 선호 인플레 |
| M2 Money Supply | M2SL | 월간 | 유동성 |
| Unemployment | UNRATE | 월간 | 고용/성장 |
| ISM Manufacturing | MANEMP | 월간 | 경기 선행 |

### 크립토 레짐 매핑
- **GOLDILOCKS** (성장↑ 인플레↓): 위험자산 선호 → 크립토 공격적 (position_pct 상향)
- **REFLATION** (성장↑ 인플레↑): 혼재 → 크립토 중립 (기본값)
- **STAGFLATION** (성장↓ 인플레↑): 위험회피 → 크립토 보수적 (position_pct 하향, 높은 trail)
- **DEFLATION** (성장↓ 인플레↓): 유동성 확대 기대 → 크립토 중립~긍정

---

## 6. 구현 로드맵

| 순서 | 항목 | 파일 | 난이도 |
|------|------|------|--------|
| 1 | Taker Delta + L/S Ratio | derivatives_monitor.py | 낮음 |
| 2 | FRED 매크로 레짐 | analysis/macro_regime.py (신규) | 중간 |
| 3 | Regime-Aware TS | core/online_learner.py | 중간 |
| 4 | Liquidation + Basis | derivatives_monitor.py, ws_stream.py | 낮음 |
| 5 | Fear & Greed 필터 | pipeline.py | 낮음 |
| 6 | OpenClaw MCP | server.py | 낮음 |