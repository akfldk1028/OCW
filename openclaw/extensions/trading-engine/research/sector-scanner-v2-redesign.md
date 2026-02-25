# Sector Scanner v2 — 논문 기반 알고리즘 재설계

> Date: 2026-02-13
> Status: DESIGN (구현 전)
> v1 문제점: 임의 가중치, 검증 없음, 균등 팩터, 키워드 감성 → 모두 논문이 반증

---

## v1 vs v2 비교

| 항목 | v1 (현재) | v2 (논문 기반) | 근거 논문 |
|------|----------|---------------|----------|
| 모멘텀 윈도우 | 1w/4w/12w 임의 가중 | **6개월 lookback** | Moskowitz-Grinblatt 1999 |
| 팩터 가중치 | 균등 (25% each) | **XGBoost 학습 가중치** | arXiv 2508.18592 |
| 감성 분석 | 키워드 사전 | **FinBERT + market-derived labels** | arXiv 2502.14897 |
| 레짐 감지 | 없음 | **HMM 2-3 states** | arXiv 2601.19504 |
| RSI 점수 | RSI=50 선호 (선형) | **RSI를 feature로 사용, 가중치는 학습** | arXiv 2507.07107 |
| 종목 선정 | 단순 스코어링 | **RL 기반 ranking + selection** | arXiv 2412.18563 |
| 검증 | 없음 | **레짐별 백테스트 필수** | arXiv 2512.10913 |

---

## Phase 1: 즉시 개선 (현재 코드 수정)

### 1.1 모멘텀 윈도우 수정
```
현재: [1w, 4w, 12w] weights [0.3, 0.4, 0.3]
변경: [4w, 12w, 26w] — 약 1m/3m/6m
근거: Moskowitz-Grinblatt (1999) — 6개월 lookback이 학술적으로 검증됨
      가중치는 백테스트로 결정 (기본: 균등 → 백테스트 결과로 교체)
```

### 1.2 팩터 가중치를 백테스트 기반으로
```
현재: (momentum + volume + rsi_score + sentiment) / 4
변경: 과거 N개월 데이터로 각 팩터의 next-month 수익률 예측력 측정
      IC (Information Coefficient) 기반 가중치 할당
근거: arXiv 2507.07107 — 적응형 팩터 가중치가 균등 가중 대비 우수
      arXiv 2508.18592 — XGBoost + AdaBoost 동적 가중치
```

### 1.3 품질 필터 추가
```
현재: 없음 (모멘텀만)
추가: 모멘텀 + 품질(ROE, 수익 성장) 결합 필터
근거: State Street 2024 분석 — 2024 모멘텀이 강한 이유는
      quality firms이 리드했기 때문. dot-com과 다름
```

### 1.4 최소 백테스트 추가
```
현재: 없음
추가: scan_sectors() 결과를 과거 12개월 rolling으로 검증
      - 추천 종목의 next-month 수익률 vs SPY
      - 섹터 랭킹의 hit rate (상위 3 섹터가 실제로 outperform했는가)
근거: arXiv 2512.10913 (167편 메타분석) — implementation quality 31% > algorithm 8%
```

---

## Phase 2: 중기 개선 (새 모듈 추가)

### 2.1 HMM 레짐 감지 모듈
```python
# regime_detector.py (신규)
class RegimeDetector:
    """HMM 기반 시장 레짐 감지.

    States:
      0 = low-volatility (정상)
      1 = high-volatility (위험)
      (선택) 2 = trending (강세/약세)

    논문: arXiv 2601.19504, arXiv 2402.05272
    성과: HMM Sharpe 1.05, 레짐 전환 시 +1-4% 연 초과수익
    """

    def detect(self, returns: np.ndarray) -> int:
        """현재 레짐 반환 (0=정상, 1=위험)"""
        ...

    def adjust_weights(self, base_weights: dict, regime: int) -> dict:
        """레짐에 따라 팩터 가중치 조정.
        정상: 모멘텀 가중치 ↑
        위험: 거래량/변동성 가중치 ↑, 방어적 섹터 선호
        """
        ...
```

### 2.2 FinBERT 감성 분석 교체
```python
# sentiment_finbert.py (신규)
class FinBERTScorer:
    """FinBERT 기반 금융 감성 분석.

    논문: arXiv 2502.14897 — market-derived labels +11% 정확도
    접근:
      1) FinBERT로 뉴스 헤드라인 임베딩
      2) market-derived label: 해당 뉴스 후 1-5일 수익률로 라벨링
      3) fine-tuning으로 가격 반응 예측 학습

    기존 keyword scorer 대비:
      - 키워드 사전: "surge" → +0.8 (인간 라벨)
      - FinBERT: "surge" → 실제 가격 반응 기반 학습된 점수
    """

    def score_text(self, text: str) -> float:
        """[-1, 1] 감성 점수. 기존 인터페이스 호환."""
        ...
```

### 2.3 XGBoost 메타 스코어러
```python
# stock_ranker.py (신규)
class StockRanker:
    """XGBoost 기반 종목 랭킹.

    논문: arXiv 2508.18592 — XGBoost ROC-AUC 0.953

    Features:
      - momentum (multiple windows: 5d, 20d, 60d, 120d)
      - volume_ratio (vs 20d, 50d SMA)
      - rsi_14
      - macd, macd_signal
      - atr_14 (변동성)
      - sentiment_score (FinBERT)
      - sector_momentum (상대강도)
      - regime (HMM state)

    Target: next_20d_return > median (binary classification)
    """

    def train(self, historical_data: pd.DataFrame) -> None:
        """과거 데이터로 학습. 6개월 rolling window."""
        ...

    def rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """종목 랭킹 반환. 확률 기반 스코어."""
        ...
```

---

## Phase 3: 장기 비전 (에이전트 경쟁 대비)

### 3.1 Multi-Agent RL 종목 선정
```
논문: arXiv 2412.18563 — Sharpe 6.83, 68.83% 연수익
접근:
  - Agent A: 수익률 최대화 reward
  - Agent B: Sharpe ratio 최대화 reward
  - Agent C: Max drawdown 최소화 reward
  - Voting: 3 agent의 합의로 최종 종목 선정
```

### 3.2 에이전트 경쟁 시대 대비
```
논문: arXiv 2507.03904 — Agent Exchange
시장 전망: $7.55B (2025) → $199B (2034), CAGR 43.84%

핵심 인사이트:
  - 모든 시장 참여자가 에이전트 → 과거 패턴 빠르게 소멸
  - 적응 속도가 알파의 원천 (static strategy → 도태)
  - 메타학습 (arXiv 2509.09751): 자기개선 루프 필수
  - 에이전트 간 경쟁에서 살아남으려면:
    1. 연속 학습 (매일/매주 재학습)
    2. 다양한 시그널 소스 (on-chain, off-chain, macro)
    3. 빠른 적응 (regime switch 감지 → 전략 전환)
    4. 반성 메커니즘 (과거 실패 분석 → 전략 수정)
```

### 3.3 목표 성과 지표
```
논문 기반 달성 가능 목표:
  - Sharpe Ratio: >1.5 (MLP: 1.61, arXiv 2508.14656)
  - 레짐 전환 alpha: +1-4% 연간 (arXiv 2402.05272)
  - 감성 정확도: +11% vs 키워드 (arXiv 2502.14897)
  - 앙상블 개선: 10-15% vs 단일 모델 (arXiv 2501.10709)
```

---

## 구현 우선순위

| 순서 | 작업 | 난이도 | 임팩트 | 논문 근거 |
|------|------|--------|--------|----------|
| 1 | 모멘텀 윈도우 6개월로 수정 | 낮음 | 중간 | Moskowitz 1999 |
| 2 | 백테스트 모듈 추가 | 중간 | **높음** | 2512.10913 |
| 3 | HMM 레짐 감지 | 중간 | 높음 | 2601.19504 |
| 4 | XGBoost 팩터 스코어러 | 중간 | **높음** | 2508.18592 |
| 5 | FinBERT 감성 교체 | 높음 | 높음 | 2502.14897 |
| 6 | Multi-agent RL 종목 선정 | 높음 | 최고 | 2412.18563 |

---

## 참고 논문 인덱스

| # | 논문 | ID | 핵심 기여 |
|---|------|----|----------|
| F1 | Moskowitz-Grinblatt 1999 | JoF 1999 | 섹터 모멘텀 6개월 검증 |
| F2 | ML Multi-Factor Trading | 2507.07107 | 적응형 팩터 가중치 |
| F3 | Dynamic Weighting Methods | 2508.18592 | XGBoost + AdaBoost 동적 가중 |
| F4 | Market-Derived Sentiment | 2502.14897 | 가격반응 기반 감성 +11% |
| F5 | Regime-Adaptive Trading | 2601.19504 | HMM 레짐 감지 |
| F6 | Jump Model Regime | 2402.05272 | 레짐 전환 +1-4% alpha |
| F7 | Multi-Agent Portfolio | 2412.18563 | Sharpe 6.83 |
| F8 | RL Meta-Analysis (167편) | 2512.10913 | 구현품질 > 알고리즘 |
| F9 | Agent Exchange | 2507.03904 | 에이전트 경제 시대 |
| F10 | Ensemble RL FinRL | 2501.10709 | 앙상블 변동성 50%↓ |
| F11 | Volatility-Guided Selection | 2505.03760 | 변동성 기반 사전 필터링 |
| F12 | LLM Regime Failure | 2505.07078 | LLM 레짐 무시 → 실패 |
| F13 | Stockformer | 2401.06139 | Price-volume 팩터 모델 |
| F14 | Behavior-DRL | Nature 2026 | 행동 편향 통합 RL |
