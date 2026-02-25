# 리서치 논문 정리

## RL 트레이딩 증거

### 백테스트 과적합 방지

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Bailey & Lopez de Prado, PBO | Combinatorially Symmetric Cross-Validation으로 백테스트 과적합 확률 측정 | SSRN 2326253 |
| Bailey & Lopez de Prado, DSR | 다중 비교 보정된 Sharpe Ratio. 수백 개 전략 테스트 시 우연한 성과 필터링 | SSRN 2460551 |

적용: 모든 전략은 PBO < 0.5, DSR 유의성 검증을 통과해야 실전 배포.

### 메타분석

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| 167편 RL 트레이딩 메타분석 | 알고리즘 선택 8%, 구현 품질 31%가 성과 결정. 데이터 전처리와 비용 모델이 알고리즘보다 중요 | arXiv 2512.10913 |

### 크립토 RL 연구

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Gort et al., 크립토 RL 오버피팅 | 크립토 시장에서 RL 에이전트의 과적합 경향 분석. IS 성과와 OOS 성과의 괴리 | arXiv 2209.05559 |

### RL 일반화 실패

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Zhang et al. (IJCAI) | RL 트레이딩 에이전트의 시장 레짐 변화 시 일반화 실패. 분포 이동(distribution shift) 문제 | arXiv 2307.11685 |

### RL 대회

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| FinRL Contest 2024 | 공개 대회에서 RL 전략들의 실질 성과 비교. 대부분 단순 벤치마크와 유사하거나 미달 | arXiv 2504.02281 |

## Agent-vs-Agent 경쟁 소스

### 자율적 담합

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Wharton NBER (2025) | RL 에이전트들이 명시적 통신 없이도 암묵적 담합(tacit collusion) 형성. 가격 경쟁이 줄어들고 소비자 후생 감소 | Wharton/NBER Working Paper |

시사점: 시장에 RL 에이전트가 늘어나면 특정 자산에서 비정상적 가격 패턴 발생 가능.

### LLM 실거래 경쟁

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Alpha Arena (2025) | GPT-4, Claude 등 LLM 6개를 실제 시장에서 경쟁. 6개 중 4개 손실. 뉴스 해석 능력은 있으나 타이밍/리스크 관리 부족 | Alpha Arena Report |

### GPU 가속 Multi-Agent RL

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| JaxMARL-HFT | JAX 기반 GPU 가속 Multi-Agent RL 환경. HFT 시뮬레이션에서 에이전트 간 상호작용 연구 | arXiv 2511.02136 |

### 시스템 리스크

AI 에이전트 간 허딩(herding) 행동과 플래시 크래시 위험:
- 동일한 학습 데이터/구조를 가진 에이전트들이 동시에 같은 방향으로 매매
- 유동성 급감 시 피드백 루프로 가격 급락 가능
- 특히 크립토 시장은 서킷브레이커 부재로 리스크 증폭

## 실시간/온라인 강화학습 (Online RL)

### Thompson Sampling / Bandit 접근

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Adaptive Portfolio via Thompson Sampling | 기존 전략을 arm으로 블렌딩, 실제 데이터셋에서 "marked superiority" | arXiv 1911.05309 |
| Portfolio Blending via Thompson Sampling | Bayes-updated 분포로 heuristic 포트폴리오 블렌딩 | IJCAI 2016 Proc/283 |
| CADTS (Combinatorial Adaptive Discounted TS) | 슬라이딩 윈도우 + 할인으로 non-stationarity 처리 | arXiv 2410.04217 |
| PRBO (Bandit-over-Bandit) | 비정상 연속 bandit, 동적 파라미터 조정 | arXiv 2208.02901 |
| Contextual MAB + Neuroevolution | "제한된 데이터에서 bandit의 샘플 효율이 full RL을 압도" | ACM 2024, 10.1145/3638530.3664145 |
| Strategy Selection Using MAB | 금융 시장에서 Multi-Armed Bandit 전략 선택 | ResearchGate 2024 |

**핵심**: Thompson Sampling은 30-50회 거래만으로 유의미한 적응 시작. 이론적 보장 있음.

### Incremental/Continual RL

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| Continual Portfolio via Incremental RL | Warm-start + importance weighting으로 시장 변화 적응 | Springer s13042-022-01639-y |
| Katsikas et al., Bi-Directional KT | 양방향 지식 전이로 catastrophic forgetting 방지 | IEEE MLSP 2024 |
| IFF-DRL (Incremental Forecast Fusion) | 자기지도 예측 + DRL 하이브리드, MSE 5.93% 개선 | Expert Sys. w/ Appl. 2025 |
| Incremental PPO with LSTM | 컨셉 드리프트 대응 포트폴리오 최적화 | MDPI Computers 2025, 14/7/242 |

**핵심**: Full RL은 950-1200 에피소드 필요 (1-2년). Incremental 접근은 Phase 2에서 고려.

### 업계 트렌드 (167편 메타분석, arXiv 2512.10913)

- 순수 RL: 85% (2020) → 58% (2025)
- **하이브리드 (RL + 전통)**: 15% (2020) → **42% (2025)**
- 알고리즘 선택과 성과 간 통계적 유의관계 **없음** (p=0.640)
- 학습 기간과 성과 간 통계적 유의관계 **없음** (p=0.591)

### 우리 시스템 적용 로드맵

- **Phase 1 (구현 완료)**: Thompson Sampling 에이전트 가중치 적응 (`online_learner.py`)
- **Phase 2 (6개월 데이터 후)**: Incremental PPO position sizing (Katsikas 2024)
- **Phase 3 (실전 검증 후)**: PRBO bandit-over-bandit 메타 전략 선택

## 최신 연구 (2024-2025): 모멘텀 + 기술적 지표 결합

### 모멘텀 + 리버전 결합 전략

| 논문/소스 | 핵심 내용 | 성과 | 레퍼런스 |
|-----------|-----------|------|----------|
| Systematic Crypto Trading (2025) | 모멘텀 + 평균회귀 50/50 블렌딩. 트렌딩 시장에서 모멘텀, 횡보장에서 리버전 | **Sharpe 1.71, 연 56%, T-stat 4.07** | Medium/@briplotnik |
| BTC-Neutral Residual MR | 잔차 평균회귀, 특히 2021년 이후 강력 | **Sharpe ~2.3** (post-2021) | 동일 |
| Risk-Managed Crypto Momentum | 리스크 관리된 모멘텀이 기존 대비 개선 | **주간 수익 3.18%->3.47%, 연 Sharpe 1.12->1.42** | ScienceDirect S1544612325011377 |

**핵심**: 모멘텀 OR 리버전이 아니라 **BOTH**. 레짐에 따라 전환.

### 레짐 스위칭 전략

| 논문/소스 | 핵심 내용 | 성과 | 레퍼런스 |
|-----------|-----------|------|----------|
| Price Action Lab (2024) | SPY/QQQ/TLT/GLD 레짐 스위칭. 베어=리버전, 불=모멘텀 | **Sharpe 1.15-1.21, 연 10.3-10.8%, MDD 16.5%** | priceactionlab.com 2024/01 |
| Regime-Switching Model | 마르코프 전환 모형: 주식 수익률의 모멘텀+리버전 레짐 | 학술 모델 | Econ. Modelling 2023, S0264999323000494 |

### RSI 모멘텀 (크립토 전용)

| 발견 | 상세 |
|------|------|
| **RSI 모멘텀은 크립토에서 작동** | 주식에서는 RSI 리버전, 크립토에서는 RSI **모멘텀** 시그널이 유효 |
| 짧은 룩백이 최적 | BTC에서 최적 RSI 기간은 상대적으로 짧음 |
| 리버전은 BTC에서 실패 | "buy the dip and sell strength doesn't work on Bitcoin" |

### Talyxion 프레임워크 (실거래 검증)

| 지표 | 수치 |
|------|------|
| 플랫폼 | Binance Futures, 30일 실거래 |
| **Sharpe** | **5.72** |
| **MDD** | **4.56%** |
| 승률 | 57.71% (131/227) |
| 수익 | +16.68% (30일) |
| 방법 | Universe선택 -> Alpha테스트 -> 변동성조정포트폴리오 -> DD리스크관리 |
| 레퍼런스 | arXiv 2511.13239 |

### Dynamic Grid Trading (크립토)

| 지표 | 수치 |
|------|------|
| 자산 | BTC, ETH (분봉 데이터) |
| 기간 | 2021.01 ~ 2024.07 |
| **IRR** | **60-70%** (최적 파라미터) |
| MDD | ~50% (vs B&H 80%) |
| 방법 | 가격이 그리드 상/하한 돌파 시 그리드 리셋 + 수익 재투자 |
| 레퍼런스 | arXiv 2506.11921, GitHub: colachenkc/Dynamic-Grid-Trading |

### Slow Momentum + Fast Reversion (Oxford)

| 지표 | 수치 |
|------|------|
| 저자 | Kieran Wood, Stephen Roberts, Stefan Zohren (Oxford) |
| 핵심 | Changepoint detection으로 모멘텀<->리버전 전환 시점 감지 |
| 결과 | 순수 TSMOM 대비 터닝포인트에서 개선 |
| 레퍼런스 | arXiv 2105.13727, JFDS Winter 2022 |

### ML + 기술적 지표 결합

| 논문/소스 | 핵심 내용 | 성과 | 레퍼런스 |
|-----------|-----------|------|----------|
| Enhanced Technical Indicators (2024) | RSI(14) + SMA(14) + 센티먼트. BTC+크립토 포트폴리오 | B&H 대비 높은 Sharpe | arXiv 2410.06935 |
| Random Forest + RSI/MACD/EMA | 매수/매도 시그널 분류 | **정확도 86%** | arXiv 2410.06935 |
| DQN 전략 선택 | RSI, SMA, BB, Momentum, VWAP 5개 전략 중 동적 선택 | 적응형 시장 대응 | tandfonline 2025.2594873 |
| Sentiment + MV Optimization | 감성분석 + 평균분산 최적화 크립토 포트폴리오 | 리스크 조정 수익 개선 | arXiv 2508.16378 |

### 크립토 트렌드 팔로잉 (학술 검증)

| 논문 | 핵심 내용 | 성과 | 레퍼런스 |
|------|-----------|------|----------|
| **Zarattini, Pagani, Barbon (2025)** | Top 20 liquid coins long-only 트렌드 팔로잉. 5-360일 룩백 앙상블 | **Net Sharpe > 1.5, Alpha 10.8% vs BTC** | SSRN 5209907 |
| Huang, Sangiorgi, Urquhart (2024) | Volume-weighted TSMOM. 승자/패자 포트폴리오 | **일간 0.94%, 연 Sharpe ~2.17** (gross) | SSRN 4825389 |
| Crypto Momentum Crashes (2025) | 크립토 모멘텀 심각한 크래시 위험. 변동성 관리 필수 | 크래시 방지 없으면 78%+ MDD | Springer s11408-025-00474-9 |

### 섹터 모멘텀 최신 증거

| 논문 | 핵심 내용 | 성과 | 레퍼런스 |
|------|-----------|------|----------|
| Mamais (2025) | 섹터 모멘텀 랭킹이 경제 레짐에 따라 예측 가능하게 변화 | OOS에서도 강건 | J. of Forecasting |
| Quantpedia 정제 버전 | 4-6 섹터 long, max 30% 비중 | **Sharpe 0.60-0.72**, 20년+ SPY 초과 | quantpedia.com |
| Daniel & Moskowitz (2016) | 모멘텀 크래시는 예측 가능. Dynamic momentum = **Sharpe 2배** | 베어마켓 반등 시 worst | JFE |

### 모멘텀 팩터 학술 증거 (원본 논문)

| 논문 | 기간 | 수익 | Sharpe | 레퍼런스 |
|------|------|------|--------|----------|
| Jegadeesh & Titman (1993) | 1965-1989 | 월 ~1.0%, 연 ~12% L/S | ~0.5 | J. of Finance |
| Moskowitz, Ooi, Pedersen (2012) | 1985-2009 | 58개 선물, 52/58 유의 | **1.17-1.34** (gross) | JFE |
| AQR Value+Momentum (2013) | 다국가 다자산 | 미국 단독 0.14, 분산 0.55-0.73 | 0.40 (50/50 blend) | J. of Finance |
| Hurst, Ooi, Pedersen (2017) | **1880-2013** | 67개 시장, 134년간 양수 | **~0.4 (net)** | SSRN 2993026 |

### 출판 후 성과 디케이 (CRITICAL)

| 논문 | 핵심 내용 | 레퍼런스 |
|------|-----------|----------|
| **McLean & Pontiff (2016)** | 97개 이상현상 분석: OOS 수익 **26% 하락**, 출판 후 **58% 하락** | J. of Finance |

**의미**: 논문 Sharpe 1.0 → 실전 **0.4-0.5**. 우리 백테스트 Sharpe 1.65 → 실전 **0.7-1.0** 예상.

### CTA 실적 (실제 펀드 성과, 수수료 후)

| 연도 | SG CTA Index | S&P 500 |
|------|-------------|---------|
| 2020 | +3.1% | +18.4% |
| 2021 | ~+7% | +28.7% |
| 2022 | **+20.1%** | **-19.4%** |
| 2023 | ~0% | +26.3% |
| 2024 | +2.4% | +25.0% |

5년 누적: CTA ~+35%, S&P 500 ~+95%. **트렌드 팔로잉은 위기 보험이지 알파가 아님.**

### MTUM vs SPY (실제 모멘텀 ETF)

| 연도 | MTUM | SPY | 차이 |
|------|------|-----|------|
| 2020 | +29.9% | +18.4% | +11.5% |
| 2021 | +13.4% | +28.7% | -15.3% |
| 2022 | -18.3% | -18.1% | -0.2% |
| 2023 | +9.2% | +26.3% | -17.1% |
| 2024 | +32.9% | +25.0% | +7.9% |

5년: MTUM ~+79% vs SPY ~+95%. **모멘텀 ETF가 SPY를 16%p 언더퍼폼.**

### 거래 비용 분석 ($5K 크립토 계좌)

| 시나리오 | 월 거래 | 수수료/건 | 연 비용 드래그 | 연 비용($) |
|----------|---------|-----------|---------------|-----------|
| 보수적 | 15 RT | 0.02% | 7.2% | $360 |
| 능동적 | 30 RT | 0.02% | 14.4% | $720 |
| Taker | 15 RT | 0.05% | 18.0% | $900 |

+펀딩레이트, 슬리피지 별도.

### $5K 소자본 현실적 기대치

| 전략 | 연 수익률 | Sharpe | MDD | 비고 |
|------|-----------|--------|-----|------|
| 보수적 (섹터 ETF 월간) | 10-15% | 0.5-0.8 | ~25% | SPY 수준, 낮은 MDD |
| 중간 (크립토+ETF 트렌드) | 15-25% | 0.7-1.2 | 30-40% | 능동적 관리 필요 |
| 공격적 (크립토 모멘텀) | 25-50%+ | 1.0-1.5 | 40-70% | 좋은 해 크고 나쁜 해 심각 |
| **백테스트 Sharpe 할인** | **30-50% 감소** | | | **Sharpe 1.65 -> 실전 0.8-1.2 예상** |

### 우리 시스템에 적용

1. **Crypto Regime Blend** (구현: `backtest_v2.py` Strategy A)
   - 레짐 감지: 20d vol + 50d SMA trend
   - TRENDING: RSI 모멘텀 (RSI > 50 + 양의 14d 모멘텀)
   - RANGING: 볼린저 밴드 평균회귀 (하단 밴드 매수)
   - Talyxion식 DD 리스크 관리 (15% 낙폭 시 전량 매도)

2. **Combined Signal** (구현: `backtest_v2.py` Strategy C)
   - RSI(0.25) + MACD(0.20) + BB(0.20) + Trend(0.20) + Volume(0.15)
   - 종합 점수 > 0.6 매수, < 0.3 매도

3. **Thompson Sampling 적응** (기존 `online_learner.py`)
   - 어떤 지표 조합이 최근에 작동하는지 실시간 학습

## 단기 타임프레임 크립토 트레이딩 연구 (2026-02-23 추가)

### 배경
3년 백테스트에서 Regime Blend가 BTC B&H와 거의 동등(+306% vs +309%). 일봉 3일 리밸런싱으로는 알파 한계. 5m/15m/1h 단기 전략으로 추가 알파 탐색.

PDF 다운로드: `python/docs/papers/` 디렉토리

### Tier 1: 직접 구현 가능

| 논문 | 핵심 | 타임프레임 | 성과 | 적용 | 레퍼런스 |
|------|------|-----------|------|------|----------|
| Multi-TF Neural Net for Crypto HFT | 멀티 타임프레임(분/시/일) 트렌드 + CNN 방향 예측. Soft attention으로 3개 TF 동적 가중 | 분봉/시봉/일봉 | Profit Factor 1.15 (tx cost 0.05% 포함) | ★★★★★ 우리 Regime Blend에 15m/1h 레이어 추가 | arXiv 2508.02356 |
| Informer for HF Bitcoin | Informer 아키텍처로 5m/15m/30m BTC 거래. GMADL loss가 5분봉에서 최강 | **5m/15m/30m** | GMADL이 B&H + MACD + RSI 전부 beat | ★★★★★ 정확히 우리가 원하는 타임프레임 | arXiv 2503.18096 |
| TA + ML Bitcoin | LightGBM vs LSTM vs EMA vs MACD+ADX. LSTM 65.23% return (1년) | 일봉 | LSTM 65% > LightGBM 53% > B&H 43% > MACD 35% (비용 후: LSTM 53%, LightGBM 40%) | ★★★★☆ LightGBM 가볍고 Mac Mini에서 가능 | arXiv 2511.00665 |
| Ensemble DRL Intraday Crypto | DRL 앙상블 + mixture distribution policy. 다중 validation 기간 모델 선택 | 인트라데이 | 단일 모델 대비 일반화 성능 개선 | ★★★★☆ Thompson Sampling과 결합 가능 | arXiv 2309.00626 |

### Tier 2: 전략 보강

| 논문 | 핵심 | 적용 | 레퍼런스 |
|------|------|------|----------|
| Meta-RL-Crypto | Transformer meta-learning + RL 자기개선 에이전트. on-chain + 뉴스 + 소셜 | ★★★☆☆ 컨셉 좋으나 구현 복잡 | arXiv 2509.09751 |
| Multi-Agent Bitcoin Trading (LLM) | LLM 기반 alpha 생성. 정적 모델 대신 실시간 센티먼트 반영 | ★★★☆☆ API 비용 고려 | arXiv 2510.08068 |
| Adaptive TFT for Crypto | Temporal Fusion Transformer 적응형 단기 예측. 동적 subseries + 패턴 분류 | ★★★☆☆ 예측 모듈로 추가 가능 | arXiv 2509.10542 |
| PulseReddit HFT Crypto | Reddit + LLM 멀티에이전트 → 단기 크립토 트레이딩 | ★★☆☆☆ 데이터 파이프라인 복잡 | arXiv 2506.03861 |

### Tier 3: 인프라/검증

| 논문 | 핵심 | 적용 | 레퍼런스 |
|------|------|------|----------|
| DRL Backtest Overfitting 방지 | 크립토 DRL 과적합 탐지 프레임워크. IS vs OOS 괴리 정량화 | ★★★★★ 우리 162 파라미터 스윕 검증 필수 | arXiv 2209.05559 |
| VWAP Execution in Crypto | 크립토 VWAP 최적 실행. 주문 분할/타이밍 | ★★★☆☆ 실행 품질 개선 | arXiv 2502.13722 |
| Non-Linear Dynamics in HF Crypto MM | 선형 모델 부적합 증명. 3차+ 비선형 drift | ★★☆☆☆ 이론적 기반 | arXiv 2509.02941 |

### 핵심 발견

1. **5분봉 GMADL Informer가 모든 벤치마크를 beat** (arXiv 2503.18096)
   - RMSE loss는 고빈도에서 성능 하락, GMADL loss는 반대로 향상
   - 즉, loss function 선택이 타임프레임보다 중요

2. **멀티 타임프레임 앙상블이 단일 TF보다 우월** (arXiv 2508.02356)
   - 분/시/일 3개 CNN head + soft attention → 시장 상황에 따라 동적 가중
   - 520K 파라미터로 추론 속도 최적화

3. **LightGBM이 크립토에서 cost-adjusted로도 B&H beat** (arXiv 2511.00665)
   - 비용 후: LightGBM 40% vs B&H 43% (근소 열세) vs LSTM 53% (승)
   - 하지만 LightGBM은 학습/추론이 매우 빠르고 해석 가능

4. **앙상블 + 다중 validation이 과적합 방지 핵심** (arXiv 2309.00626)
   - 단일 모델 DRL은 OOS에서 실패율 높음
   - Mixture distribution policy로 다수 모델 결합 시 일반화 개선

### 실험 결과: Option C (LightGBM 시그널 앙상블) — FAILED

**실험 (2026-02-23)**: `backtests/backtest_lgbm.py`
- 데이터: Binance 15분봉, 2025-03 ~ 2026-02 (32,317바 x 3종목)
- Feature: 35개 TA 지표 (RSI, MACD, BB, EMA, ATR, ADX, Stochastic, CCI 등)
- Target: 다음 4시간 수익률 방향 (binary)
- Model: LightGBM 300 trees, weekly retrain (90일 rolling window)

| 전략 | 수익률 | Sharpe | MDD | Alpha | 거래 |
|------|--------|--------|-----|-------|------|
| **Regime Blend** | **+12.5%** | **0.75** | **-10.1%** | **+12.4%** | 50 RT |
| Ensemble | +0.2% | 0.09 | -11.8% | +0.2% | 25 RT |
| LightGBM only | -6.9% | -0.30 | -22.5% | -7.0% | 174 RT |
| Buy & Hold | +0.0% | 0.31 | -41.6% | 0 | - |

**AUC 추이 (BTC/USDT, 월별):**
0.575 → 0.476 → 0.478 → 0.559 → 0.538 → 0.562 → 0.591 → 0.551 → 0.513 → 0.574 → 0.537 → 0.518
평균: **0.539** (동전던지기 0.50 대비 근소 우위, 비용 커버 불가)

**결론: LightGBM 15분봉 시그널 폐기**
- AUC 평균 0.54 < 0.55 기준선 미달
- 174 round-trip에 비용 드래그가 알파를 잡아먹음
- 규칙 기반 Regime Blend가 ML 대비 압도적 우위 (Sharpe 0.75 vs -0.30)
- **Feature importance**: volatility(64/192), MACD, ret_64가 상위 → 단기 패턴보다 중기 추세가 중요

**교훈**: 크립토 15분봉에서 ML 방향 예측은 비용 대비 알파 부족. 에이전트 기반 판단 + Thompson Sampling 적응이 정답.

## 백테스트 결과 (2026-02-22 기준)

### Crypto Regime Blend (GO)

| 지표 | 결과 |
|------|------|
| 기간 | 2023-01-01 ~ 2026-02-01 |
| 자산 | BTC-USD, ETH-USD, SOL-USD |
| **수익률** | **+382.1%** ($5K -> $24,104) |
| BTC B&H | +372.9% |
| **Alpha** | **+9.2%** |
| **Sharpe** | **1.65** |
| **MDD** | **35.2%** (vs BTC 37.0%) |
| 승률 | 53% (31/58) |
| 평균 수익/손실 | +26.7% / -8.2% (Payoff 3.3x) |
| 거래 수 | 116 (3년) |
| 레짐 비율 | trending 48%, ranging 52% |

**최적 파라미터** (162-combo sweep → 튜닝 후):
- max_exposure = 70%, position_pct = 30%, trail = 12%, dd_trigger = 15%, trail_activation = 8%, rebalance = 3일
- portfolio_dd_trigger = 12% (신규, 포트폴리오 전체 DD 분리)

**1년 백테스트 (튜닝 후, 2025-02 ~ 2026-02):**
- 수익률: +14.0% ($5K → $5,701) vs BTC B&H -21.9%
- **Alpha: +35.9%** (BTC 하락장에서 수익)
- Sharpe: 0.57, MDD: -20.2% (vs BTC -37.0%)
- 승률: 44% (11W/14L), 평균 수익 +20.3% / 손실 -10.5%
- Risk Exit: 12회 (48%) — 튜닝 전 81%에서 대폭 감소

**작동 원리:**
- TRENDING 레짐: RSI 모멘텀 매수 (RSI > 50 + 양의 14d momentum)
- RANGING 레짐: 볼린저 밴드 하단 매수 (평균회귀)
- Talyxion식 DD 리스크 관리 (8% 낙폭 시 전량 매도)
- 트레일링 스톱 15% (고정 스톱로스 없음)

**과적합 경고:**
- 단일 기간(2023-2025 불마켓) 백테스트
- 162개 파라미터 조합에서 최적 선택 = 데이터 스누핑 위험
- PBO/DSR 검증 필요 (Bailey & Lopez de Prado)
- **반드시 3-6개월 페이퍼 트레이딩으로 OOS 검증**

### Equity Regime Switch (MARGINAL)

| 지표 | 결과 |
|------|------|
| 수익률 | +41.4% vs SPY +54.0% |
| Alpha | -12.7% |
| Sharpe | 0.51 |
| 거래 | 5 (4년) |
| 문제 | 2022년 TLT(채권) 폭락으로 헤지 실패 |

### 실패 기록

| 전략 | 결과 | 실패 원인 |
|------|------|-----------|
| ETF Sector Rotation (v1) | Alpha -37.4% | 스톱로스(-8%)가 불마켓 딥에서 매도 |
| Crypto Momentum (v1) | Alpha -247.3% | 진입 조건 과도, 거래 4건뿐 |
| pipeline.py 기본 | Alpha -27.3% | 개별 종목 XGBoost 과적합 |
| pipeline.py + Intelligence | Alpha -32.5% | 거래 많을수록 비용 증가 |
| Combined Signal (v2) | Alpha -266.3% | 시그널 과다 = 너무 보수적 |

## 실전 교훈 요약

1. RL은 알고리즘보다 구현 품질이 4배 중요하다.
2. 실거래에서 AI가 벤치마크를 이긴 검증된 사례는 극소수다.
3. 백테스트 성과는 PBO/DSR로 반드시 디플레이션해야 한다.
4. 에이전트 수 증가 시 전략 디케이가 가속된다.
5. 단순하고 비용 효율적인 룰 기반이 복잡한 RL보다 강건하다.
6. **Thompson Sampling은 소자본에서 full RL보다 실용적이며 이론적 보장이 있다.**
7. **하이브리드 접근(RL+전통)이 업계 트렌드이며 순수 RL은 감소 추세다.**
8. **모멘텀 + 리버전 결합이 단일 전략보다 우월** (Sharpe 1.71 blended vs ~1.0 단일)
9. **RSI는 크립토에서 모멘텀 시그널** (주식에서는 리버전 시그널)
10. **레짐 감지 + 전략 전환이 핵심** - 단일 전략은 한 레짐에서만 작동
11. **과적합 경계 필수** - 162개 파라미터 스윕에서 최적 = 데이터 스누핑 위험
12. **백테스트 Sharpe는 실전 대비 30-50% 과대** (QuantStart) - Sharpe 1.65 -> 실전 0.8-1.2
13. **Zarattini (2025): 크립토 트렌드 팔로잉 Net Sharpe > 1.5** - 가장 강력한 학술 증거
14. **모멘텀 크래시는 예측 가능** (Daniel & Moskowitz 2016) - 베어마켓 반등이 최악
15. **200d SMA는 알파가 아닌 리스크 필터** - 불마켓에서 B&H 대비 2-4% 언더퍼폼
16. **5분봉 GMADL loss가 RMSE보다 고빈도에서 우월** (arXiv 2503.18096) - loss function 선택 > 타임프레임
17. **멀티 타임프레임 앙상블 (분/시/일)이 단일 TF 대비 일관 우위** (arXiv 2508.02356)
18. **LightGBM은 크립토에서 cost-adjusted B&H와 근소 열세지만 해석성/속도 우위** (arXiv 2511.00665)
19. **3년 Regime Blend ≈ BTC B&H (+306% vs +309%)** — 일봉 리밸런싱만으로는 알파 한계
20. **LightGBM 15분봉 방향 예측 FAILED** (AUC 0.54, -6.9% return) — ML 단기 예측은 비용 대비 무가치
21. **Regime Blend 규칙 기반이 ML 대비 압도적** — Sharpe 0.75 vs -0.30, 단순함이 강건함
