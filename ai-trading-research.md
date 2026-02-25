# AI 강화학습 트레이딩 리서치 (2026-02-12)

## 핵심 논문 성과

| 논문 | arxiv ID | 방법 | 결과 |
|------|----------|------|------|
| 앙상블 DRL 전략 | 2511.12120 | PPO+A2C+DDPG 앙상블 | 샤프 1.30, 연 13%, 낙폭 -9.7% |
| LLM+RL 하이브리드 | 2508.02366 | LLM 전략가 + DDQN | 샤프 1.10 (RL단독 0.64 대비 72%↑) |
| DAPO+LLM (FinRL 2025 2위) | 2505.06408 | GRPO+DAPO+LLM 감성 | 누적수익 230.49% |
| Trade-R1 | 2601.03948 | LLM+RL 프로세스검증 | reward hacking 방지 |
| 뉴스+RL | 2510.19173 | LLM 감성 + LSTM + RL | buy-and-hold 대비 우수 |
| xLSTM+PPO | 2503.09655 | xLSTM + PPO | LSTM 대비 전지표 우수 |
| 동적 그리드 | 2506.11921 | 그리드+동적리셋 | BTC/ETH에서 수익, 코드공개 |
| Trading-R1 | 2509.11420 | LLM reasoning via RL | 구조화된 금융 추론 |
| Generating Alpha | 2601.19504 | EMA+MACD+RSI+감성+ML | 레짐 전환 적응형 |
| 뉴스+감성+RL 포트폴리오 | 2411.11059 | LLM 감성 + DQN/PPO | AAPL, LEXCX에서 우수 |

## 서베이/벤치마크

| 논문 | arxiv ID | 규모 | 핵심 결론 |
|------|----------|------|----------|
| 167편 메타분석 | 2512.10913 | 167편 | 알고리즘 8%, 구현품질 31% |
| RL in QF 서베이 | 2408.10932 | 167편, ACM | 마켓메이킹에서 RL 최강 |
| AI in 퀀트 서베이 | 2503.21422 | 종합 | DL→LLM 패러다임 전환 중 |
| StockBench | 2510.02209 | GPT-5, Claude 등 | buy-hold보다 나으나 일관성 부족 |
| AI-Trader | 2512.10971 | 6개 LLM | 일반지능≠매매능력, 리스크관리가 핵심 |
| FinRL 대회 | 2504.02281 | 2023-2025 | 실전 적용은 engineering-heavy |

## 최적 아키텍처 (논문 수렴 방향)

```
LLM (뉴스/감성/거시경제) → 방향성 시그널
       ↓
RL 앙상블 (PPO+A2C+DDPG) → 타이밍/사이징
       ↓
리스크 관리 (손절/포지션사이징) → 실행
```

## MCP 서버 맵

- 데이터: Alpha Vantage, Financial Datasets, TradingView
- 분석: MaverickMCP (29+도구), Claude Stock MCP
- 백테스트: QuantConnect MCP, MaverickMCP (VectorBT)
- 실거래: Alpaca MCP, MetaTrader MCP
- RL 프레임워크: FinRL (오픈소스)

## Alpha Arena 대회 결과 (실거래 $10K)

| 모델 | 최종잔액 | 수익률 |
|------|---------|--------|
| Qwen 3 Max | $12,231 | +22.3% |
| DeepSeek | $10,489 | +4.9% |
| Claude Sonnet 4.5 | $5,799 | -42% |
| Gemini 2.5 Pro | $5,445 | -45.5% |
| Grok 4 | $4,208 | -57.9% |

Claude 실패 원인: 100% 롱, 헤징 없음, 손절 없음

## 오픈소스 코드

- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- Dynamic Grid Trading: https://github.com/colachenkc/Dynamic-Grid-Trading
- QuantConnect MCP: https://github.com/taylorwilsdon/quantconnect-mcp
- MaverickMCP: https://github.com/wshobson/maverick-mcp
- Alpaca MCP: https://github.com/alpacahq/alpaca-mcp-server

---

# 오픈소스 프레임워크/도구 심층 리서치 (2026-02-13)

## 1. FinRL -- Financial Reinforcement Learning

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/AI4Finance-Foundation/FinRL |
| Stars | ~12,000 |
| 최신 버전 | v0.3.7 (PyPI) |
| 라이선스 | MIT |
| 활동 | 활발 -- FinRL Contest 2025 운영, FinAI Contest 2025로 확장 |

### 2024-2026 주요 변경사항
- **FinRL-DeepSeek 통합**: LLM(DeepSeek)에서 생성한 감성 점수/리스크 점수를 RL 에이전트 관측 공간에 주입. 뉴스 기반 트레이딩 시그널 생성
- **FinRL-AlphaSeek (크립토)**: 팩터 마이닝 + 앙상블 학습 2단계 파이프라인. RNN 기반 8개 강한 팩터 추출 후 DQN 기반 거래
- **FinRL-DeFi**: Uniswap v3 유동성 공급 RL 에이전트 태스크 신설
- **병렬 환경**: GPU에서 수천 개 동시 시뮬레이션으로 샘플링 병목 해소
- **FNSPID 데이터셋**: 나스닥 기업 대상 1,500만건 뉴스+주가 통합 데이터셋 (1999-2023)
- **FinRL_Crypto**: BTC, ETH, SOL 등 10개 크립토 전용 환경, CCXT/Binance API 연동

### openclaw 비교
- openclaw은 이미 PPO+A2C+DDPG 앙상블 + 감성 통합 아키텍처를 사용 중 -- FinRL의 핵심과 동일
- **채택 가능**: FinRL의 병렬 GPU 환경, DeFi/유동성 공급 에이전트, 팩터 마이닝 파이프라인

---

## 2. FinGPT -- Financial LLM

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/AI4Finance-Foundation/FinGPT |
| Stars | ~14,000 |
| HuggingFace | https://huggingface.co/FinGPT |
| 최신 모델 | FinGPT v3.3 (llama2-13b 기반), v3.2 (llama2-7b), v3.1 (chatglm2-6B) |

### 핵심 기능
- **LoRA 파인튜닝**: 뉴스/트윗 감성 데이터셋으로 경량 적응. 저비용으로 GPT-4급 감성 분석 달성
- **자동 데이터 큐레이션**: 실시간 금융 데이터 파이프라인 자동화
- **성능**: 감성 분석 F1 87.62%, 헤드라인 분류 95.50% (GPT-4 비교급). 단, 주가 방향 예측은 45-53% 수준
- **FinAgents**: 멀티모달 에이전트 (검색, 트레이딩, 신용평가) 프로토타입
- **한계**: 복잡한 추론/QA에서 GPT-4 대비 크게 열세 (EM 28.47% vs 76%)

### openclaw 비교
- openclaw의 `sentiment_scorer.py`는 키워드 사전 + LLM 게이트웨이 2단계 구조
- **채택 가능**: FinGPT v3.3을 로컬 배포하여 sentiment_scorer의 LLM 백엔드로 사용. 키워드 대비 대폭 정확도 향상 가능. 특히 크립토 뉴스 감성은 FinGPT보다는 크립토 특화 파인튜닝 필요

---

## 3. TradingAgents -- 멀티에이전트 LLM 트레이딩

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/TauricResearch/TradingAgents |
| 논문 | arxiv 2412.20138 |
| 버전 | v0.2.0 (2026년 2월) |
| 아키텍처 | LangGraph 기반 |

### 핵심 기능
- **7개 전문 에이전트**: Fundamentals Analyst, Sentiment Analyst, News Analyst, Technical Analyst, Researcher, Trader, Risk Manager
- **멀티 LLM 프로바이더**: GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x, Ollama 지원
- **ReAct 프레임워크**: 추론+행동 반복으로 의사결정
- **GPU 불필요**: LLM API 기반이므로 추론만으로 트레이딩 가능
- **태스크별 모델 선택**: 데이터 수집은 빠른 모델, 분석/결정은 깊은 추론 모델 사용
- **성능**: 누적수익, 샤프, 최대낙폭 모두에서 베이스라인 대비 우수

### openclaw 비교
- openclaw은 단일 RL 앙상블 + 단일 감성 스코어러 구조. TradingAgents는 멀티에이전트 LLM 협업 구조
- **채택 가능**: 멀티에이전트 분석 레이어를 openclaw 스킬 시스템과 결합 가능. 특히 risk-guardian 훅을 TradingAgents식 Risk Manager 에이전트로 강화 가능

---

## 4. Freqtrade + FreqAI

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/freqtrade/freqtrade |
| Stars | ~46,000 (가장 인기) |
| 최신 버전 | 2026.1 (2026년 1월 31일) |
| FreqAI | 내장 ML 모듈 |

### FreqAI 핵심 기능
- **적응형 재훈련**: 라이브 환경에서 백그라운드 스레드로 지속적 모델 재훈련
- **대규모 피처 엔지니어링**: 사용자 전략에서 10,000+ 피처 빠르게 생성
- **RL 지원**: stable_baselines3 + OpenAI Gym 기반. PPO 모델 타입 지원
- **연속 학습(continual_learning)**: 이전 모델 상태에서 이어서 학습 가능 (처음부터 재학습 불필요)
- **RL 상태 정보**: 현재 수익, 포지션, 거래 지속 시간 등을 네트워크에 피드
- **액션 공간**: long entry, long exit, short entry, short exit, neutral (5개)
- **현실적 백테스트**: 주기적 재훈련을 시뮬레이션하여 역사 데이터에서 적응형 훈련 에뮬레이션

### openclaw 비교
- openclaw의 트레이닝은 일괄 배치 방식. FreqAI는 라이브 연속 재훈련 + 적응형
- **채택 가능**:
  - 연속 학습 모드 (모델을 처음부터 재훈련하지 않고 이전 체크포인트에서 계속)
  - 적응형 재훈련 (15분/1시간 주기로 최신 데이터 반영)
  - 10K+ 피처 자동 생성 파이프라인
  - 5개 이산 액션 (현재 openclaw은 연속 [-1,1])

---

## 5. Jesse + AI 통합

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/jesse-ai/jesse |
| Stars | ~7,300 |
| 웹사이트 | https://jesse.trade |
| 언어 | Python |

### AI 통합 현황
- **Optimize Mode**: AI 기반 전략 파라미터 최적화
- **JesseGPT**: GPT 기반 AI 어시스턴트 -- 전략 작성/디버깅 지원
- **핵심 철학**: AI는 전략 개발 보조 도구. 코어는 결정론적 백테스터/트레이더
- **강점**: Python 생태계 (ML 라이브러리 자유롭게 통합 가능)
- **약점**: 자체 ML 모델 통합은 없음. 외부에서 가져와야 함

### openclaw 비교
- openclaw이 Jesse보다 AI 통합이 훨씬 깊음 (RL 앙상블, 감성 분석 내장)
- **채택 가능**: Jesse의 전략 파라미터 최적화 접근법 (Optuna/베이지안 최적화)

---

## 6. 추가 주목할 프레임워크

### 6a. Hummingbot -- 마켓메이킹/아비트리지

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/hummingbot/hummingbot |
| Stars | ~16,000 |
| 특화 | 고빈도 마켓메이킹, 크로스 거래소 아비트리지 |

- **Hummingbot MCP**: AI 어시스턴트와 연동하는 MCP 서버 제공
- **Dashboard**: 멀티봇 전략 배포 웹 UI
- **Quants Lab**: Python 노트북 기반 리서치/백테스트 샌드박스
- **DEX 지원**: AMM 커넥터로 탈중앙화 거래소 지원
- **$34B+**: 작년 사용자 거래량
- **채택 가능**: DEX/AMM 커넥터, 크로스 거래소 아비트리지 전략, MCP 서버

### 6b. OctoBot -- AI 통합 트레이딩

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/Drakkar-Software/OctoBot |
| Stars | ~5,100 |
| 특화 | AI/Grid/DCA 전략 자동화 |

- OpenAI/Ollama 모델 직접 연동 가능
- TradingView 커넥터
- 소셜 인디케이터 (Google Trends, Reddit)
- 내장 백테스팅 엔진
- 15+ 거래소 지원 (Binance, Hyperliquid 등)

### 6c. NautilusTrader -- 고성능 알고트레이딩

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/nautechsystems/nautilus_trader |
| Stars | ~9,100 |
| 코어 | Rust + Python(Cython/PyO3) |

- **초저지연**: 나노초 해상도 시뮬레이션, 초당 500만 행 스트리밍
- **멀티 자산**: FX, 주식, 선물, 옵션, 크립토, DeFi, 베팅
- **고급 주문**: IOC, FOK, GTD, 아이스버그, OCO, OTO
- **채택 가능**: 실제 라이브 고빈도 트레이딩 실행 엔진으로 사용 (openclaw의 Alpaca 대체/보완)

### 6d. Graph-R1 -- 지식 하이퍼그래프 + RL 에이전트

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/LHRLAB/Graph-R1 |
| 논문 | arxiv 2507.21892 |
| 아키텍처 | GraphRAG + RL |

- 지식 하이퍼그래프로 가격, 오더북, 온체인, 소셜, 뉴스를 통합 표현
- LLM 에이전트가 RL로 최적화된 다중 턴 검색 수행
- 크립토 멀티모달 데이터에 특히 적합
- **채택 가능**: Graphiti MCP(이미 사용 중)와 결합하여 지식 그래프 기반 트레이딩 컨텍스트 구축

### 6e. PrimoGPT

| 항목 | 내용 |
|------|------|
| GitHub | https://github.com/ivebotunac/PrimoGPT |
| 구성 | PrimoGPT (NLP) + PrimoRL (DRL) |

- Transformer 기반 금융 텍스트 피처 추출
- DRL로 적응형 거래 결정
- openclaw의 LLM+RL 하이브리드 구조와 유사한 접근

---

## 7. Bitcoin 예측 정확도 SOTA (2025-2026)

### 최고 성능 방법론

| 방법 | 정확도/성능 | 비고 |
|------|-----------|------|
| Boruta + CNN-LSTM 하이브리드 | 82.44% 정확도 | 피처 선택이 핵심 |
| GRU | MAPE ~2.5% | 장기 의존성 학습에 강점 |
| LSTM | RMSE 0.0054 | 기존 표준 |
| CNN-PPO 하이브리드 | +17.9% 향상 | RL+CNN 결합 시 |
| xLSTM+PPO | LSTM 대비 전지표 우수 | 2025년 신기술 |
| Transformer (LiT) | LOB 단기 예측 최고 | 한계호가창 데이터 특화 |

### 핵심 발견
- **피처 엔지니어링 > 모델 복잡도**: "Better Inputs Matter More Than Stacking Another Hidden Layer" (arxiv 2506.05764)
- **하이브리드 모델이 단일 모델 압도**: CNN+LSTM, Transformer+CNN 조합이 단일 모델 대비 15-20% 향상
- **크립토 특수성**: 24/7 시장, 규제 미비, 글로벌 분산 주문 흐름으로 전통 금융 모델 직접 적용 어려움
- **한계**: 기존 모델이 "랜덤 추측보다 약간 나은" 수준이라는 비관적 평가도 존재. 변동성과 노이즈가 근본적 도전

---

## 8. 크립토 AI 리스크 관리 베스트 프랙티스

### 필수 구현 항목
1. **변동성 기반 포지션 사이징**: 자산별 변동성에 따라 포지션 크기 조정. 각 포지션이 포트폴리오 리스크에 균등 기여
2. **동적 손절**: 트레일링 스톱 활성화 -- 유리한 가격 이동에 자동 조정, 이익 확보
3. **켈리 기준**: 수학적 포지션 사이징 (openclaw은 이미 Quarter-Kelly 사용중 -- 적절)
4. **드로다운 한도**: 최대 드로다운 도달 시 자동 거래 중단 (openclaw: -10% 설정 -- 적절)
5. **변동성 필터**: 고변동성 시 포지션 축소 또는 거래 중단 (openclaw: VIX>35 회로차단기 -- 적절)
6. **페이퍼 트레이딩 검증**: 최소 1개월 라이브 데이터로 시그널 검증 후 실거래

### openclaw 현재 리스크 시스템 평가
- TP +1%, SL -0.5%: 보수적이지만 크립토에는 너무 좁음. BTC 일일 변동 5-10% 일반적
- Quarter-Kelly: 적절
- 트레일링 스톱: 비활성 -- **활성화 권장**
- 회로차단기: 적절 (-10% 드로다운, 연속 5회 손실)
- **누락**: ATR 기반 동적 SL/TP, 레짐 감지 기반 파라미터 전환

---

## 9. 온체인 데이터 소스 (ML용)

### Glassnode
- 800+ 온체인 메트릭 (활성 주소, 거래량, 채굴 해시레이트, 거래소 유출입, 실현 시총)
- 월렛 코호트 분석으로 장기 축적/분배 국면 식별
- **ML 활용**: 피처로 사용 시 가격 방향 예측 정확도 향상. 특히 exchange netflow가 강력한 시그널

### CryptoQuant
- 단기 유동성/파생상품 포지셔닝 특화
- 실시간 거래소 유출입, 채굴자 매도 압력, 스테이블코인 리저브
- **ML 활용**: 마켓 타이밍 피처로 최적. 거래소 유입 급증 = 매도 압력 시그널

### Santiment
- 소셜 감성 + 개발자 활동 추적
- "Social Dominance" 메트릭으로 하이프 사이클/조정 사전 감지
- Telegram, Reddit, X 데이터 통합
- **ML 활용**: 감성 피처. FinGPT보다 크립토 특화된 소셜 데이터

### Nansen
- 스마트 머니 추적 -- 고래 지갑 실시간 모니터링
- **ML 활용**: 스마트 머니 플로우를 시그널로 활용

### 통합 권장사항
- **openclaw 현재**: yfinance (OHLCV) + 뉴스 감성만 사용
- **추가 권장**: Glassnode exchange netflow + CryptoQuant derivatives data + Santiment social sentiment를 피처로 추가하면 크립토 예측 정확도 대폭 향상 예상

---

## 10. 크립토 마켓 마이크로스트럭처 연구

### 최신 연구 (2025-2026)
- **LiT (Limit Order Book Transformer)**: 한계호가창(LOB) 데이터로 단기 시장 변동 예측. Transformer + CNN 결합이 최고 성능
- **"Better Inputs > Deeper Models"** (arxiv 2506.05764): 데이터 전처리와 피처 엔지니어링이 레이어 추가보다 중요. 크립토 LOB의 높은 노이즈 때문
- **멀티스케일 분석**: 일일 거시경제 데이터 + 분 단위 오더북 스냅샷 결합이 최적
- **크립토 특수성**: 24/7 운영, 규제 미비, 글로벌 분산 주문 흐름

### openclaw 적용 가능성
- 현재 OHLCV + 기술적 지표만 사용. LOB 데이터 미활용
- **권장**: Binance/Bybit WebSocket으로 LOB depth 데이터 수집, Transformer 기반 단기 예측 모듈 추가

---

## 11. MEV (Maximal Extractable Value) + AI

### 현황 (2025-2026)
- AI 에이전트가 MEV 추출에 자율적으로 참여하는 시대 돌입
- **프론트러닝/샌드위치 공격 감지**: RNN 기반 시계열 동역학 분석으로 MEV 감지
- **PBS (Proposer-Builder Separation)**: 프로토콜 레벨 완화 전략
- **ESMA 보고서** (2025-07): EU 규제 관점에서 MEV가 시장 건전성에 미치는 영향 분석
- **AI-on-AI MEV**: 2026년 AI 에이전트 간 MEV 경쟁 심화 -- "Dark Forest 2.0"

### openclaw 적용 가능성
- DEX 트레이딩 시 MEV 보호 필요 (Flashbots Protect, MEV Blocker 등)
- MEV 감지를 위한 모니터링 모듈 추가 고려
- **현재 미해당**: openclaw은 CEX(Alpaca) 기반이므로 MEV 직접 영향 없음. 향후 DeFi 확장 시 필수

---

## 12. 크로스 거래소 아비트리지 + ML

### 현황 (2025-2026)
- 이익 마진 축소 추세 -- 시장 효율성 증가
- ML/RL로 수익 가능한 기회를 사전 예측하는 접근이 핵심
- **CCXT**: 100+ 거래소 통합 API (핵심 인프라)
- **Hummingbot**: 크로스 거래소/DEX 아비트리지 전용 전략

### ML 적용 방식
- RL(강화학습)로 자원 최적 배분 -- 어떤 아비트리지가 최고 샤프를 내는지 학습
- ML로 아비트리지 기회 사전 예측 (개선된 결정 임계값)
- **논문**: "Predicting Arbitrage Occurrences With ML" (Wiley, 2026) -- 라이브 환경에서 ML 기반 아비트리지 예측 검증

### 운영 제약
- API 레이트 제한이 핵심 병목 (Binance: 반복 위반 시 IP 차단)
- 트랜잭션 비용, 전송 시간, 슬리피지 고려 필수

### openclaw 적용 가능성
- 현재 단일 거래소(Alpaca) 사용. 아비트리지 미지원
- **확장 시**: CCXT 통합 + Hummingbot 전략 참조 + RL 기반 최적 배분

---

## 13. openclaw 대비 종합 비교 및 권장사항

### openclaw 현재 강점
1. PPO+A2C+DDPG 앙상블 -- 논문 SOTA와 동일한 접근
2. Sharpe 기반 동적 가중치 -- 적절한 앙상블 전략
3. LLM 감성 통합 -- 핵심 아키텍처 포함
4. 리스크 관리 (Kelly, TP/SL, 회로차단기) -- 기본 체계 갖춤
5. MCP 서버 생태계 (Alpaca, Graphiti, Alpha Vantage) -- 확장성 있음

### 주요 개선 기회 (우선순위순)

#### P0 (즉시 개선)
1. **크립토 파라미터 조정**: TP/SL을 크립토에 맞게 조정 (TP +3-5%, SL -2-3%). BTC 일일 변동 대비 현재 TP +1%/SL -0.5%는 너무 좁음
2. **트레일링 스톱 활성화**: `risk.default.json`에서 `trailing_stop: true` + ATR 기반 동적 조정
3. **연속 학습 모드**: 매주 처음부터 재훈련 대신 이전 체크포인트에서 계속 학습 (FreqAI 방식)

#### P1 (단기 개선)
4. **온체인 피처 추가**: Glassnode exchange netflow, CryptoQuant 파생상품 데이터를 관측 공간에 추가
5. **xLSTM 네트워크**: Actor/Critic에 xLSTM 사용 (기존 MLP 대비 시계열 처리 우수)
6. **레짐 감지**: 시장 상태(상승/하락/횡보)를 감지하여 RL 파라미터/전략 동적 전환
7. **적응형 재훈련**: 15분~1시간 주기 백그라운드 재훈련 (FreqAI 방식)

#### P2 (중기 개선)
8. **멀티에이전트 분석 레이어**: TradingAgents식 전문 분석 에이전트 (펀더멘탈/감성/기술적/리스크) 추가
9. **크립토 감성 파인튜닝**: FinGPT 기반 또는 Santiment 소셜 데이터 통합
10. **LOB 데이터 활용**: 오더북 depth 데이터 수집 + Transformer 기반 단기 예측

#### P3 (장기 비전)
11. **DEX/DeFi 확장**: Hummingbot AMM 커넥터 참조, Uniswap v3 유동성 공급 RL
12. **크로스 거래소 아비트리지**: CCXT 통합 + RL 기반 최적 배분
13. **Graph-R1 지식 그래프**: Graphiti MCP + 하이퍼그래프로 멀티모달 시장 데이터 통합
14. **MEV 보호/감지**: DEX 확장 시 Flashbots Protect 통합
15. **NautilusTrader 실행 엔진**: 고빈도 실행이 필요할 때 Rust 기반 실행 엔진 도입

### 프레임워크 스타 순위 요약

| 프레임워크 | Stars | 핵심 특화 | openclaw 관련성 |
|-----------|-------|----------|---------------|
| Freqtrade + FreqAI | ~46K | 적응형 ML 트레이딩 봇 | 높음 (연속학습, 피처 엔지니어링) |
| CCXT | ~41K | 거래소 통합 API | 중간 (멀티 거래소 확장 시) |
| Hummingbot | ~16K | 마켓메이킹/아비트리지 | 중간 (DEX/아비트리지 확장 시) |
| FinGPT | ~14K | 금융 LLM | 높음 (감성 분석 강화) |
| FinRL | ~12K | 금융 RL | 이미 사용중 (코어) |
| NautilusTrader | ~9.1K | 고성능 실행 | 낮음 (현재 불필요) |
| Jesse | ~7.3K | 크립토 백테스팅 | 낮음 |
| OctoBot | ~5.1K | AI 트레이딩 자동화 | 낮음 |
| TradingAgents | 신규 | 멀티에이전트 LLM | 높음 (아키텍처 참조) |

---
---

# Comprehensive Academic Paper Survey: AI/ML Crypto Trading (2024-2026)

> Added: 2026-02-13
> Total papers reviewed: 27
> Coverage: RL, DL, LLM Agents, Sentiment, Multi-Agent, Ensemble, DeFi, Diffusion/Foundation Models, Risk, HFT

---

## Category 1: Reinforcement Learning for Crypto Trading

### Paper 1.1: Cryptocurrency Portfolio Management with RL: SAC and DDPG
- **Authors:** Kamal Paykan
- **Year:** 2025 (November)
- **Venue:** arXiv (q-fin.CP) -- [2511.20678](https://arxiv.org/abs/2511.20678)
- **Key Contribution:** RL-based framework for dynamic cryptocurrency portfolio management using SAC and DDPG, enhanced with LSTM networks for temporal feature extraction.
- **Methods:** SAC, DDPG, LSTM-based state encoder, continuous action spaces for portfolio weight allocation, simulated trading environment with transaction costs.
- **Results:** Both RL agents outperform classical Mean-Variance (MPT) benchmark. SAC demonstrates superior risk-adjusted performance, stability, and adaptability in volatile crypto environments vs. DDPG.
- **Practical Relevance:** Directly applicable to portfolio management. SAC's entropy regularization helps exploration in volatile markets. LSTM encoder for temporal features is proven for crypto. Confirms SAC > DDPG for crypto.

### Paper 1.2: Meta-Learning Reinforcement Learning for Crypto-Return Prediction (Meta-RL-Crypto)
- **Authors:** Junqiao Wang, Zhaoyang Guan, Guanyu Liu et al. (10 authors)
- **Year:** 2025 (September)
- **Venue:** arXiv (cs.LG) -- [2509.09751](https://arxiv.org/abs/2509.09751)
- **Key Contribution:** Meta-RL-Crypto: unified transformer-based architecture combining meta-learning and RL to create a fully self-improving trading agent operating in a closed-loop without human supervision.
- **Methods:** Starting from instruction-tuned LLM, iteratively alternates between (1) meta-learning: learning from multi-modal data (on-chain + off-chain sentiment) and (2) RL: refining trading policy via reward-weighted optimization.
- **Results:** Self-improving loop yields continuously improving Sharpe ratios. Multi-modal data integration (on-chain + off-chain) captures volatility and complexity better than single-source.
- **Practical Relevance:** **Highly relevant** -- cutting edge of LLM+RL for crypto. Self-improving loop and multi-modal integration are key architectural patterns. Meta-learning addresses non-stationarity of crypto markets.

### Paper 1.3: Time-Inhomogeneous Volatility Aversion for Financial RL
- **Authors:** Federico Cacciamani, Roberto Daluiso, Marco Pinciroli et al.
- **Year:** 2026 (February)
- **Venue:** arXiv (q-fin.CP) -- [2602.12030](https://arxiv.org/abs/2602.12030)
- **Key Contribution:** Addresses gap between standard RL (maximize expected return) and financial needs (return-risk trade-off). Proposes time-inhomogeneous volatility aversion for temporal distribution of returns.
- **Methods:** Modified RL objective with time-varying risk aversion parameters for settings where temporal profile of returns matters.
- **Results:** Outperforms standard risk-aware RL generalizations in financial settings.
- **Practical Relevance:** Critical for real trading where drawdown timing matters. Time-varying risk aversion relevant for crypto's regime-dependent volatility. Should be incorporated into reward shaping.

---

## Category 2: Deep Learning and Transformers for Price Prediction

### Paper 2.1: Review of Deep Learning Models for Crypto Price Prediction
- **Authors:** Jingyang Wu, Xinyi Zhang, Fangyixuan Huang et al.
- **Year:** 2024 (May)
- **Venue:** arXiv (cs.LG) -- [2405.11431](https://arxiv.org/abs/2405.11431)
- **Key Contribution:** Comprehensive review and empirical evaluation of LSTM variants, CNN variants, and Transformer models for crypto price forecasting across BTC, ETH, BNB.
- **Methods:** LSTM, BiLSTM, GRU, CNN, TCN, Transformer with attention. Consistent experimental setup across assets.
- **Results:** **Transformer models provide best performance** for multivariate strategies. LSTM variants strong for univariate. TCN competitive with lower compute.
- **Practical Relevance:** Clear model selection guide: Transformers for multivariate, LSTM/GRU for univariate, TCN for compute-efficient alternative.

### Paper 2.2: Enhancing Price Prediction with Performer + BiLSTM
- **Authors:** Mohammad Ali Labbaf Khaniki, Mohammad Manthouri
- **Year:** 2024 (March)
- **Venue:** arXiv (q-fin.CP) -- [2403.03606](https://arxiv.org/abs/2403.03606)
- **Key Contribution:** Integrates technical indicators with Performer neural network (linear-attention Transformer via FAVOR+) and BiLSTM for crypto prediction.
- **Methods:** Performer (O(n) attention), BiLSTM, technical indicators (RSI, MACD, Bollinger Bands). BTC, ETH, LTC.
- **Results:** Performer demonstrates superior computational efficiency and scalability vs. standard multi-head attention while maintaining competitive accuracy.
- **Practical Relevance:** Performer architecture key for production needing fast inference. Technical indicators + efficient attention + BiLSTM is a practical, implementable architecture.

### Paper 2.3: Algorithmic Crypto Trading with Info-Driven Bars and Triple Barrier Labeling
- **Authors:** (Multiple)
- **Year:** 2025 (December)
- **Venue:** Financial Innovation (Springer) -- [DOI](https://link.springer.com/article/10.1186/s40854-025-00866-w)
- **Key Contribution:** Information-driven sampling (CUSUM filter, range/volume/dollar bars) + Triple Barrier labeling, departing from traditional fixed-interval candles.
- **Methods:** CUSUM filter, volume/dollar/range bars, Triple Barrier labeling (Lopez de Prado), Transformer encoder, FEDformer, Autoformer, classical ML. Tick-level BTC/ETH 2018-2023.
- **Results:** CUSUM-filtered + Triple Barrier **consistently outperforms** traditional time bars + next-bar prediction. Positive trading returns even after transaction costs.
- **Practical Relevance:** **Extremely relevant.** Information-driven bars capture microstructure better than fixed candles. Triple Barrier aligns ML target with trading objectives (TP/SL/timeout). Must-implement from Lopez de Prado's framework.

### Paper 2.4: Comprehensive Analysis of 41 ML Models for Bitcoin Trading
- **Authors:** Abdul Jabbar, Syed Qaisar Jalil
- **Year:** 2024 (July)
- **Venue:** arXiv (q-fin.TR) -- [2407.18334](https://arxiv.org/abs/2407.18334)
- **Key Contribution:** Evaluates 41 ML models (21 classifiers, 20 regressors) for BTC price prediction under various market conditions.
- **Methods:** XGBoost, Gradient Boosting, Random Forest, SVM, LSTM, GRU, CNN, Transformer variants. Bull/bear/sideways comparison.
- **Results:** Ensemble trees (XGBoost, GBM) and LSTM most consistent across regimes. No single model dominates all conditions.
- **Practical Relevance:** Confirms need for ensemble approaches and regime-aware model selection. XGBoost remains strong baseline for any ensemble.

---

## Category 3: LLM-Based Trading Agents

### Paper 3.1: CryptoTrade -- Reflective LLM Agent for Zero-Shot Crypto Trading
- **Authors:** Yuan Li, Bingqiao Luo, Qian Wang et al.
- **Year:** 2024
- **Venue:** **EMNLP 2024** (Main Conference) -- [2407.09546](https://arxiv.org/abs/2407.09546)
- **Key Contribution:** First LLM-based trading agent combining on-chain and off-chain data for crypto trading. Includes reflective mechanism refining daily decisions by analyzing prior outcomes.
- **Methods:** GPT-4 agent with: (1) on-chain data (transactions, wallet activity), (2) off-chain data (news, social), (3) market data (OHLCV), (4) reflective self-improvement. Zero-shot, no fine-tuning.
- **Results:** Significantly outperforms time-series baselines (Informer, PatchTST). Comparable to MACD across bull/sideways/bear. Successfully demonstrates "buy the rumor, sell the news" during BTC ETF approval (Jan 2024).
- **Practical Relevance:** **Landmark paper** for LLM crypto trading. On-chain + off-chain fusion and reflection directly implementable. Zero-shot = no expensive fine-tuning. Published at top NLP venue (EMNLP).

### Paper 3.2: Exploring LLM Crypto Trading Through Fact-Subjectivity Aware Reasoning
- **Authors:** Qian Wang, Yuchen Gao, Zhenheng Tang et al.
- **Year:** 2024 (October)
- **Venue:** **ICLR 2025 Financial AI Workshop** -- [2410.12464](https://arxiv.org/abs/2410.12464)
- **Key Contribution:** Discovers stronger LLMs sometimes underperform weaker ones in crypto trading. Proposes separating reasoning into factual and subjective components with dynamic weighting.
- **Methods:** GPT-3.5-turbo, GPT-4, GPT-4o, o1-mini on BTC/ETH/SOL. Factual Reasoning Agent + Subjectivity Reasoning Agent + Trade Agent with dynamic weighting.
- **Results:** Separating factual from subjective reasoning significantly improves performance. Stronger LLMs prefer facts, which hurts in sentiment-driven markets.
- **Practical Relevance:** Critical for system design: architect reasoning pipeline to separate facts from sentiment. Factual/subjective split and dynamic weighting are directly implementable patterns.

---

## Category 4: Sentiment Analysis with LLMs

### Paper 4.1: Market-Derived Financial Sentiment Analysis for Crypto Forecasting
- **Authors:** Hamid Moradi-Kamali et al.
- **Year:** 2025 (February)
- **Venue:** arXiv (cs.CE) -- [2502.14897](https://arxiv.org/abs/2502.14897)
- **Key Contribution:** Market-derived labeling approach assigning labels based on ensuing short-term price trends instead of human-annotated sentiment, capturing actual market impact of text.
- **Methods:** Context-aware language model fine-tuned with market-derived labels (price reaction within window). 227 curated high-impact BTC news events. Compared vs. VADER, FinBERT.
- **Results:** 89.6% accuracy on high-impact BTC news. Up to 11% improvement in short-term trend prediction vs. traditional sentiment benchmarks.
- **Practical Relevance:** **Key insight**: train sentiment on actual price reactions, not human labels. 89.6% on high-impact events is actionable. Directly implementable in news-signal pipeline.

### Paper 4.2: Sentiment-Aware Mean-Variance Portfolio Optimization for Crypto
- **Authors:** Qizhao Chen
- **Year:** 2025 (August)
- **Venue:** arXiv (cs.CE) -- [2508.16378](https://arxiv.org/abs/2508.16378)
- **Key Contribution:** Integrates technical indicators (RSI, SMA) with LLM-enhanced sentiment into mean-variance portfolio optimization.
- **Methods:** VADER base scoring + LLM (GPT) verification and context-aware refinement. RSI/SMA technical signals. Dynamic optimization over multiple cryptos.
- **Results:** Higher cumulative return and Sharpe vs. BTC buy-and-hold and equal-weighted crypto portfolio.
- **Practical Relevance:** LLM-refined sentiment meaningfully improves portfolio optimization. Two-stage (VADER + LLM verification) is computationally practical.

---

## Category 5: Multi-Agent Trading Systems

### Paper 5.1: LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management
- **Authors:** Yichen Luo, Yebo Feng, Jiahua Xu et al.
- **Year:** 2025 (January)
- **Venue:** arXiv (q-fin.TR) -- [2501.00826](https://arxiv.org/abs/2501.00826)
- **Key Contribution:** Multi-agent LLM system with specialized agents collaborating within/across teams for top 30 crypto portfolio management.
- **Methods:** Specialized agents for data analysis, literature integration, market research, investment decisions. Inter/intra-agent collaboration. Nov 2023 - Sep 2024.
- **Results:** Outperforms single-agent and market benchmarks. Multi-team collaboration yields more robust, explainable decisions.
- **Practical Relevance:** Multi-agent with specialized roles outperforms monolithic agents. Team-based collaboration strong architectural choice.

### Paper 5.2: An Adaptive Multi-Agent Bitcoin Trading System
- **Authors:** Aadi Singhi
- **Year:** 2025 (October)
- **Venue:** arXiv (q-fin.PM) -- [2510.08068](https://arxiv.org/abs/2510.08068)
- **Key Contribution:** Multi-agent BTC trading with LLM agents for technical analysis, sentiment, decision-making, and performance reflection. Novel verbal feedback for continuous improvement.
- **Methods:** Specialized agents: Technical Analysis, Sentiment, Decision, Reflection (verbal feedback). Backtested BTC Jul 2024 - Apr 2025.
- **Results:** 30%+ higher returns in bullish phases, 15% overall gains vs. buy-and-hold.
- **Practical Relevance:** Verbal feedback/reflection for continuous improvement is novel and practical. Agent specialization maps well to modular architecture.

### Paper 5.3: Orchestration Framework for Financial Agents (Algorithmic to Agentic Trading)
- **Authors:** Jifeng Li, Arnav Grover, Abraham Alpuerto et al.
- **Year:** 2025 (December)
- **Venue:** **NeurIPS 2025 Workshop on Generative AI in Finance** -- [2512.02227](https://arxiv.org/abs/2512.02227)
- **Key Contribution:** Orchestration framework mapping each traditional algo trading component to an AI agent. Blueprint for "agentic trading."
- **Methods:** Planner, Orchestrator, Alpha Agents, Risk Agents, Portfolio Agents, Backtest Agents, Execution Agents, Audit Agents, Memory Agent.
- **Results:** Accepted at NeurIPS. Framework democratizes financial intelligence.
- **Practical Relevance:** **Excellent architectural blueprint.** Mapping from traditional algo components to agents directly applicable to trading engine design. Memory Agent concept aligns with Graphiti MCP.

### Paper 5.4: AI-Trader -- Benchmarking Autonomous Agents in Real-Time Financial Markets
- **Authors:** Tianyu Fan et al.
- **Year:** 2025 (December)
- **Venue:** arXiv (q-fin.CP) -- [2512.10971](https://arxiv.org/abs/2512.10971)
- **Key Contribution:** First fully automated, live, data-uncontaminated evaluation benchmark for LLM agents covering US stocks, A-shares, and crypto.
- **Methods:** Real-time market data, LLM agent evaluation, standardized metrics. Highlights backtesting vs. live performance gap.
- **Practical Relevance:** Essential benchmarking reference. Data-uncontaminated methodology prevents lookahead bias. Live evaluation framework template for deployment testing.

### Paper 5.5: Modelling Crypto Markets by Multi-Agent Reinforcement Learning
- **Authors:** Johann Lussange, Stefano Vrizzi et al.
- **Year:** 2024 (February)
- **Venue:** arXiv (q-fin.CP) -- [2402.10803](https://arxiv.org/abs/2402.10803)
- **Key Contribution:** MARL model simulating crypto markets, calibrated to Binance daily prices of 153 cryptos (2018-2022). RL-based agents create emergent market dynamics.
- **Methods:** Multi-agent RL with individual learning, calibrated to real data. Each agent learns independently.
- **Results:** Reproduces statistical properties of real crypto markets (volatility clustering, fat tails).
- **Practical Relevance:** Useful for building realistic market simulator for strategy backtesting. Calibrated MARL as more realistic gym for RL training.

---

## Category 6: Ensemble Methods for Crypto Trading

### Paper 6.1: Revisiting Ensemble Methods for Stock/Crypto Trading (FinRL Contests 2023-2024)
- **Authors:** Nikolaus Holzer, Keyi Wang et al.
- **Year:** 2025 (January)
- **Venue:** arXiv (cs.CE) -- [2501.10709](https://arxiv.org/abs/2501.10709)
- **Key Contribution:** Ensemble methods with massively parallel GPU simulations for enhanced computational efficiency and robustness.
- **Methods:** Ensemble of PPO + DDPG + SAC + TD3. GPU-parallel training. Model selection across validation periods. Novel mixture distribution policy.
- **Results:** Ensemble standard deviation ~50% lower than individual agents. Annualized: BTC 1.25%, ETH 9.62%, LTC 5.73% after costs. Significant decrease in max drawdown and CVaR.
- **Practical Relevance:** **Core methodology.** Ensembling multiple RL agents reduces variance, improves robustness. GPU-parallel essential for production. 50% reduction in return volatility is highly significant for risk management.

### Paper 6.2: FinRL Contests -- Benchmarking Data-Driven Financial RL Agents
- **Authors:** Keyi Wang, Nikolaus Holzer et al.
- **Year:** 2025 (April)
- **Venue:** arXiv (cs.CE) -- [2504.02281](https://arxiv.org/abs/2504.02281)
- **Key Contribution:** Standardized benchmarks for financial RL across stock and crypto. Contest tasks include crypto ensemble and LLM+RLMF signals.
- **Methods:** Second-level LOB data for BTC. Ensemble methods + LLM signal integration. 200+ participants, 100+ institutions.
- **Results:** Hybrid RL+LLM outperforms pure RL by 15-20%. CNN-PPO shows 17.9% improvement for crypto.
- **Practical Relevance:** Standardized benchmarks and evaluation. 15-20% improvement from RL+LLM hybrid validates multi-modal design. CNN-PPO identified as strong crypto baseline.

---

## Category 7: DeFi and On-Chain Analysis with ML

### Paper 7.1: From Rules to Rewards -- RL for Interest Rate Adjustment in DeFi Lending
- **Authors:** Hanxiao Qu, Krzysztof Gogol et al.
- **Year:** 2025 (May)
- **Venue:** arXiv (cs.LG) -- [2506.00505](https://arxiv.org/abs/2506.00505)
- **Key Contribution:** Offline RL (TD3-BC) for optimizing DeFi lending interest rates, replacing rule-based logic.
- **Methods:** TD3-BC on historical Aave V2/V3 WETH/WBTC data. State: utilization, liquidity, debt.
- **Results:** Outperforms rule-based logic in responsiveness, LP profitability, stress resilience.
- **Practical Relevance:** Offline RL practical (learns from history, no risky online exploration). Applicable to yield farming optimization.

### Paper 7.2: Efficient Liquidity Provisioning in Uniswap v3 with Deep RL
- **Authors:** Haonan Xu, Alessio Brini
- **Year:** 2025 (January)
- **Venue:** **AAAI 2025 Workshop** -- [2501.07508](https://arxiv.org/abs/2501.07508)
- **Key Contribution:** PPO for optimizing concentrated liquidity provisioning in Uniswap v3.
- **Methods:** PPO, MDP for LP decisions, Uniswap v3 Ethereum subgraph data (hourly). Agent decides tick range and rebalancing timing.
- **Results:** DRL agent outperforms static/simple dynamic LP strategies in fee income and impermanent loss management.
- **Practical Relevance:** Directly applicable to DeFi yield optimization. PPO for LP management proven. Integrable into broader trading system.

### Paper 7.3: Benchmarking Classical and Quantum Models for DeFi Yield Prediction
- **Authors:** Chi-Sheng Chen, Aidan Hung-Wen Tsai
- **Year:** 2025 (July)
- **Venue:** arXiv (q-fin.ST) -- [2508.02685](https://arxiv.org/abs/2508.02685)
- **Key Contribution:** First benchmark comparing classical ML and quantum ML for DeFi yield prediction on Curve Finance.
- **Methods:** XGBoost, Random Forest, LSTM, Transformer, QNN, QSVM-QNN. 28 Curve Finance pools, 1 year.
- **Results:** Classical models (XGBoost, RF) outperform deep learning and quantum for DeFi yield prediction.
- **Practical Relevance:** Simpler models (XGBoost/RF) remain strong for DeFi yield prediction. No need to over-engineer.

### Paper 7.4: Machine Learning on Blockchain Data -- Systematic Mapping Study
- **Authors:** Georgios Palaiokrassas et al.
- **Year:** 2024 (March)
- **Venue:** arXiv (cs.CR) -- [2403.17081](https://arxiv.org/abs/2403.17081)
- **Key Contribution:** Systematic review of 211 papers on ML applied to blockchain data. Taxonomy of use cases, data types, ML methods.
- **Methods:** Literature review covering anomaly detection (43%), classification (44% of ML tasks), Ethereum (46%).
- **Results:** Key areas: address classification, anomaly detection, price prediction, smart contract analysis.
- **Practical Relevance:** Comprehensive reference for ML on blockchain data. On-chain anomaly detection and address classification are mature, practical areas.

---

## Category 8: Novel Approaches -- Diffusion Models and Foundation Models

### Paper 8.1: Exploring Diffusion Models for Generative Forecasting of Financial Charts
- **Authors:** Taegyeong Lee, Jiwon Park et al.
- **Year:** 2025 (September)
- **Venue:** arXiv (cs.AI) -- [2509.02308](https://arxiv.org/abs/2509.02308)
- **Key Contribution:** Treats time-series as images and uses text-to-image diffusion models to generate "next chart image" for trend prediction.
- **Methods:** Text-to-image diffusion adapted for chart generation. Input: current chart + instruction prompt. Output: predicted next chart.
- **Results:** Feasibility demonstrated. Captures visual patterns (H&S, double tops) hard to encode numerically.
- **Practical Relevance:** Innovative but experimental. Image-based chart pattern recognition interesting as auxiliary signal.

### Paper 8.2: FinCast -- Foundation Model for Financial Time-Series Forecasting
- **Authors:** Zhuohang Zhu, Haodong Chen et al.
- **Year:** 2025 (August)
- **Venue:** arXiv (cs.LG) -- [2508.19609](https://arxiv.org/abs/2508.19609)
- **Key Contribution:** Specialized financial foundation model handling temporal non-stationarity, multi-domain diversity, varying resolutions.
- **Methods:** Pre-trained on large-scale financial data (stocks, commodities, futures). Domain-specific tokenization. Compared vs. TimesFM (Google), Chronos-T5 (Amazon), TimesMOE.
- **Results:** Outperforms general-purpose TSFMs on financial tasks. Domain-specific pre-training significantly helps.
- **Practical Relevance:** **Important.** Domain-specific foundation model outperforms general ones. Consider using FinCast or fine-tuning TSFM on crypto data.

### Paper 8.3: Re(Visiting) Time Series Foundation Models in Finance
- **Authors:** Eghbal Rahimikia, Hao Ni, Weiguan Wang
- **Year:** 2025 (November)
- **Venue:** arXiv (q-fin.CP) -- [2511.18578](https://arxiv.org/abs/2511.18578)
- **Key Contribution:** First comprehensive empirical study of TSFMs in global financial markets.
- **Methods:** Multiple TSFMs evaluated on forecasting, portfolio optimization, and risk management.
- **Results:** TSFMs show promise for zero-shot but lag task-specific models in some settings. Gap smaller for data-rich assets.
- **Practical Relevance:** Realistic expectations for TSFMs. For crypto (limited clean data), TSFMs may offer better zero-shot than small models from scratch.

### Paper 8.4: Diffusion-Based Generative Model via Geometric Brownian Motion
- **Authors:** Gihun Kim, Sun-Yong Choi, Yeoneung Kim
- **Year:** 2025 (July)
- **Venue:** arXiv (cs.LG) -- [2507.19003](https://arxiv.org/abs/2507.19003)
- **Key Contribution:** Incorporates GBM (Black-Scholes foundation) into diffusion forward noising process, reflecting financial heteroskedasticity.
- **Methods:** Modified score-based diffusion with noise proportional to asset prices, not uniform.
- **Results:** More realistic synthetic financial data. Better captures volatility clustering and fat tails.
- **Practical Relevance:** Useful for synthetic data augmentation. GBM-informed noising produces more realistic simulated markets.

### Paper 8.5: DiffSTOCK -- Probabilistic Relational Predictions Using Diffusion Models
- **Authors:** Divyanshu Daiya, Monika Yadav, Harshit Singh Rao
- **Year:** 2024 (March)
- **Venue:** **ICASSP 2024** (IEEE) -- [2403.14063](https://arxiv.org/abs/2403.14063)
- **Key Contribution:** Denoising diffusion models for stock predictions with inter-asset relational graphs.
- **Methods:** Graph-based diffusion capturing inter-asset relationships. Probabilistic outputs.
- **Results:** Outperforms deterministic graph approaches by handling low SNR of financial data.
- **Practical Relevance:** Probabilistic output valuable for risk assessment. Graph-based inter-asset modeling applicable to crypto correlation structure.

---

## Category 9: Risk Management in Algorithmic Crypto Trading

### Paper 9.1: RL in Financial Decision Making -- Systematic Review (167 papers)
- **Authors:** Mohammad Rezoanul Hoque et al.
- **Year:** 2025 (December)
- **Venue:** arXiv (q-fin.CP) -- [2512.10913](https://arxiv.org/abs/2512.10913) (submitted to Management Science)
- **Key Contribution:** Meta-analysis of 167 articles (2017-2025) on RL for finance. Unified framework and key risk management strategies.
- **Key Risk Management Findings:**
  - Implementation quality (31%) > algorithm choice (8%) for success
  - **Reward shaping** with risk-adjusted metrics (Sharpe, Sortino, CVaR) is critical
  - **Position sizing** should be part of action space, not just direction
  - **Transaction cost modeling** must be realistic (slippage, market impact)
  - Hybrid RL + traditional quant shows 15-20% improvement
- **Practical Relevance:** **Essential reading.** Emphasis on implementation quality over algorithm complexity is the most important takeaway. Risk-adjusted rewards and realistic costs are non-negotiable.

---

## Category 10: High-Frequency Crypto Trading

### Paper 10.1: Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books
- **Authors:** Haochuan Wang
- **Year:** 2025 (June)
- **Venue:** arXiv (cs.LG) -- [2506.05764](https://arxiv.org/abs/2506.05764)
- **Key Contribution:** Challenges assumption that deeper models are better for LOB prediction. Better input engineering matters more than model complexity.
- **Methods:** BTC/USDT LOB at 100ms-multi-second (Bybit). Simple vs. deep architectures with various preprocessing.
- **Results:** "Better inputs matter more than stacking another hidden layer." Well-engineered LOB features (order imbalance, depth ratios, trade flow) outperform raw deep learning.
- **Practical Relevance:** **Critical for HFT**: invest in feature engineering before model complexity. Microstructural features more valuable than raw LOB data.

---

## Synthesis: Architecture Recommendations for a Practical Trading System

### Recommended Architecture (converged from 27 papers)

```
                    +------------------+
                    | Orchestrator     |  (Paper 5.3)
                    | Agent            |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+  +--------v--------+  +--------v--------+
| Alpha Gen     |  | Sentiment       |  | Risk Mgmt       |
| Agents        |  | Agent           |  | Agent            |
| (Papers 1.2,  |  | (Papers 4.1,   |  | (Papers 1.3,    |
|  2.3, 6.1)    |  |  3.2)           |  |  9.1)            |
+-------+-------+  +--------+--------+  +--------+--------+
        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------v---------+
                    | Portfolio Agent  |
                    | (SAC Ensemble)   |
                    | (Papers 1.1, 6.1)|
                    +--------+---------+
                             |
                    +--------v---------+
                    | Execution Agent  |
                    | (Paper 10.1)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Reflection /     |
                    | Memory Agent     |
                    | (Papers 3.1, 5.2)|
                    +------------------+
```

### Technology Stack (paper-backed recommendations)

| Component | Approach | Reference |
|-----------|----------|-----------|
| RL Algorithms | SAC (primary) + PPO + TD3 ensemble | Papers 1.1, 6.1 |
| Price Prediction | Transformer + BiLSTM | Paper 2.2 |
| Data Sampling | CUSUM + Volume Bars | Paper 2.3 |
| Label Method | Triple Barrier | Paper 2.3 |
| LLM Agent | GPT-4/Claude with reflection | Paper 3.1 |
| Sentiment | Market-derived labels + LLM | Paper 4.1 |
| Reasoning | Fact/Subjectivity separation | Paper 3.2 |
| Agent Architecture | Multi-agent with orchestrator | Paper 5.3 |
| Risk Management | Time-varying risk aversion | Paper 1.3 |
| DeFi LP | PPO for Uniswap v3 | Paper 7.2 |
| DeFi Rates | Offline RL (TD3-BC) | Paper 7.1 |
| Synthetic Data | GBM-diffusion augmentation | Paper 8.4 |
| Foundation Model | FinCast or fine-tuned TSFM | Paper 8.2 |
| Backtesting | MARL market simulation | Paper 5.5 |
| Feature Engineering | Microstructure > model depth | Paper 10.1 |
| Self-Improvement | Meta-learning closed loop | Paper 1.2 |

### Key Principles (converged findings)

1. **Implementation quality > algorithm complexity** (Paper 9.1: 31% vs 8%)
2. **Feature engineering > model depth** (Paper 10.1)
3. **Ensemble > single model** (Papers 6.1, 2.4: 50% volatility reduction)
4. **Market-derived labels > human labels** for sentiment (Paper 4.1)
5. **Multi-modal data** (on-chain + off-chain + market) essential (Papers 1.2, 3.1)
6. **Separate factual from subjective reasoning** in LLMs (Paper 3.2)
7. **Risk-adjusted rewards** (Sharpe/Sortino/CVaR) required in RL (Paper 9.1)
8. **Information-driven bars** outperform time bars (Paper 2.3)
9. **Self-improving agents** with reflection/meta-learning (Papers 1.2, 3.1, 5.2)
10. **SAC best for volatile crypto** due to entropy regularization (Paper 1.1)

---

## Quick Reference: Paper Index

| # | Short Name | ID | Year | Venue | Topic |
|---|-----------|-----|------|-------|-------|
| 1.1 | SAC-DDPG Crypto | [2511.20678](https://arxiv.org/abs/2511.20678) | 2025 | arXiv | RL |
| 1.2 | Meta-RL-Crypto | [2509.09751](https://arxiv.org/abs/2509.09751) | 2025 | arXiv | RL+Meta |
| 1.3 | Volatility Aversion RL | [2602.12030](https://arxiv.org/abs/2602.12030) | 2026 | arXiv | RL Risk |
| 2.1 | DL Crypto Review | [2405.11431](https://arxiv.org/abs/2405.11431) | 2024 | arXiv | DL Survey |
| 2.2 | Performer+BiLSTM | [2403.03606](https://arxiv.org/abs/2403.03606) | 2024 | arXiv | DL |
| 2.3 | Info Bars+Triple Barrier | [Springer](https://link.springer.com/article/10.1186/s40854-025-00866-w) | 2025 | Fin. Innovation | DL+Trading |
| 2.4 | 41 ML Models BTC | [2407.18334](https://arxiv.org/abs/2407.18334) | 2024 | arXiv | ML Survey |
| 3.1 | CryptoTrade | [2407.09546](https://arxiv.org/abs/2407.09546) | 2024 | EMNLP 2024 | LLM Agent |
| 3.2 | Fact-Subjectivity | [2410.12464](https://arxiv.org/abs/2410.12464) | 2024 | ICLR 2025 WS | LLM Agent |
| 4.1 | Market-Derived Sentiment | [2502.14897](https://arxiv.org/abs/2502.14897) | 2025 | arXiv | Sentiment |
| 4.2 | Sentiment Portfolio | [2508.16378](https://arxiv.org/abs/2508.16378) | 2025 | arXiv | Sentiment |
| 5.1 | Multi-Agent Crypto | [2501.00826](https://arxiv.org/abs/2501.00826) | 2025 | arXiv | Multi-Agent |
| 5.2 | Adaptive Multi-Agent BTC | [2510.08068](https://arxiv.org/abs/2510.08068) | 2025 | arXiv | Multi-Agent |
| 5.3 | Orchestration Framework | [2512.02227](https://arxiv.org/abs/2512.02227) | 2025 | NeurIPS 2025 WS | Multi-Agent |
| 5.4 | AI-Trader Benchmark | [2512.10971](https://arxiv.org/abs/2512.10971) | 2025 | arXiv | Benchmark |
| 5.5 | MARL Crypto Markets | [2402.10803](https://arxiv.org/abs/2402.10803) | 2024 | arXiv | Multi-Agent |
| 6.1 | Ensemble RL FinRL | [2501.10709](https://arxiv.org/abs/2501.10709) | 2025 | arXiv | Ensemble |
| 6.2 | FinRL Contests | [2504.02281](https://arxiv.org/abs/2504.02281) | 2025 | arXiv | Benchmark |
| 7.1 | RL DeFi Lending | [2506.00505](https://arxiv.org/abs/2506.00505) | 2025 | arXiv | DeFi |
| 7.2 | DRL Uniswap v3 | [2501.07508](https://arxiv.org/abs/2501.07508) | 2025 | AAAI 2025 WS | DeFi |
| 7.3 | DeFi Yield Prediction | [2508.02685](https://arxiv.org/abs/2508.02685) | 2025 | arXiv | DeFi |
| 7.4 | ML Blockchain Survey | [2403.17081](https://arxiv.org/abs/2403.17081) | 2024 | arXiv | On-Chain |
| 8.1 | Diffusion Charts | [2509.02308](https://arxiv.org/abs/2509.02308) | 2025 | arXiv | Diffusion |
| 8.2 | FinCast Foundation | [2508.19609](https://arxiv.org/abs/2508.19609) | 2025 | arXiv | Foundation |
| 8.3 | TSFMs in Finance | [2511.18578](https://arxiv.org/abs/2511.18578) | 2025 | arXiv | Foundation |
| 8.4 | GBM-Diffusion | [2507.19003](https://arxiv.org/abs/2507.19003) | 2025 | arXiv | Diffusion |
| 8.5 | DiffSTOCK | [2403.14063](https://arxiv.org/abs/2403.14063) | 2024 | ICASSP 2024 | Diffusion |
| 9.1 | RL Finance Review | [2512.10913](https://arxiv.org/abs/2512.10913) | 2025 | arXiv (Mgmt Sci) | Survey |
| 10.1 | Crypto LOB Micro | [2506.05764](https://arxiv.org/abs/2506.05764) | 2025 | arXiv | HFT |

---
---

# 트레이딩 결정 시스템 구현 상세 리서치 (2026-02-14)

> 6개 주제별 실제 구현에 필요한 수치 파라미터, 아키텍처, 임계값 중심 조사
> 논문 + 실무 소스 종합

---

## Topic 1: 자동화 트레이딩 결정 프레임워크 -- 다중 시그널 → 단일 결정

### 1A. TradingAgents (arxiv 2412.20138v7)
- **저자:** Yijia Xiao, Edward Sun, Di Luo et al.
- **날짜:** 2024-12 (v7: 2025 업데이트)
- **카테고리:** q-fin.TR
- **구조:** 7개 LLM 에이전트 (Fundamental/Sentiment/News/Technical Analyst + Bullish/Bearish Researcher + Trader + Risk Manager + Fund Manager)
- **결정 메커니즘:**
  - 각 분석가가 구조화된 리포트 생성 (자연어 → 구조화 출력)
  - Bull/Bear 연구원 n라운드 토론 → Facilitator가 우세 관점 선택
  - Risk 팀 3명 (risk-seeking, neutral, conservative) n라운드 심의
  - Fund Manager가 최종 승인/실행
- **성과 (2024 Q1, AAPL/GOOGL/AMZN):**

| 종목 | 누적수익 | 연간수익 | 샤프 | 최대낙폭 |
|------|---------|---------|------|---------|
| AAPL | 26.62% | 30.5% | 8.21 | 0.91% |
| GOOGL | 24.36% | 27.58% | 6.39 | 1.69% |
| AMZN | 23.21% | 24.90% | 5.60 | 2.11% |

- **핵심 인사이트:** 구조화 출력(리포트) + 자연어 토론(에이전트 간) 하이브리드 통신이 핵심. 긴 대화에서 "message corruption" 방지

### 1B. 오케스트레이션 프레임워크 (arxiv 2512.02227, NeurIPS 2025 WS)
- **에이전트 맵핑:**
  - Data Agent → Alpha Agent → Risk Agent → Portfolio Agent → Backtest Agent → Execution Agent → Audit Agent
  - Planner/Orchestrator + Memory Agent
- **BTC 포지션 사이징 (레짐별):**
  - Strong trend: 1.8x base
  - Breakout: 2.5x base
  - Sideways: 0.7x base
  - High volatility: 0.8x base
  - 모멘텀 트리거: 최대 +30% 증가
  - 레버리지 상한: 4.0x
  - 단일 포지션: 자본의 3-5%
- **BTC 실행 세이프가드:**
  - 변동성 기반 SL: -0.8% (스케일 0.5-1.5x)
  - 최대 DD 한도: -3.0% → 전 포지션 청산
  - 시그널 스무딩: 지수 감쇄 (alpha1=0.25, alpha2=0.15)
  - 데드밴드: 0.08 (노이즈 필터)
  - 최소 보유: 8분 (급반전 방지)
- **XGBoost 모델:**
  - 300 trees, max depth 6, learning rate 0.08
  - 24시간마다 재훈련, 7일+ 히스토리
  - 1분 선행 예측, 2분 피처-라벨 갭

### 1C. Sentiment-Based Ensemble (arxiv 2402.01441v2)
- **앙상블 구성:** DDPG + PPO + A2C
- **에이전트 전환 조건:** 기간 간 감성 변화 > beta=15 → 에이전트 교체
- **검증 스코어:** chi_i = 0.25 * Sharpe + 0.75 * Sortino
- **전환 기간:** 62일 (2개월)
- **성과:**

| 메트릭 | 감성앙상블 | 앙상블 | DDPG | DJIA |
|--------|----------|--------|------|------|
| 누적수익 | 40.10% | 29.65% | 16.87% | 17.43% |
| 연간수익 | 18.36% | 13.86% | 8.11% | 8.38% |
| 샤프 | 1.32 | 0.97 | 0.66 | 0.66% |
| 소르티노 | 1.87 | 1.34 | 0.88 | 0.90 |
| 최대낙폭 | -14.57% | -15.43% | -16.11% | -18.77% |

### 1D. 구현 가능한 시그널 결합 공식
```
1. Z-score 정규화: 각 시그널을 z-score로 변환
2. 가중 결합: final_signal = w_tech * z_tech + w_sent * z_sent + w_regime * z_regime
3. 기본 가중치: w=0.5 (기술+감성 동등, arxiv 2510.10526)
4. 신뢰도: FinBERT softmax 확률 * 시그널 방향
5. 에이전트 전환: 감성 변화 > beta=15 시 활성 에이전트 교체
6. 검증: 0.25*Sharpe + 0.75*Sortino (다운사이드 리스크 중시)
```

---

## Topic 2: FinBERT 금융 감성분석 -- 실전 베스트 프랙티스

### 2A. FinBERT 기본 사양 (ProsusAI/finBERT)
- **모델:** BERT-base를 Financial PhraseBank + TRC2 등으로 사전학습 후 감성 파인튜닝
- **출력:** softmax(negative, neutral, positive) 확률 3개
- **테스트 정확도:** 97% (Financial PhraseBank 전체 합의 데이터)
- **오분류 패턴:** 73%가 positive↔neutral 혼동, 5%만 positive↔negative (방향성 오류 적음)

### 2B. 감성 점수 계산법
```python
# 단일 문장 감성 점수
sentiment_score = P(positive) - P(negative)  # 범위: [-1, 1]

# 일별 뉴스 감성 집계 (MDPI 2025 논문 방식)
daily_features = {
    'sentiment_mean': mean(scores),    # 평균 감성
    'sentiment_max': max(scores),       # 최대 감성
    'sentiment_std': std(scores),       # 감성 분산 (의견 불일치)
}

# SHAP 기반 피처 중요도 (MDPI 2025 결과):
# FinBERT Sentiment (Mean, Max, Std): 28.6%
# Volatility Indicators (GARCH, rolling std): 21.4%
```

### 2C. FinBERT+LSTM 아키텍처 (arxiv 2306.02136v3)
- **LSTM 구조:** 100유닛 LSTM → 100유닛 LSTM → 25유닛 Dense → 1유닛 출력
- **커스텀 라벨링 (NSI):**
  - +1: 가격 상승 > +1%
  - 0: 가격 변동 +/-1% 이내
  - -1: 가격 하락 > -1%

### 2D. FinBERT+XGBoost (MDPI Mathematics 2025, 13:2747)
- **프레임워크:** FinBERT 감성 + 기술지표 + 통계지표 → XGBoost 분류기
- **대상:** S&P 500 종목, 2018-2023
- **결과:** AUC, F1, 시뮬레이션 수익 모두에서 기술지표만/사전기반 감성 대비 우수
- **SHAP 해석:** FinBERT 감성이 가장 영향력 있는 예측자 중 하나
- **핵심:** 감성의 Mean, Max, Std 3가지를 모두 피처로 사용하는 것이 중요

### 2E. 실전 권장사항
```
1. 모델 선택: ProsusAI/finBERT (HuggingFace)
2. 감성 집계: P(pos) - P(neg) 점수 → 일별 Mean/Max/Std 3피처
3. 중립 필터링: P(max_class) < 0.6이면 중립으로 처리 (노이즈 제거)
4. 시간 집계 창: 지난 10일 뉴스 (QuantConnect 방식)
5. 피처 결합: XGBoost/LightGBM에 감성 피처 + 기술지표 함께 입력
6. 주의사항: FinBERT 단독 방향 예측은 45-53% 수준. 반드시 다른 피처와 결합
7. 대안: FinGPT v3.3 (F1 87.62%), 시장가격 파생 라벨링 (89.6% 정확도, arxiv 2502.14897)
```

---

## Topic 3: 현실적 백테스팅 -- 슬리피지/수수료/거래비용 모델링

### 3A. 거래비용 구성요소
```
총 거래비용 = 수수료(Commission) + 슬리피지(Slippage) + 시장충격(Market Impact)

- 수수료: 고정 또는 거래금액 비례 (가장 모델링 쉬움)
- 슬리피지: 호가 스프레드 + 유동성 + 변동성 + 실행 알고리즘의 함수
- 시장충격: 주문 크기 / 평균 거래량의 함수 (비선형)
```

### 3B. 논문별 거래비용 파라미터

| 소스 | 수수료 | 슬리피지 | 총비용 |
|------|--------|---------|--------|
| Walk-Forward (2512.12924) | $1/거래 | 5 bps (0.05%) | $1 + 0.05% * 수량 * 가격 |
| Sentiment+RL (2510.10526) Rule-based | 0 | 0 (기본) | 5 bps 테스트 시 성과 급감 |
| Sentiment+RL (2510.10526) TD3 | 0 | 10 bps (보수적) | 10 bps |
| Orchestration BTC (2512.02227) | 미공개 | 선형 모델 | 미공개 |
| FinRL Ensemble (2501.10709) | 포함 | 포함 | 5 bps 거래 + 2 bps 슬리피지 = 7 bps |
| Quantopian 기본값 | 미포함 | 5 bps | 볼륨 10%/분 제한 |

### 3C. 슬리피지 모델 유형
```
1. 고정 모델 (가장 단순):
   slippage = 고정 bps * 거래금액
   - 일반 주식: 5-10 bps
   - 대형주: 2-5 bps
   - 크립토: 5-15 bps
   - 소형주/비유동: 20-50 bps

2. 볼륨 참여 모델 (Quantopian VolumeShareSlippage):
   slippage% = price_impact * (order_qty / bar_volume)^2
   기본값: price_impact=0.1, volume_limit=0.025 (분봉 볼륨의 2.5%)

3. 제곱 모델 (가장 정확):
   market_impact = sigma * sqrt(order_qty / ADV) * lambda
   - sigma: 일일 변동성
   - ADV: 평균 일 거래량
   - lambda: 충격 계수 (보통 0.1-0.5)

4. Almgren-Chriss 모델 (기관용):
   TC = 0.5 * gamma * sigma^2 * T + eta * (X/T)
   - gamma: 위험 회피 계수
   - eta: 일시적 시장 충격 계수
```

### 3D. Walk-Forward 백테스트 프레임워크 (arxiv 2512.12924)
```
- 훈련 창: 252일 (1년)
- 테스트 창: 63일 (1분기)
- 스텝: 63일
- 총 테스트 기간: 34개 독립 폴드
- 포지션 제한: 최대 5개 동시, 각 20% 이하, 섹터당 50% 이하
- 현금 비율: 80% 유지
- 목표 수익: 5-10% (가설 유형별)
- 손절: 3-5% (패턴별)
- 최대 보유: 30일
- 시그널 임계값:
  - 볼륨 불균형: > 0.30
  - 볼륨 비율: > 1.5
  - 가격 효율성: > 0.50
  - 20일 수익 크기: < 0.10 (안정 조건)
```

### 3E. 거래비용 민감도 (2510.10526 결과)
```
비용 0 bps → 수익 20.14%, 샤프 1.61
비용 5 bps → 수익 3.66% (82% 감소!)

결론: 거래 빈도가 높을수록 비용 모델링이 생사를 가름
일일 거래: 5 bps면 충분
분 단위 거래: 10 bps 이상 보수적 적용 필요
```

---

## Topic 4: 다중 시그널 통합 -- 레짐/모멘텀/감성/RL

### 4A. LLM+RL 감성 통합 (arxiv 2510.10526)
- **기술 시그널:** RSI(14일), MACD(12-26-9), VWAP, Garman-Klass 변동성, Volume Ratio(20일 평균 대비), 실현 변동성(5일)
- **감성 시그널:** FinGPT → Positive(+1)/Negative(-1)/Neutral(0) * confidence score → Z-score 정규화
- **시그널 결합:**
```python
# Z-score 정규화 후 가중 결합
final_signal = w_t * z_technical + (1 - w_t) * z_sentiment
# 기본 w_t = 0.5 (동등 가중)
```
- **TD3 아키텍처:**
  - FC 256유닛 x 2, ReLU
  - Learning rate: 1e-4 (Adam)
  - Discount (gamma): 0.99
  - Policy noise (sigma): 0.2, clip +/-0.5
  - Delay: 2 (actor 2스텝마다 업데이트)
  - Replay buffer: ~200,000 transitions
- **보상 함수:**
```
r_t = (V_{t+1}/V_t - 1) - turnover * c_tcost - short_exposure * c_borrow
```
- **OOS 성과 (2024.01-2025.01):**
  - Rule-based (감성=0): 샤프 1.61, 수익 20.14%
  - TD3: 샤프 1.38, 수익 23.65% (buy-hold 17.17% 대비)

### 4B. Increase Alpha AI 프레임워크 (arxiv 2509.16707v2)
- **유니버스:** 814개 미국 주식
- **출력:** 종목당 10개 방향 예측 (t+1 ~ t+10)
- **분류:** 삼진코드 (+1 long, -1 short, 0 neutral)
- **그리드 서치 최적화 (2,280 시나리오/종목):**
  - 최대 보유기간 (MHP): 1-10일
  - 이익실현 (PT): 0.1%-2.0% (0.05% 단위)
  - 손절 (SL): -4% ~ -1% (0.5% 단위)
- **포트폴리오:**
  - 사이드별 낙폭 최소 20종목 선택 (MDDEW)
  - 분기별 리밸런싱, 6분기 롤링 캘리브레이션
- **성과:**
  - 개별 시그널 정확도: 50-71% (통계적 유의)
  - MDDEW 포트폴리오: 누적 26.4%, 샤프 2.54, 최대DD 3.0%, S&P 상관 -5%
  - 승률: 56.6% (long 57.9%, short 55.4%)
  - 평균 보유: 0.98일

### 4C. 실전 통합 파이프라인
```
Step 1: 레짐 감지
  - 변동성 기반 (VIX > 25 = 고변동)
  - 추세 기반 (SMA20 > SMA50 = 상승)
  - 레짐에 따라 에이전트/파라미터 전환

Step 2: 개별 시그널 생성
  - 기술적: RSI(14), MACD(12-26-9), VWAP
  - 감성: FinBERT/FinGPT → P(pos)-P(neg) → 일별 집계
  - 모멘텀: 20일/60일 수익률
  - RL: 학습된 정책의 행동값 (-1 ~ +1)

Step 3: 정규화 + 결합
  - 각 시그널 Z-score 정규화
  - 가중 선형 결합 (기본 동등 가중)
  - 또는 TD3로 최적 가중치 학습

Step 4: 결정 + 실행
  - 결합 시그널 > threshold → BUY
  - 결합 시그널 < -threshold → SELL
  - 나머지 → HOLD
  - 신뢰도 = max(softmax 확률) 또는 시그널 일치도
```

---

## Topic 5: Alpaca API 통합 -- 실전 아키텍처

### 5A. alpaca-py SDK 핵심 패턴
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.stream import TradingStream

# 클라이언트 초기화 (paper=True for 페이퍼 트레이딩)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 브래킷 주문 (진입 + TP + SL 동시 설정)
bracket_order = MarketOrderRequest(
    symbol="SPY",
    qty=5,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest(limit_price=400),
    stop_loss=StopLossRequest(stop_price=300)
)
trading_client.submit_order(order_data=bracket_order)
```

### 5B. WebSocket 실시간 스트리밍
```python
# 주문 업데이트 스트림
trading_stream = TradingStream(API_KEY, SECRET_KEY, paper=True)

@trading_stream.subscribe('trade_updates')
async def on_trade_update(data):
    # 주문 체결/취소/부분체결 이벤트 처리
    if data.event == 'fill':
        update_position(data)
    elif data.event == 'cancelled':
        handle_cancel(data)

# 시장 데이터 스트림 (별도 연결)
# wss://paper-api.alpaca.markets/stream (페이퍼)
# wss://api.alpaca.markets/stream (라이브)
# 주의: 동시 WebSocket 연결은 1개만 허용
```

### 5C. 아키텍처 권장사항
```
1. 모듈 분리:
   - DataCollector: 히스토리컬 + 실시간 데이터 수집
   - SignalGenerator: 기술지표/감성/RL 시그널 생성
   - DecisionEngine: 시그널 결합 → 매수/매도/홀드 결정
   - OrderManager: Alpaca API 주문 제출/추적
   - RiskManager: 포지션 사이징/SL/TP 관리

2. 데이터:
   - REST: 히스토리컬 데이터, 주문 제출 (비동기)
   - WebSocket: 실시간 가격, 주문 상태 업데이트
   - 하나의 WebSocket에서 멀티 스트림 구독

3. 페이퍼 → 라이브 전환:
   - paper=True → paper=False 변경만으로 전환
   - 최소 1개월 페이퍼 트레이딩 검증 후 전환
   - 에러 핸들링: API 에러, 마켓 클로즈, 부분체결 처리 필수
```

### 5D. Alpaca 수수료 구조 (2025)
```
- 주식: 수수료 무료 (commission-free)
- 옵션: 계약당 비용 있음
- 크립토: 스프레드 기반 (명시적 수수료 없음)
- 숨은 비용: 스프레드, 체결 품질(NBBO 대비)
```

---

## Topic 6: Kelly Criterion 실전 구현

### 6A. 기본 공식
```python
# 기본 Kelly
f_star = (b * p - q) / b
# 또는 동등하게:
f_star = W - (1 - W) / R

# f_star: 자본 대비 베팅 비율
# b: 승리 시 배당률 (decimal odds)
# p: 승리 확률
# q: 패배 확률 (1 - p)
# W: 승률 (win rate)
# R: 보상/위험 비율 (avg_win / avg_loss)
```

### 6B. 분수 Kelly (Fractional Kelly)
```python
# 분수 Kelly
actual_bet = h * f_star

# h 값 권장:
# h = 0.25 (Quarter Kelly) -- 가장 보수적, 큰 드로다운 방지
# h = 0.33 (Third Kelly) -- 보수적
# h = 0.50 (Half Kelly) -- 가장 일반적 실전 사용
# h = 1.00 (Full Kelly) -- 이론적 최적, 실전에서는 위험

# 드로다운 비교 (p=0.55, 짝수 배당, 5연패 시):
# Full Kelly (f*=10%): ~40% 드로다운
# Half Kelly (f*=5%): ~22.6% 드로다운
# Quarter Kelly (f*=2.5%): ~12% 드로다운
```

### 6C. 실전 구현 (Carta & Conversano, Frontiers 2020)
```
핵심 발견:
- Kelly 포트폴리오는 평균-분산 효율적 프런티어 위에 위치
- Markowitz 접선 포트폴리오보다 기대수익 높지만 분산도 큼
- 최적 리밸런싱: 2년 롤링 윈도우 캘리브레이션
- 동적 Kelly가 정적 Kelly 대비 우수
- 레버리지/공매도 제약이 실전에서 필수
```

### 6D. 트레이딩 시스템 구현 가이드
```python
def fractional_kelly_position_size(
    win_rate: float,       # 과거 승률 (0-1)
    avg_win: float,        # 평균 이익 금액
    avg_loss: float,       # 평균 손실 금액
    kelly_fraction: float, # 분수 Kelly 계수 (0.25 권장)
    account_value: float,  # 계좌 잔액
    max_position_pct: float = 0.20  # 최대 포지션 비율
) -> float:
    """분수 Kelly 포지션 사이징"""

    R = avg_win / abs(avg_loss)  # 보상/위험 비율

    # Kelly 비율 계산
    kelly_pct = win_rate - (1 - win_rate) / R

    # 음수면 베팅하지 않음
    if kelly_pct <= 0:
        return 0.0

    # 분수 Kelly 적용
    position_pct = kelly_fraction * kelly_pct

    # 최대 포지션 제한
    position_pct = min(position_pct, max_position_pct)

    return account_value * position_pct

# 사용 예:
# win_rate=0.55, avg_win=$200, avg_loss=$150
# R = 200/150 = 1.33
# kelly = 0.55 - 0.45/1.33 = 0.55 - 0.338 = 0.212 (21.2%)
# quarter_kelly = 0.25 * 0.212 = 0.053 (5.3%)
# $100,000 계좌 → $5,300 포지션
```

### 6E. 실전 주의사항
```
1. 파라미터 추정 오류: win_rate, R 추정이 틀리면 Kelly가 공격적으로 과대 투자
   → 해결: 분수 Kelly (0.25-0.50) 사용

2. 비정상성: 시장 레짐 변화로 과거 통계가 미래에 적용 안 될 수 있음
   → 해결: 롤링 윈도우 (2년)로 파라미터 갱신

3. 다중 자산: 단순 Kelly는 단일 베팅용. 포트폴리오에서는 상관관계 고려 필요
   → 해결: Kelly 행렬 (공분산 기반) 또는 자산별 독립 적용 후 합산

4. openclaw 현재: Quarter-Kelly 사용 중 -- 적절
   → 개선: 롤링 윈도우 파라미터 갱신, 레짐별 kelly_fraction 조정
```

---

## 추가 논문 인덱스

| # | 약칭 | ID | 연도 | 주제 |
|---|------|-----|------|------|
| T1A | TradingAgents | [2412.20138](https://arxiv.org/abs/2412.20138) | 2024 | 멀티에이전트 결정 |
| T1B | 오케스트레이션 | [2512.02227](https://arxiv.org/abs/2512.02227) | 2025 | 에이전트 프레임워크 |
| T1C | 감성앙상블 | [2402.01441](https://arxiv.org/abs/2402.01441) | 2024 | 앙상블+레짐전환 |
| T2A | FinBERT+LSTM | [2306.02136](https://arxiv.org/abs/2306.02136) | 2023 | 감성분석 |
| T2B | FinBERT+XGBoost+SHAP | [MDPI 2025](https://www.mdpi.com/2227-7390/13/17/2747) | 2025 | 감성+해석성 |
| T3A | Walk-Forward | [2512.12924](https://arxiv.org/abs/2512.12924) | 2025 | 현실적 백테스트 |
| T3B | 거래비용 영향 | [ResearchGate](https://www.researchgate.net/publication/384458498) | 2024 | 슬리피지 |
| T4A | LLM+RL 감성거래 | [2510.10526](https://arxiv.org/abs/2510.10526) | 2025 | 시그널통합 |
| T4B | Increase Alpha | [2509.16707](https://arxiv.org/abs/2509.16707) | 2025 | AI 프레임워크 |
| T5A | TradingGroup | [2508.17565](https://arxiv.org/abs/2508.17565) | 2025 | 멀티에이전트 |
| T6A | Kelly 실전 | [Frontiers 2020](https://www.frontiersin.org/articles/10.3389/fams.2020.577050) | 2020 | 포지션사이징 |
