# 플랫폼 비교

## 비교표

| 플랫폼 | 자산 | 입금 | API 라이브러리 | 수수료 |
|--------|------|------|---------------|--------|
| Binance | 크립토 | USDT | ccxt (107개 거래소 지원) | Spot 0.1%, Futures maker 0.02% / taker 0.05% |
| 한국투자증권 | 미국주식 | KRW | mojito2 / python-kis | 0.25% |
| Alpaca | 미국주식 | USD 송금 | alpaca-py | 무료 (페이퍼 트레이딩용) |

## ccxt 선택 이유

ccxt는 크립토 트레이딩 자동화의 사실상 표준이다.

**생태계 검증:**
- Freqtrade (46K GitHub stars): ccxt 기반 오픈소스 봇
- Hummingbot (16K GitHub stars): ccxt 기반 마켓메이킹 봇
- 107개 거래소 통합 API로 거래소 전환 비용 최소화

**테스트넷 전환:**
```python
import ccxt

exchange = ccxt.binance({
    'apiKey': 'YOUR_KEY',
    'secret': 'YOUR_SECRET',
})
exchange.set_sandbox_mode(True)  # 한 줄로 테스트넷 전환
```

페이퍼 -> 실전 전환 시 `set_sandbox_mode(False)`만 변경하면 된다.

## 플랫폼별 상세

### Binance (크립토 - 1순위)

- Futures maker 0.02%는 소규모 자본에서 결정적 우위
- USDT-M Perpetual: 레버리지 가능하나 초기에는 1x만 사용
- Testnet 제공: testnet.binancefuture.com
- ccxt로 주문/잔고/포지션 조회 통합

### 한국투자증권 (미국주식 - 2순위)

- KRW 입금 가능 (환전 자동)
- mojito2: 한투 공식 오픈API 래퍼
- python-kis: 커뮤니티 라이브러리 (더 활발한 유지보수)
- 모의투자 지원 (별도 계좌 신청 필요)
- 수수료 0.25%로 빈도 제한 필수

### Alpaca (페이퍼 전용)

- USD 송금 없이 페이퍼 트레이딩만 활용
- 미국주식 전략 검증 후 한투로 전환
- 무료이므로 초기 시그널 파이프라인 테스트에 적합

## 구현 우선순위

1. ccxt + Binance Testnet (Phase 1)
2. Alpaca 페이퍼 (Phase 3, 미국주식 시그널 검증용)
3. python-kis + 한투 모의투자 (Phase 5)
