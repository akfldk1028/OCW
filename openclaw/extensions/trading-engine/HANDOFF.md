# Trading Engine Handoff — GCE VM 배포

## 현재 상태: 로컬에서 완성, 배포 전

코드는 전부 완성됨. `ENABLE_CRON=true`로 실행하면 자율 스윙 트레이딩이 동작함.
남은 작업: GCE VM에 Docker 배포.

## 시스템 요약

```
cron_decide (15분마다, 포지션 인식):
  Alpaca 현재 보유 → MarketAgent(HMM+섹터) → QuantAgent v8(XGBoost z-score)
  → RL Ensemble(PPO+A2C+SAC) → FinBERT → Synthesizer(6-signal + EXIT관리)
  → SELL-before-BUY 정렬 → Alpaca 주문 실행

cron_risk_check (5분마다, 독립):
  Alpaca 보유 → TP(+4%)/SL(-2.5%)/trailing stop → Auto-SELL

cron_weekly_retrain (일요일):
  XGBoost 18개월 재학습 + RL을 최근 스캔 상위 15종목으로 재학습
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `python/server.py` | FastAPI 서버 + 4개 크론 (decide/risk/regime/retrain) |
| `python/agents/synthesizer.py` | 6-signal 가중투표 + EXIT 관리 + SELL-before-BUY |
| `python/agents/quant_agent.py` | XGBoost v8 (13팩터 z-score, top-quartile) |
| `python/broker_alpaca.py` | Alpaca 주문 실행 + `get_positions_detail()` |
| `python/risk_manager.py` | TP/SL/trailing stop, Kelly sizing |
| `python/config.py` | SWING_EXIT_CONFIG, RISK_CONFIG, SECTOR_MAP |
| `python/backtest_pipeline.py` | 풀 파이프라인 백테스트 |
| `Dockerfile` | Docker 빌드 (python:3.11-slim + CPU PyTorch) |

## 배포: GCE VM

### 왜 VM인가 (Cloud Run 아닌 이유)
- 내부 APScheduler 크론이 상시 실행 필요 (5분/15분 간격)
- `_scan_miss_counts` 등 메모리 상태 유지 필요
- 모델 파일 로컬 디스크 저장 (GCS 불필요)
- Cloud Run은 stateless → 크론 리팩토링 + GCS 연동 필요 → 오버엔지니어링

### 추천 스펙

```
VM: e2-medium (2 vCPU, 4GB RAM)
OS: Container-Optimized OS 또는 Ubuntu 22.04
디스크: 30GB SSD (모델 + 데이터)
리전: us-east1 (NYSE 레이턴시 최소화)
비용: ~$25/월
```

GPU 불필요 — RL 학습(50K timesteps)이 CPU에서 5-15분이면 충분.

### 배포 단계

```bash
# 1. GCE VM 생성
gcloud compute instances create trading-engine \
  --machine-type=e2-medium \
  --zone=us-east1-b \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=30GB \
  --tags=trading-engine

# 2. 방화벽 (8787 포트는 내부만 — 외부 노출 불필요)
# Alpaca API는 outbound만 필요하므로 inbound 룰 불필요
# 모니터링이 필요하면 IAP tunnel 사용

# 3. VM에 SSH 접속
gcloud compute ssh trading-engine --zone=us-east1-b

# 4. Docker 빌드 & 실행
# Container-Optimized OS에는 docker가 기본 설치됨
# 소스를 GCS에 올리거나 git clone으로 가져옴

docker build -t trading-engine .
docker run -d \
  --name trading-engine \
  --restart unless-stopped \
  -p 8787:8787 \
  -e ENABLE_CRON=true \
  -e LIVE_TRADING=false \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  -e ALPACA_PAPER=true \
  -v /home/user/models:/app/models \
  -v /home/user/logs:/app/logs \
  trading-engine

# 5. 헬스체크
curl http://localhost:8787/health
```

### 환경변수

| 변수 | 값 | 설명 |
|------|---|------|
| `ENABLE_CRON` | `true` | 크론 활성화 (decide@15m, risk@5m, regime@daily, retrain@weekly) |
| `LIVE_TRADING` | `false` | false=dry_run, true=실제주문 |
| `ALPACA_API_KEY` | (비밀) | Alpaca API 키 |
| `ALPACA_SECRET_KEY` | (비밀) | Alpaca 시크릿 키 |
| `ALPACA_PAPER` | `true` | 페이퍼 트레이딩 |

### 서버 설정 변경 필요

`config.py`의 `SERVER_CONFIG`에서 host를 `0.0.0.0`으로 변경하거나,
Dockerfile의 `TRADING_SERVER_HOST=0.0.0.0` 환경변수가 이미 있으므로
`server.py`에서 이 환경변수를 읽도록 확인 필요.

현재 `server.py` 마지막:
```python
uvicorn.run(app, host=SERVER_CONFIG["host"], port=SERVER_CONFIG["port"])
```

이것을 환경변수 우선으로 변경:
```python
host = os.environ.get("TRADING_SERVER_HOST", SERVER_CONFIG["host"])
port = int(os.environ.get("TRADING_SERVER_PORT", SERVER_CONFIG["port"]))
uvicorn.run(app, host=host, port=port)
```

### 모니터링

```bash
# 로그 확인
docker logs -f trading-engine

# 헬스체크
curl localhost:8787/health

# 포지션 확인 (브로커 연결 필요)
curl localhost:8787/status

# 크론 동작 확인 — 로그에서 cron_decide, cron_risk_check 출력 확인
docker logs trading-engine 2>&1 | grep "cron_"
```

### 비밀 관리

Alpaca 키를 환경변수로 직접 전달하는 대신 GCP Secret Manager 사용 추천:
```bash
# 시크릿 생성
echo -n "your_key" | gcloud secrets create alpaca-api-key --data-file=-
echo -n "your_secret" | gcloud secrets create alpaca-secret-key --data-file=-

# VM에서 읽기 (startup script에서)
ALPACA_API_KEY=$(gcloud secrets versions access latest --secret=alpaca-api-key)
```

## 알려진 제한사항

1. **max_hold_days 미구현 (라이브)**: Alpaca API가 entry_date를 제공하지 않아 보유일수 기반 EXIT 불가. 백테스트에서만 작동.
2. **cron_risk_check에 ATR 미전달**: static threshold만 사용 (RISK_CONFIG의 use_atr_dynamic은 무시됨).
3. **5분/15분 크론 경미한 race condition**: 동시 실행 시 같은 포지션에 중복 SELL 가능. 페이퍼에선 문제 없음, 라이브 전에 lock 추가 필요.

## 다음 단계 (우선순위순)

1. **GCE VM 배포** — 위 단계 실행
2. **페이퍼 트레이딩 1주일 관찰** — dry_run=false + ALPACA_PAPER=true
3. **backtest_pipeline.py 실행** — 풀 파이프라인 수익률 검증
4. **server.py host 환경변수 처리** — 위 "서버 설정 변경 필요" 항목
5. **라이브 전 lock 추가** — cron_decide + cron_risk_check 동시 실행 방지
