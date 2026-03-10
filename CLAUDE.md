# OCW Project Instructions

## What This Is
Agent-vs-Agent crypto trading engine. Claude(Sonnet) = PRIMARY trader.
Monorepo: `openclaw/` 안에 trading engine이 있음.

## Key Paths
- **Trading Engine**: `openclaw/extensions/trading-engine/python/`
- **Entry Point**: `binance/main.py` (CLI: `--testnet --futures --leverage 3`)
- **Runner**: `binance/runner.py` (6 async loops — WS, derivatives, tracker, heartbeat, watchdog, API)
- **Claude Agent**: `core/claude_agent.py` + `core/claude_auth.py`
- **Config**: `binance/crypto_config.py` (gate, intervals) + `config.py` (risk params)
- **Dashboard API**: `binance/api.py` (FastAPI, port 8080)
- **Dashboard Frontend**: `dashboard/` (Next.js + Tailwind)
- **Deployment**: `fly.toml` (Fly.io, ocw-trader, nrt=Tokyo)

## Critical Rules
- **REST 최소화**: Binance IP ban 위험. WS 우선, REST는 bootstrap 1회만
- **ccxt >= 4.5 필수**: 2026-01-15 서명 변경
- **Max 구독 CLI 사용**: `claude --print` subprocess 호출. API key 절대 사용 금지 (main.py 시작 시 환경에서 자동 제거)
- **Python 3.11**: venv 사용, `pip install -r binance/requirements.txt`
- `.env` 파일 절대 커밋 금지 (`.gitignore`에 이미 차단됨)

## Current State (2026-02-28)
- Testnet validated, 63+ trades recorded
- Fly.io 배포됨 (서브시스템 작동)
- Dashboard API 구현 완료, Vercel 미배포

## H-TS Architecture (Hierarchical Thompson Sampling)
3-level Bayesian RL, 레짐별 독립 학습:
- **Level 0**: Meta-parameters (6 params × regime) — "어떻게 트레이딩?" (position_scale, entry_selectivity, hold_patience, trade_frequency, trend_vs_reversion, risk_aversion)
- **Level 1**: Group Betas (6 groups × regime) — "어떤 분석 그룹이 유효?"
- **Level 2**: Signal Betas (28 signals × regime) — "어떤 지표가 유효?"
- **Fallback**: 레짐 트레이드 < 10 → _GLOBAL_ 사용
- **State**: `binance/models/online_learner.json` (v4)
- **Key files**: `core/online_learner.py`, `core/claude_agent.py`, `binance/runner.py`

## When Editing Trading Engine
- Working directory: `openclaw/extensions/trading-engine/python/`
- `binance/` 내 파일은 `from crypto_config import ...` (parent path 자동 추가됨)
- `core/`, `analysis/`, `brokers/`는 parent dir에 있음
- 테스트: `python3 -m pytest binance/tests/ -v`

## Language
- 한국어 사용
