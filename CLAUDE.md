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
- **ANTHROPIC_API_KEY 사용**: OAuth는 데이터센터에서 리프레시 불가 (Cloudflare 403)
- **Python 3.11**: venv 사용, `pip install -r binance/requirements.txt`
- `.env` 파일 절대 커밋 금지 (`.gitignore`에 이미 차단됨)

## Current State (2026-02-25)
- 0 live trades, cash holding, testnet validated
- Fly.io 배포됨 (서브시스템 작동, Claude Agent만 API key 필요)
- Dashboard API 구현 완료, Vercel 미배포

## When Editing Trading Engine
- Working directory: `openclaw/extensions/trading-engine/python/`
- `binance/` 내 파일은 `from crypto_config import ...` (parent path 자동 추가됨)
- `core/`, `analysis/`, `brokers/`는 parent dir에 있음
- 테스트: `python3 -m pytest binance/tests/ -v`

## Language
- 한국어 사용
