# 폴더간 의존성 & 데이터 흐름 맵

## 전체 의존성 그래프

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Android  │  │   iOS    │  │  macOS   │  │   Web UI     │    │
│  │ (apps/)  │  │ (apps/)  │  │ (apps/)  │  │   (ui/)      │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│       │              │             │               │             │
│       │         ┌────▼─────┐      │               │             │
│       │         │OpenClawKit│◄─────┘               │             │
│       │         │ (apps/)  │                       │             │
│       │         └────┬─────┘                       │             │
└───────┼──────────────┼─────────────────────────────┼─────────────┘
        │              │                             │
        │   WebSocket  │  WebSocket                  │ WebSocket
        │              │                             │ + A2UI Protocol
        ▼              ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GATEWAY LAYER (src/)                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Gateway Server                           │   │
│  │                (src/gateway/server.ts)                     │   │
│  │                                                           │   │
│  │  ┌─────────────────────┐  ┌───────────────────────────┐  │   │
│  │  │  42 RPC Handlers    │  │    Event Broadcasting     │  │   │
│  │  │ (server-methods/)   │  │  agent.*, chat.*, exec.*  │  │   │
│  │  └────────┬────────────┘  └───────────────────────────┘  │   │
│  └───────────┼───────────────────────────────────────────────┘   │
│              │                                                    │
│  ┌───────────▼───────────────────────────────────────────────┐   │
│  │                    Core Services                           │   │
│  │                                                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐   │   │
│  │  │ Agents   │ │ Config   │ │ Memory │ │   Models     │   │   │
│  │  │ System   │ │ Store    │ │ System │ │  (LLM API)   │   │   │
│  │  └────┬─────┘ └──────────┘ └────────┘ └──────────────┘   │   │
│  │       │                                                    │   │
│  │  ┌────▼─────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐   │   │
│  │  │ Skills   │ │ Sandbox  │ │  TTS   │ │   Daemon     │   │   │
│  │  │ Loader   │ │ (격리)   │ │ Engine │ │  (systemd)   │   │   │
│  │  └──────────┘ └──────────┘ └────────┘ └──────────────┘   │   │
│  └────────────────────────────────────────────────────────────┘   │
│              │                                                    │
│              │ Plugin SDK API (OpenClawPluginApi)                 │
└──────────────┼────────────────────────────────────────────────────┘
               │
┌──────────────▼────────────────────────────────────────────────────┐
│                      EXTENSION LAYER (extensions/)                 │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  Channel Plugins │  │  Tool Plugins   │  │ Auth Providers  │   │
│  │    (20개)        │  │    (5개)        │  │    (5개)        │   │
│  │                  │  │                  │  │                  │   │
│  │ Slack, Discord   │  │ Trading Engine  │  │ Google OAuth    │   │
│  │ Telegram, Signal │  │ Memory LanceDB  │  │ Copilot Proxy   │   │
│  │ WhatsApp ...     │  │ Voice Call      │  │ Qwen Auth       │   │
│  └────────┬─────────┘  └────────┬────────┘  └────────┬────────┘   │
└───────────┼──────────────────────┼─────────────────────┼───────────┘
            │                      │                     │
            ▼                      ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES                            │
│                                                                  │
│  Slack API    Discord API    Telegram Bot API    Signal Service  │
│  yfinance     OpenAI API     Anthropic API       Twilio/Telnyx  │
│  LanceDB      ElevenLabs     Google AI           GitHub API     │
└─────────────────────────────────────────────────────────────────┘
```

## 의존 방향 상세

### 1. 클라이언트 → 게이트웨이

```
apps/android/  ─── WebSocket/HTTP ──→  src/gateway/
apps/ios/      ─── WebSocket/HTTP ──→  src/gateway/
apps/macos/    ─── WebSocket/HTTP ──→  src/gateway/
ui/            ─── WebSocket + A2UI ─→  src/gateway/
Swabble/       ─── HTTP (voicewake) ──→  src/gateway/
```

| 클라이언트 | 프로토콜 | 연결 대상 |
|-----------|----------|----------|
| Android | WebSocket | gateway server |
| iOS/macOS | WebSocket (via OpenClawKit) | gateway server |
| Web UI | WebSocket + A2UI | gateway server |
| Swabble | HTTP POST | gateway voicewake endpoint |
| CLI | 직접 함수 호출 | gateway (in-process) |

### 2. 게이트웨이 내부 의존성 (src/ 내부)

```
openclaw.mjs (부트스트랩)
  └── dist/entry.js
       └── src/entry.ts
            ├── src/infra/env.ts (환경 정규화)
            ├── src/infra/warning-filter.ts (경고 필터)
            ├── src/process/child-process-bridge.ts (프로세스 브릿지)
            └── src/cli/run-main.ts → src/cli/program.ts (Commander.js)
                 │
                 └── src/gateway/server.ts (게이트웨이 서버)
                      ├── src/gateway/server-methods/* (42 RPC 핸들러)
                      ├── src/gateway/server-plugins.ts (플러그인 로딩)
                      ├── src/gateway/server-channels.ts (채널 관리)
                      ├── src/gateway/server-chat.ts (채팅 스트리밍)
                      ├── src/gateway/server-lanes.ts (메시지 큐)
                      ├── src/agents/ (에이전트 시스템)
                      │    ├── src/agents/skills/ ← skills/*
                      │    ├── src/agents/tools/ (내장 도구)
                      │    └── src/agents/sandbox/ (격리 실행)
                      ├── src/providers/ (LLM 프로바이더 - Anthropic, OpenAI 등)
                      ├── src/config/ (설정 스토어)
                      ├── src/memory/ (메모리)
                      ├── src/sessions/ (세션)
                      ├── src/routing/ (메시지 라우팅)
                      ├── src/tts/ (음성 합성)
                      ├── src/channels/ (채널 추상화)
                      ├── src/auto-reply/ (자동 응답)
                      ├── src/plugins/ (플러그인 런타임)
                      ├── src/plugin-sdk/ (Plugin SDK - extensions가 의존)
                      └── src/infra/ (네트워크, 포트, TLS, 진단 등)
```

**주의:** `src/models/`는 존재하지 않음. LLM 관련은 `src/providers/`.

### 3. 게이트웨이 → 확장 (Plugin SDK)

```
src/gateway/ ──── OpenClawPluginApi ────→ extensions/*/index.ts
                                              │
                  ┌───────────────────────────┤
                  │                           │
                  ▼                           ▼
           registerChannel()           registerTool()
           registerProvider()          registerGatewayMethod()
           registerCommand()           registerService()
           registerCli()               addEventListener()
           registerHttpHandler()
```

### 4. 확장 → 외부 서비스

```
extensions/trading-engine/
  └── python-bridge.ts ─── HTTP ──→ python/server.py (FastAPI :8787)
                                        ├── yfinance (시장 데이터)
                                        ├── stable-baselines3 (RL)
                                        └── ta (기술지표)

extensions/voice-call/
  └── runtime ──→ Twilio/Telnyx API (전화)
               ──→ OpenAI Realtime API (스트리밍)

extensions/memory-lancedb/
  └── vector store ──→ LanceDB (벡터 DB)
                   ──→ OpenAI Embeddings API

extensions/slack/
  └── @slack/bolt ──→ Slack API

extensions/discord/
  └── discord.js ──→ Discord API

extensions/telegram/
  └── grammy ──→ Telegram Bot API
```

### 5. 보조 폴더 의존성

```
skills/*        ←──── 읽기 ────  src/agents/skills/ (스킬 로딩)
vendor/a2ui/    ←──── import ──  src/canvas-host/a2ui/ → ui/
patches/        ←──── 적용 ────  pnpm (의존성 패치, package.json patchedDependencies)
packages/       ←──── import ──  레거시 코드 (호환성 심)
scripts/        ←──── 실행 ────  package.json scripts (빌드, 테스트, 릴리스)
docs/           ←──── 참조 ────  Mintlify 사이트 + docs.acp.md
test/setup.ts   ←──── import ──  src/ (스텁 채널 등록)
.github/        ←──── 실행 ────  GitHub Actions CI/CD (13 jobs)
git-hooks/      ←──── 실행 ────  git commit (pre-commit: Oxlint/Oxfmt)
.agents/.pi/    ←──── 참조 ────  AI 에이전트 (Copilot, Claude)
```

## 데이터 흐름

### A. 채팅 메시지 흐름

```
[사용자 메시지]
     │
     ▼
 1. 채널 수신 (extensions/slack/discord/telegram...)
     │ InboundMessage
     ▼
 2. 게이트웨이 라우팅 (src/gateway/server-chat.ts)
     │ chat.send RPC
     ▼
 3. 에이전트 실행 (src/agents/agent-runner.ts)
     │ SystemPrompt + UserMessage
     ▼
 4. LLM 호출 (src/models/anthropic.ts|openai.ts)
     │ 스트리밍 응답
     ▼
 5. 도구 실행 (필요시)
     │ 확장 도구 호출 (extensions/*/tools)
     ▼
 6. 응답 생성
     │ OutboundMessage
     ▼
 7. 채널 송신 (extensions/slack/discord/telegram...)
     │
     ▼
[사용자 수신]
```

### B. 트레이딩 예측 흐름

```
[사용자: "AAPL 예측해줘"]
     │
     ▼
 1. 에이전트가 trading_predict 도구 선택
     │
     ▼
 2. extensions/trading-engine/index.ts
     │ callApi("POST", "/predict", { tickers: ["AAPL"] })
     ▼
 3. python-bridge.ts → HTTP → python/server.py
     │
     ▼
 4. data_processor.py: yfinance에서 60일 데이터 수집
     │
     ▼
 5. 기술지표 계산 (EMA, MACD, RSI, BB, ATR...)
     │
     ▼
 6. ensemble_agent.py: PPO+A2C+DDPG 앙상블 예측
     │ Sharpe 가중 투표
     ▼
 7. risk_manager.py: 리스크 체크
     │ TP/SL, Kelly sizing, 노출 한도
     ▼
 8. 결과 반환: { actions, ensemble_weights, risk_check }
     │
     ▼
[사용자에게 예측 결과 전달]
```

### C. 스킬 로딩 흐름

```
skills/*/
     │ 파일 시스템 읽기
     ▼
src/agents/skills/ (파싱/로딩)
     │ YAML → 도구 정의 + 프롬프트
     ▼
에이전트 시스템 프롬프트 구성
     │ 기본 프롬프트 + 스킬 프롬프트 + 컨텍스트
     ▼
LLM 호출 (시스템 프롬프트로 주입)
```

### D. 채널 이중 구조 흐름 (핵심)

```
[Discord 메시지 수신]
     │
     ▼
 extensions/discord/index.ts
     │ api.registerChannel({ plugin: discordPlugin })
     │ Plugin SDK wrapper
     ▼
 src/channels/plugins/ (채널 플러그인 인터페이스)
     │
     ▼
 src/discord/ (핵심 구현)
     ├── accounts.ts (계정 관리)
     ├── monitor/ (메시지 모니터링)
     └── ... (메시지 파싱, 상태 관리)
     │
     ▼
 src/routing/ (세션/에이전트 라우팅)
     │
     ▼
 src/gateway/server-chat.ts (에이전트 호출)
```

## 요약: 계층 구조

```
Layer 1: CLIENT          apps/ | ui/ | Swabble/ | CLI
Layer 2: GATEWAY         src/gateway/ (서버 + RPC)
Layer 3: CORE SERVICES   src/agents/ | src/config/ | src/memory/ | src/models/
Layer 4: EXTENSIONS      extensions/* (Plugin SDK 통해 연결)
Layer 5: EXTERNAL        Slack API | Discord API | yfinance | OpenAI | etc.
Layer 6: SUPPORT         skills/ | docs/ | test/ | vendor/ | packages/
```

**핵심 원칙:**
- **단방향 의존**: 상위 계층 → 하위 계층 (역방향 금지)
- **Plugin SDK 경계**: 게이트웨이와 확장은 `OpenClawPluginApi` 인터페이스로만 통신 (src/plugin-sdk/ 390줄 exports)
- **확장 격리**: 각 확장은 독립적, 확장간 직접 의존 없음. plugin-only 의존성은 확장 자체 package.json에
- **채널 이중 구조**: 코어 채널 로직은 src/에, 게이트웨이 등록 wrapper는 extensions/에
- **네이티브 앱 통일**: iOS/macOS는 `apps/shared/OpenClawKit` 공유, Android는 독립
- **LLM 프로바이더**: `src/providers/`에 위치 (src/models/ 아님)
