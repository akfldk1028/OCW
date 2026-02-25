# OpenClaw Architecture Overview

> Version: 2026.2.10 | License: MIT | Runtime: Node.js 22+ (ESM-only)

## 프로젝트 정체성

**OpenClaw**는 **멀티채널 AI 게이트웨이 플랫폼**이다.
하나의 AI 에이전트를 20개 이상 메시징 채널(Slack, Discord, Telegram, Signal 등)에 동시 배포하고, 확장 플러그인으로 기능을 무한 확장할 수 있는 구조.

## 핵심 아키텍처 다이어그램

```
                          ┌──────────────────────────────┐
                          │      Native Apps (apps/)       │
                          │  Android / iOS / macOS         │
                          │  (shared/OpenClawKit 공유)      │
                          └──────────┬───────────────────┘
                                     │ WebSocket/HTTP
                                     ▼
┌──────────┐    ┌──────────────────────────────────────────────────┐
│  Skills  │───▶│              GATEWAY SERVER (src/gateway/)        │
│ (skills/)│    │  100+ 파일, 50+ RPC 메서드                        │
└──────────┘    │                                                   │
                │  ┌────────────────────────────────────────────┐  │
                │  │         42 Server-Method 핸들러              │  │
                │  │  chat / sessions / agents / config / nodes  │  │
                │  │  tts / usage / health / cron / wizard ...   │  │
                │  └──────────────────┬─────────────────────────┘  │
                │                     │                             │
                │  ┌──────────────────▼─────────────────────────┐  │
                │  │           Core Services (src/)              │  │
                │  │                                             │  │
                │  │  agents/     ← AI 에이전트 런타임            │  │
                │  │  providers/  ← LLM 프로바이더 (Anthropic 등) │  │
                │  │  channels/   ← 채널 추상화 레이어            │  │
                │  │  config/     ← 설정 스토어                   │  │
                │  │  memory/     ← 세션/대화 메모리              │  │
                │  │  routing/    ← 메시지 라우팅                 │  │
                │  │  plugins/    ← 플러그인 런타임               │  │
                │  │  plugin-sdk/ ← Plugin SDK (390줄 exports)   │  │
                │  │  infra/      ← 네트워크/TLS/포트/env 유틸   │  │
                │  └──────────────────┬─────────────────────────┘  │
                └─────────────────────┼────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │ Plugin SDK API             │
         ▼                            ▼                            ▼
┌────────────────┐  ┌────────────────────────┐  ┌────────────────────┐
│ 채널 코어 구현  │  │   Extension Plugins     │  │  채널 Extension     │
│ (src/ 내장)     │  │   (extensions/)         │  │  Plugins            │
│                 │  │                         │  │  (extensions/)      │
│ src/discord/    │  │ trading-engine          │  │ msteams             │
│ src/slack/      │  │ memory-lancedb          │  │ matrix              │
│ src/telegram/   │  │ voice-call              │  │ zalo / zalouser     │
│ src/signal/     │  │ diagnostics-otel        │  │ feishu / googlechat │
│ src/imessage/   │  │ copilot-proxy           │  │ nostr / tlon ...    │
│ src/web/ (WA)   │  │ google-*-auth           │  │                     │
│ src/line/       │  │ ...                     │  │                     │
└────────────────┘  └────────────────────────┘  └────────────────────┘
```

**핵심 구조적 특징:** 채널 코드가 `src/`와 `extensions/` **양쪽에 존재**한다.
- **코어 채널** (Discord, Slack, Telegram, Signal, iMessage, WhatsApp, Line): `src/` 내부에 직접 구현
- **확장 채널** (MS Teams, Matrix, Zalo, Feishu 등): `extensions/`에서 Plugin SDK로 등록

## 기술 스택

| 영역 | 기술 |
|------|------|
| Runtime | Node.js 22+ (ESM-only), Bun도 지원 |
| Language | TypeScript (strict, ES2023, NodeNext) |
| Package Manager | pnpm 10.23 workspaces (monorepo) |
| Build | tsdown (6 entry points) |
| Test | Vitest (forks pool) + V8 coverage (70% threshold) |
| Lint/Format | Oxlint + Oxfmt |
| Schema Validation | TypeBox + AJV + Zod |
| Web UI | Lit + Vite |
| Deployment | Docker / Fly.io / Render.com |
| Native Apps | Kotlin/Compose (Android), Swift/SwiftUI (iOS/macOS) |

## 폴더 구조 전체 맵

```
PROJECT
├── agent/            ← 프로젝트단  메인 에이전트 (WORKFLOW - Architecture)
├── agents/           ←   프로젝트단 메인에이전트 검토 (merge , prepare(, review , PR
├── src/              ← 메인 소스 (50+ 서브디렉토리, 아래 01-SRC.md 참조)
├── extensions/       ← 36개 확장 플러그인 (아래 02-EXTENSIONS.md 참조)
├── apps/             ← 네이티브 앱 (android, ios, macos, shared/OpenClawKit)
├── skills/           ← 53+ AI 스킬 정의 (YAML 프롬프트)
├── ui/               ← 웹 컨트롤 UI (Lit + Vite)
├── docs/             ← Mintlify 문서 사이트 (i18n: zh-CN)
├── vendor/           ← 벤더 코드 (a2ui 프로토콜)
├── packages/         ← 호환성 심 (clawdbot, moltbot)
├── Swabble/          ← Swift 웨이크워드 데몬 (Speech.framework)
├── test/             ← 통합/E2E 테스트
├── scripts/          ← 빌드/릴리스/테스트 스크립트 (60+ 파일)
├── patches/          ← pnpm 의존성 패치
├── dist/             ← 빌드 출력
├── assets/           ← 아이콘/이미지
├── git-hooks/        ← pre-commit (Oxlint/Oxfmt)
├── .github/          ← CI/CD, PR 라벨링, Dependabot
├── .agents/          ← AI 에이전트 워크플로우 (PR 리뷰)
├── .pi/              ← 프롬프트 엔지니어링
├── .agent/           ← 에이전트 런타임 설정
├── .vscode/          ← VS Code 설정
├── AGENTS.md         ← 개발자/AI 에이전트 가이드라인 (178줄)
├── CLAUDE.md         ← → AGENTS.md 참조
├── openclaw.mjs      ← 부트스트랩 진입점
├── package.json      ← 프로젝트 매니페스트
├── tsconfig.json     ← TypeScript 설정
├── tsdown.config.ts  ← 빌드 설정
└── vitest.config.ts  ← 테스트 설정
```

## 문서 목차

| 파일 | 내용 |
|------|------|
| [01-SRC.md](01-SRC.md) | `src/` 핵심 소스코드 (50+ 모듈 상세) |
| [02-EXTENSIONS.md](02-EXTENSIONS.md) | `extensions/` 36개 플러그인 시스템 |
| [03-APPS-PACKAGES.md](03-APPS-PACKAGES.md) | `apps/`, `packages/`, `vendor/`, `Swabble/` |
| [04-INFRA-CONFIG.md](04-INFRA-CONFIG.md) | 인프라, 빌드, 배포, CI/CD |
| [05-SKILLS-DOCS.md](05-SKILLS-DOCS.md) | `skills/`, `docs/`, `scripts/`, `.agents/`, `test/` |
| [06-DEPENDENCY-MAP.md](06-DEPENDENCY-MAP.md) | 폴더간 의존성 & 데이터 흐름 |
