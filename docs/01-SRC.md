# src/ - 핵심 소스코드 아키텍처

> 50+ 서브디렉토리를 가진 메인 소스. 게이트웨이, 에이전트, 채널, CLI 전부 여기 있다.
> 출처: 실제 디렉토리 구조 + `src/README.md` + `AGENTS.md`

## 메시지 처리 핵심 흐름 (src/README.md 기준)

```
1. channels/     → 채널 추상화 (메시지 수신)
2. routing/      → 에이전트/세션 라우팅
3. gateway/      → 제어 플레인 (WebSocket 서버)
4. agents/       → AI 에이전트 런타임 (모델 호출, 도구 실행)
5. sessions/     → 세션 관리
```

## 전체 서브디렉토리 맵 (실제 구조)

### A. 진입점 & 글로벌

| 파일/폴더 | 역할 |
|-----------|------|
| `entry.ts` | CLI 진입점. 환경 정규화 → Windows argv 정리 → ExperimentalWarning 억제 (필요시 respawn) → `cli/run-main.js` 로드 |
| `index.ts` | 라이브러리 진입점. 공개 API export (loadConfig, session 관리, port 유틸 등) |
| `extensionAPI.ts` | 확장 전용 API export (에이전트 디렉토리, 모델 기본값, 세션 스토어) |
| `runtime.ts` | 런타임 환경 타입 (`RuntimeEnv`) |
| `globals.ts` | 전역 상수/타입 |
| `version.ts` | 버전 정보 |
| `utils.ts` | 범용 유틸 (E164 정규화, JSON 파싱, sleep 등) |
| `logging.ts` | 콘솔 캡처 (`enableConsoleCapture`) |
| `logger.ts` | 로거 |
| `polls.ts` | 폴링 시스템 |
| `channel-web.ts` | 웹 채널 모니터 |

### B. 게이트웨이 서버 (gateway/)

**100+ 파일.** 모든 클라이언트(앱, 채널, CLI)와의 통신 허브.

```
gateway/
├── server.ts              ← 메인 서버 (WebSocket + HTTP)
├── server.impl.ts         ← 서버 구현
├── server-methods/        ← 42개 RPC 메서드 핸들러 (아래 테이블)
├── server-methods.ts      ← 메서드 등록기
├── server-methods-list.ts ← 메서드 목록
├── server-channels.ts     ← 채널 통신 인프라
├── server-chat.ts         ← 채팅 스트리밍
├── server-cron.ts         ← 크론 실행 엔진
├── server-http.ts         ← HTTP 엔드포인트
├── server-plugins.ts      ← 플러그인 로딩/관리
├── server-lanes.ts        ← 메시지 레인 (큐 시스템)
├── server-broadcast.ts    ← 이벤트 방송
├── server-browser.ts      ← 브라우저 인스턴스 관리
├── server-discovery.ts    ← 서비스 디스커버리
├── server-maintenance.ts  ← 유지보수 모드
├── server-model-catalog.ts ← 모델 카탈로그
├── server-mobile-nodes.ts ← 모바일 노드 관리
├── server-node-events.ts  ← 노드 이벤트 처리
├── server-reload-handlers.ts ← 핫 리로드
├── server-restart-sentinel.ts ← 재시작 감시
├── server-runtime-config.ts ← 런타임 설정
├── server-runtime-state.ts ← 런타임 상태
├── server-startup.ts      ← 시작 시퀀스
├── server-startup-memory.ts ← 시작시 메모리 로드
├── server-tailscale.ts    ← Tailscale 통합
├── server-wizard-sessions.ts ← 마법사 세션
├── server-ws-runtime.ts   ← WebSocket 런타임
├── protocol/              ← 프로토콜 정의
├── server/                ← 서버 헬퍼
│
├── boot.ts               ← 게이트웨이 부팅
├── auth.ts               ← 인증
├── call.ts               ← RPC 호출 브릿지
├── client.ts             ← 게이트웨이 클라이언트
├── net.ts                ← 네트워크 유틸
├── probe.ts              ← 상태 프로브
├── origin-check.ts       ← CORS/Origin 검증
├── openai-http.ts        ← OpenAI 호환 HTTP API
├── openresponses-http.ts ← OpenResponses API
│
├── assistant-identity.ts ← 어시스턴트 아이덴티티
├── chat-abort.ts         ← 채팅 중단
├── chat-attachments.ts   ← 첨부파일 처리
├── chat-sanitize.ts      ← 채팅 입력 정제
├── config-reload.ts      ← 설정 핫 리로드
├── control-ui.ts         ← 웹 UI 제공
├── device-auth.ts        ← 장치 인증
├── exec-approval-manager.ts ← 실행 승인 관리
├── hooks.ts              ← 훅 시스템
├── hooks-mapping.ts      ← 훅 매핑
├── http-common.ts        ← HTTP 공통
├── http-utils.ts         ← HTTP 유틸
├── live-image-probe.ts   ← 이미지 프로브
├── node-command-policy.ts ← 노드 명령 정책
├── node-registry.ts      ← 노드 레지스트리
├── sessions-patch.ts     ← 세션 패치
├── sessions-resolve.ts   ← 세션 해석
├── session-utils.ts/.fs.ts ← 세션 유틸
├── tools-invoke-http.ts  ← 도구 HTTP 호출
├── ws-log.ts             ← WebSocket 로깅
└── ws-logging.ts
```

**RPC Server-Methods (42개 핸들러 파일):**

| 파일 | 주요 RPC 메서드 |
|------|----------------|
| `chat.ts` | chat.send, chat.abort, chat.status, chat.inspect, chat.merge |
| `sessions.ts` | sessions.list, sessions.preview, sessions.patch, sessions.reset, sessions.delete, sessions.compact |
| `agents.ts` | agents.list, agents.search, agents.mutateBulk |
| `agent.ts` | agent.get, agent.setDefault |
| `agent-job.ts` | waitForAgentJob |
| `agent-timestamp.ts` | agent.timeStamps |
| `config.ts` | config.list, config.get, config.set, config.reset, config.restart |
| `nodes.ts` | nodes.list, nodes.ping, nodes.restart, nodes.send, nodes.invoke |
| `usage.ts` | sessions.usage (8차원 집계) |
| `tts.ts` | tts.status, tts.enable, tts.disable, tts.convert, tts.setProvider |
| `cron.ts` | cron 작업 관리 |
| `health.ts` | health.probe |
| `logs.ts` | logs.read (커서 기반 페이지네이션) |
| `skills.ts` | 스킬 발견 & 설정 |
| `channels.ts` | 채널 시작/중지 |
| `send.ts` | 메시지 송수신 |
| `connect.ts` | 연결 확인 |
| `web.ts` | web.login.start/wait (QR 코드) |
| `browser.ts` | 브라우저 인스턴스 |
| `devices.ts` | 장치 등록 |
| `exec-approval.ts` | exec.approval.request/resolve |
| `exec-approvals.ts` | 승인 파일 관리 |
| `wizard.ts` | wizard.start/next/cancel/status |
| `talk.ts` | talk.mode |
| `voicewake.ts` | voicewake.get/set |
| `models.ts` | 모델 카탈로그 |
| `update.ts` | 게이트웨이 업데이트 |
| `system.ts` | 시스템 명령어 |

### C. 에이전트 시스템 (agents/)

```
agents/
├── auth-profiles/         ← 인증 프로필 관리
├── cli-runner/            ← CLI 에이전트 실행기
├── pi-embedded-helpers/   ← Pi 내장 헬퍼
├── pi-embedded-runner/    ← Pi 내장 실행기
├── pi-extensions/         ← Pi 확장
├── sandbox/               ← 샌드박스 격리 실행
├── schema/                ← 스키마 정의 (TypeBox)
├── skills/                ← 스킬 로딩/관리
├── test-helpers/          ← 테스트 헬퍼
└── tools/                 ← 내장 도구 (exec, browse, discord-actions, slack-actions 등)
```

### D. 채널 시스템 - **이중 구조** (핵심!)

채널 코드가 **src/와 extensions/ 양쪽에** 존재하는 것이 이 프로젝트의 핵심 구조:

**src/ 내 코어 채널 구현:**
```
src/discord/      ← Discord (+ monitor/)
src/slack/        ← Slack (+ http/, monitor/)
src/telegram/     ← Telegram (+ bot/)
src/signal/       ← Signal (+ monitor/)
src/imessage/     ← iMessage (+ monitor/)
src/web/          ← WhatsApp Web (+ auto-reply/, inbound/)
src/whatsapp/     ← WhatsApp 유틸
src/line/         ← Line
src/channels/     ← 채널 추상화 레이어
  ├── allowlists/ ← 허용 목록 관리
  ├── plugins/    ← 플러그인 시스템 (타입, 설정, 온보딩, 정규화)
  └── web/        ← 웹 채널
```

**extensions/ 내 확장 채널:** (Plugin SDK를 통해 등록)
```
extensions/discord/      extensions/msteams/
extensions/slack/        extensions/matrix/
extensions/telegram/     extensions/zalo/
extensions/signal/       extensions/zalouser/
extensions/imessage/     extensions/feishu/
extensions/bluebubbles/  extensions/googlechat/
extensions/irc/          extensions/nextcloud-talk/
extensions/line/         extensions/nostr/
extensions/whatsapp/     extensions/tlon/
extensions/twitch/
```

**즉, Discord 같은 채널은:**
1. `src/discord/` — 핵심 로직 (accounts, monitor, 메시지 처리)
2. `extensions/discord/` — Plugin SDK를 통한 게이트웨이 등록 wrapper

### E. CLI & 명령어

```
src/cli/
├── program.ts             ← Commander.js 프로그램 빌드
├── run-main.ts            ← CLI 메인 실행
├── deps.ts                ← 의존성 주입 (createDefaultDeps)
├── profile.ts             ← CLI 프로필 (dev, prod)
├── prompt.ts              ← 프롬프트 유틸
├── wait.ts                ← 대기 유틸
├── browser-cli-actions-input/ ← 브라우저 CLI 액션
├── cron-cli/              ← 크론 CLI
├── daemon-cli/            ← 데몬 CLI
├── gateway-cli/           ← 게이트웨이 CLI
├── node-cli/              ← 노드 CLI
└── nodes-cli/             ← 복수 노드 CLI

src/commands/
├── agent/                 ← 에이전트 명령어
├── channels/              ← 채널 명령어
├── gateway-status/        ← 게이트웨이 상태
├── models/                ← 모델 관리
├── onboarding/            ← 온보딩 플로우
├── onboard-non-interactive/ ← 비대화형 온보딩
└── status-all/            ← 전체 상태
```

### F. 인프라 & 유틸리티

```
src/infra/          ← 저수준 인프라
  ├── format-time/  ← 시간 포맷팅
  ├── net/          ← 네트워크
  ├── outbound/     ← 아웃바운드 메시지
  ├── tls/          ← TLS
  ├── env.ts        ← 환경변수 (normalizeEnv, isTruthyEnvValue)
  ├── binaries.ts   ← 바이너리 관리
  ├── dotenv.ts     ← .env 로딩
  ├── errors.ts     ← 에러 포맷팅
  ├── ports.ts      ← 포트 관리 (PortInUseError)
  ├── warning-filter.ts ← Node 경고 필터
  ├── device-pairing.ts ← 장치 페어링
  ├── diagnostic-events.ts ← 진단 이벤트
  ├── wsl.ts        ← WSL 감지
  └── ...

src/utils/          ← 범용 유틸리티
src/shared/         ← 공유 코드
  └── text/         ← 텍스트 처리
src/types/          ← TypeScript 타입 정의
src/security/       ← 보안
src/compat/         ← 호환성 레이어
```

### G. 기능 모듈

```
src/auto-reply/       ← 자동 응답 시스템
  └── reply/          ← 응답 로직 (history, templating)
src/browser/          ← 브라우저 자동화 (Playwright)
  └── routes/         ← 브라우저 라우트
src/canvas-host/      ← Canvas 호스팅
  └── a2ui/           ← A2UI 프로토콜 번들
src/config/           ← 설정 관리
  └── sessions/       ← 세션 설정
src/cron/             ← 크론 스케줄러
  ├── isolated-agent/ ← 격리된 에이전트 실행
  └── service/        ← 크론 서비스
src/daemon/           ← 시스템 데몬 (systemd/launchd)
src/docs/             ← 문서 관련 코드
src/hooks/            ← 훅 시스템
  └── bundled/        ← 내장 훅
src/link-understanding/ ← 링크 분석/이해
src/logging/          ← 구조적 로깅
src/macos/            ← macOS 전용 코드
src/markdown/         ← 마크다운 처리
src/media/            ← 미디어 처리 (mime, store)
src/media-understanding/ ← 미디어 이해 (이미지/비디오 분석)
  └── providers/      ← 미디어 프로바이더
src/memory/           ← 메모리 시스템
src/node-host/        ← 노드 호스팅
src/pairing/          ← 장치 페어링
src/plugins/          ← 플러그인 시스템
  └── runtime/        ← 플러그인 런타임
src/plugin-sdk/       ← Plugin SDK (390줄 exports, 모든 확장이 의존)
src/process/          ← 프로세스 관리 (exec, child-process-bridge)
src/providers/        ← LLM 프로바이더 (Anthropic, OpenAI 등)
src/routing/          ← 메시지 라우팅 (session-key, resolve-route)
src/scripts/          ← 내부 스크립트
src/sessions/         ← 세션 관리
src/terminal/         ← 터미널 유틸 (ansi, links, palette, table)
src/tts/              ← Text-to-Speech (OpenAI, ElevenLabs, Edge)
src/tui/              ← Terminal UI
  ├── components/     ← UI 컴포넌트
  └── theme/          ← 테마
src/web/              ← 웹 (WhatsApp Web)
  ├── auto-reply/     ← 웹 자동 응답
  └── inbound/        ← 웹 인바운드
src/wizard/           ← 설정 마법사

src/test-helpers/     ← 테스트 헬퍼
src/test-utils/       ← 테스트 유틸
```

### H. plugin-sdk/ (Plugin SDK) — 확장의 핵심 인터페이스

`src/plugin-sdk/index.ts`는 390줄 이상의 re-export 모듈.
내부 여러 모듈에서 타입과 함수를 모아 **확장이 사용할 수 있는 공개 API**를 구성:

```typescript
// 확장에서 import하는 주요 타입/함수:
export type { OpenClawPluginApi, AnyAgentTool } from "../plugins/types.js";
export type { ChannelPlugin, ChannelConfigSchema } from "../channels/plugins/types.plugin.js";
export type { GatewayRequestHandler, RespondFn } from "../gateway/server-methods/types.js";
export type { PluginRuntime, RuntimeLogger } from "../plugins/runtime/types.js";
export { emptyPluginConfigSchema } from "../plugins/config-schema.js";
export { normalizePluginHttpPath } from "../plugins/http-path.js";

// 채널별 계정/설정/온보딩 함수 (Discord, Slack, Telegram, Signal, WhatsApp, Line, iMessage, BlueBubbles)
// 채널 허용목록, 멘션 게이팅, ACK 리액션
// 도구 스키마 헬퍼, 진단 이벤트, 미디어 유틸
// ... 총 200+ export
```

## tsconfig.json 경로 별칭 (실제)

```json
{
  "paths": {
    "*": ["./*"],
    "openclaw/plugin-sdk": ["./src/plugin-sdk/index.ts"]
  }
}
```
- `*` → `./*`: 모든 import가 프로젝트 루트 기준으로 해석
- `openclaw/plugin-sdk` → Plugin SDK 직접 매핑
