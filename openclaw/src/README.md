# src/ - 메인 소스코드

OpenClaw의 핵심 TypeScript 소스코드. 모든 서버 로직이 여기에 있습니다.

## 진입점
- `entry.ts` - CLI 진입점 (프로세스 초기화 → cli/run-main.js)
- `index.ts` - 라이브러리 진입점 (공개 API export)

## 핵심 모듈 (메시지 처리 흐름 순)
1. `channels/` → 채널 추상화 (메시지 수신)
2. `routing/` → 에이전트/세션 라우팅
3. `gateway/` → 제어 플레인 (WebSocket 서버)
4. `agents/` → AI 에이전트 런타임 (모델 호출, 도구 실행)
5. `sessions/` → 세션 관리

## 채널 구현
`whatsapp/`, `telegram/`, `discord/`, `slack/`, `signal/`, `imessage/`, `line/`, `web/`

## 인프라/유틸
`config/`, `infra/`, `utils/`, `shared/`, `types/`, `logging/`, `security/`

## 기능 모듈
`browser/`, `canvas-host/`, `cron/`, `hooks/`, `media/`, `memory/`, `tts/`, `tui/`, `wizard/`
