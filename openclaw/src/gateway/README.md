# src/gateway/ - 게이트웨이 서버 (핵심 제어 플레인)

OpenClaw의 심장부. WebSocket 기반 제어 플레인으로 모든 채널, 에이전트, 노드를 연결합니다.

## 핵심 파일
- `server.impl.ts` - 게이트웨이 부트스트랩 (모든 서브시스템 초기화)
- `server-chat.ts` - 채팅 이벤트 처리 (메시지 수신 → 에이전트 호출 → 응답)
- `server-channels.ts` - 채널 매니저 (채널 플러그인 시작/중지)
- `server-ws-runtime.ts` - WebSocket 핸들러 (클라이언트 연결 관리)
- `server-http.ts` - HTTP API 엔드포인트
- `server-browser.ts` - 브라우저 제어 서버
- `server-cron.ts` - 크론 스케줄러 통합
- `server-plugins.ts` - 플러그인 로딩
- `server-startup.ts` - 사이드카 서비스 시작

## 서브 폴더
- `server/` - 서버 내부 모듈 (health, TLS 등)
- `server-methods/` - WS RPC 메서드 핸들러
- `protocol/` - 프로토콜 스키마 정의

## 연동
- ← `src/channels/` (메시지 수신)
- → `src/agents/` (에이전트 호출)
- ← `ui/` (Control UI 서빙)
- ← `apps/` (네이티브 앱 WS 연결)
