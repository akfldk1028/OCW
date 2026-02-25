# src/channels/ - 채널 추상화 계층

모든 메시징 채널의 공통 인터페이스와 레지스트리.

## 핵심 파일
- `registry.ts` - 채널 등록/메타데이터 (telegram, whatsapp, discord, slack 등)
- `session.ts` - 채널→세션 매핑
- `chat-type.ts` - 채팅 유형 (DM, 그룹, 스레드)
- `typing.ts` - 타이핑 인디케이터 관리
- `mention-gating.ts` - 그룹에서 멘션 기반 활성화
- `command-gating.ts` - 명령어 게이팅
- `allowlists/` - DM 허용 목록 관리
- `dock.ts` - 채널 도킹 (시작/중지)

## 하위 폴더
- `plugins/` - 채널 플러그인 인터페이스 (ChannelPlugin 타입 정의)
- `web/` - WebChat 채널 구현
- `allowlists/` - 채널별 허용 목록

## 연동
- ← `extensions/` (실제 채널 플러그인들이 이 인터페이스 구현)
- → `src/routing/` (수신 메시지를 라우팅으로 전달)
- ← `src/gateway/server-channels.ts` (게이트웨이가 채널 시작/관리)
