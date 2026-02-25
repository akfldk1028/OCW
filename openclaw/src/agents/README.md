# src/agents/ - AI 에이전트 런타임

프로젝트에서 가장 큰 모듈. AI 모델 호출, 도구 실행, 스킬 관리를 모두 담당합니다.

## AI 모델 실행
- `pi-embedded-runner.ts` - AI 모델 호출 (Anthropic/OpenAI/Google/Bedrock 등)
- `pi-embedded-subscribe.ts` - 스트리밍 응답 구독 + 청킹
- `pi-embedded-helpers.ts` - 모델 호출 헬퍼 (에러 분류, 세션 정리 등)
- `model-selection.ts` - 모델 선택 + 페일오버 로직
- `model-auth.ts` - 모델별 인증 (API 키, OAuth)
- `auth-profiles.ts` / `auth-profiles/` - 인증 프로필 로테이션
- `model-catalog.ts` - 사용 가능 모델 카탈로그

## 도구 시스템
- `pi-tools.ts` - 에이전트 도구 정의 (browse, canvas, send 등)
- `bash-tools.ts` - Bash 실행 도구 (exec, process)
- `openclaw-tools.ts` - OpenClaw 전용 도구 (카메라, 세션, 서브에이전트)
- `channel-tools.ts` - 채널 관련 도구
- `tool-policy.ts` - 도구 실행 정책/권한

## 에이전트 관리
- `agent-scope.ts` - 에이전트 ID/워크스페이스 관리
- `workspace.ts` - 워크스페이스 격리 (설정, 스킬, boot.md)
- `system-prompt.ts` - 시스템 프롬프트 생성
- `identity.ts` - 에이전트 아이덴티티 (이름, 프로필)
- `skills.ts` - 스킬 로딩 + 프롬프트 주입
- `sandbox.ts` - Docker/로컬 샌드박스 실행 환경
- `subagent-registry.ts` - 서브에이전트 생성/관리

## 세션
- `session-slug.ts` - 세션 슬러그 생성
- `compaction.ts` - 컨텍스트 윈도우 압축
- `usage.ts` - 사용량 추적
