# src/hooks/ - 훅 시스템

에이전트 실행 전후에 동작하는 훅 파이프라인.

## 번들 훅 (bundled/)
- `boot-md/` - boot.md 파일 로드 (에이전트별 부트스트랩 프롬프트)
- `session-memory/` - 세션 메모리 자동 저장/복원
- `command-logger/` - 명령어 로깅
- `soul-evil/` - 성격 훅 (evil 모드)

## 핵심 파일
- `llm-slug-generator.ts` - LLM 기반 세션 슬러그 생성

## 연동
- ← `src/gateway/hooks.ts` (게이트웨이에서 훅 실행)
- ← `src/agents/bootstrap-hooks.ts` (에이전트 부트스트랩 시 훅 실행)
