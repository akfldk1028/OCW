# src/cli/ - CLI 명령어 정의

`openclaw` CLI의 명령어 구조. commander.js 기반.

## 핵심 파일
- `program.ts` - 최상위 Commander 프로그램 정의
- `run-main.ts` - CLI 실행 진입점
- `deps.ts` - CLI 의존성 주입
- `profile.ts` - 프로필 (--dev, --profile) 파싱
- `prompt.ts` - 인터랙티브 프롬프트 유틸

## 연동
- ← `src/entry.ts` (진입점에서 호출)
- → `src/gateway/` (gateway 명령)
- → `src/wizard/` (onboard 명령)
- → `src/agents/` (agent 명령)
