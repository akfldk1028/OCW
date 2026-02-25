# src/routing/ - 메시지 라우팅

수신 메시지를 올바른 에이전트와 세션으로 라우팅하는 모듈.

## 핵심 파일
- `resolve-route.ts` - 채널+피어+계정 → 에이전트ID+세션키 결정
- `bindings.ts` - 라우팅 바인딩 규칙 (채널/계정/피어/길드별)
- `session-key.ts` - 세션 키 생성 (에이전트ID + 채널 + 피어)

## 라우팅 우선순위
1. binding.peer (특정 사용자)
2. binding.guild (서버/그룹)
3. binding.team (팀)
4. binding.account (계정)
5. binding.channel (채널)
6. default (기본 에이전트)

## 연동
- ← `src/channels/` (메시지 수신 시 라우팅 요청)
- → `src/agents/` (결정된 에이전트로 전달)
- ← `src/config/` (라우팅 설정 로드)
