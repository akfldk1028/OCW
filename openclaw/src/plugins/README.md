# src/plugins/ - 플러그인 시스템

확장 플러그인의 로딩, 레지스트리, 런타임 관리.

## 역할
- `extensions/` 폴더의 플러그인들을 발견하고 로드
- 채널 플러그인, 메모리 플러그인, 기능 플러그인 통합 관리
- 런타임 플러그인 레지스트리 제공

## 연동
- → `extensions/` (플러그인 코드 로드)
- → `src/channels/` (채널 플러그인 등록)
- ← `src/gateway/server-plugins.ts` (게이트웨이 시작 시 로드)
