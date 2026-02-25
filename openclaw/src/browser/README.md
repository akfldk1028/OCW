# src/browser/ - 브라우저 제어

Playwright 기반 전용 Chrome/Chromium 브라우저 제어.

## 기능
- 웹 페이지 스냅샷 촬영
- DOM 요소 클릭/입력
- 파일 업로드
- 브라우저 프로필 관리
- AI 에이전트가 웹을 탐색할 때 사용

## 연동
- ← `src/agents/pi-tools.ts` (browse 도구로 호출)
- ← `src/gateway/server-browser.ts` (브라우저 서버 시작)
