# scripts/ - 빌드/테스트/배포 스크립트

프로젝트의 빌드, 테스트, 배포, 유틸리티 스크립트 모음.

## 주요 스크립트
- `run-node.mjs` - openclaw CLI 실행 진입점 (tsx로 TS 직접 실행)
- `watch-node.mjs` - 개발 모드 auto-reload
- `ui.js` - UI 빌드/개발 서버 관리
- `test-parallel.mjs` - 병렬 테스트 실행
- `bundle-a2ui.sh` - A2UI 캔버스 번들링
- `package-mac-app.sh` - macOS 앱 패키징
- `codesign-mac-app.sh` - macOS 코드 서명

## 하위 폴더
- `dev/` - 개발 유틸리티
- `docker/` - Docker 관련 스크립트
- `e2e/` - E2E 테스트 스크립트
- `pre-commit/` - pre-commit 훅
- `shell-helpers/` - 셸 헬퍼
- `systemd/` - systemd 서비스 파일
