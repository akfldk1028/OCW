# ui/ - Control UI 웹앱

Gateway에서 서빙되는 웹 기반 제어 UI.

- **프레임워크**: Vite + Lit (Web Components)
- **빌드 결과**: `dist/control-ui/`에 출력 → Gateway가 정적 파일로 서빙
- **기능**: 세션 관리, 채널 상태, 설정, WebChat

빌드: `pnpm ui:build`
개발: `pnpm ui:dev`
