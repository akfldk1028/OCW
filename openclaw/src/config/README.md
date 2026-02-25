# src/config/ - 설정 관리

OpenClaw 설정 파일 로드, 유효성 검사, 마이그레이션.

## 핵심 파일
- `config.ts` - 설정 로드/저장 (YAML/JSON), 마이그레이션
- `sessions.ts` - 세션 스토어 관리 (세션 키, 저장 경로)
- `plugin-auto-enable.ts` - 플러그인 자동 활성화

## 설정 파일 위치
- `~/.openclaw/config.yaml` (또는 `config.json`)
- 프로필별 격리: `~/.openclaw-<profile>/`

## 연동
- → 거의 모든 모듈에서 설정을 참조
- ← `src/wizard/` (온보딩 시 설정 생성)
- ← `src/gateway/config-reload.ts` (실시간 설정 리로드)
