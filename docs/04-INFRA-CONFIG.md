# 인프라, 빌드, 배포, CI/CD

## 빌드 시스템

### 진입점 (openclaw.mjs)
```javascript
// 모듈 컴파일 캐시 최적화
// → dist/entry.js 또는 dist/entry.mjs 로드
// → 에러 핸들링 + 핫 리로드 지원
```

### tsdown 빌드 (tsdown.config.ts)
6개 빌드 설정 (defineConfig 배열):
```
1. src/index.ts                              → dist/index.js     (공개 API)
2. src/entry.ts                              → dist/entry.js     (메인 CLI 진입점)
3. src/infra/warning-filter.ts               → dist/infra/warning-filter.js
4. src/plugin-sdk/index.ts                   → dist/plugin-sdk/  (Plugin SDK)
5. src/extensionAPI.ts                       → dist/extensionAPI.js
6. src/hooks/bundled/*/handler.ts +          → dist/hooks/       (내장 훅)
   src/hooks/llm-slug-generator.ts
```
- 모두 platform: "node", NODE_ENV: "production"
- warning-filter 경로: `src/infra/warning-filter.ts` (src/ 직접 하위가 아님)

### TypeScript 설정 (tsconfig.json)
```json
{
  "compilerOptions": {
    "target": "es2023",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "lib": ["DOM", "DOM.Iterable", "ES2023", "ScriptHost"],
    "noEmit": true,
    "paths": {
      "*": ["./*"],
      "openclaw/plugin-sdk": ["./src/plugin-sdk/index.ts"]
    }
  },
  "include": ["src/**/*", "ui/**/*", "extensions/**/*"],
  "exclude": ["node_modules", "dist", "src/**/*.test.ts", "extensions/**/*.test.ts"]
}
```
- ES2023 타겟 (Node 22+)
- ESM-only (NodeNext 모듈)
- strict 모드 필수
- 경로 별칭: `*` → `./*` (프로젝트 루트 기준), `openclaw/plugin-sdk` → Plugin SDK 직접 매핑
- DOM lib 포함 (웹 UI + Lit 지원)

### pnpm 워크스페이스 (pnpm-workspace.yaml)
```yaml
packages:
  - "."
  - "packages/*"
  - "ui"
  - "extensions/*"
```

---

## 개발 명령어 (package.json scripts)

| 명령어 | 역할 |
|--------|------|
| `pnpm install` | 의존성 설치 |
| `pnpm dev` | 개발 모드 시작 |
| `pnpm build` | 프로덕션 빌드 |
| `pnpm tsgo` | Watch 모드 빌드 |
| `pnpm openclaw` | CLI 실행 |
| `pnpm check` | Lint + Format + Type-check |
| `pnpm test` | 전체 테스트 |
| `pnpm test:unit` | 유닛 테스트만 |
| `pnpm test:e2e` | E2E 테스트 |
| `pnpm test:live` | 라이브 API 테스트 |

---

## 테스트 설정

### Vitest 설정 계층

```
vitest.config.ts              ← 기본 설정 (유닛 + 통합)
├── vitest.unit.config.ts     ← 유닛 테스트 전용
├── vitest.e2e.config.ts      ← E2E 테스트 (2 workers)
├── vitest.live.config.ts     ← 라이브 테스트 (1 worker, 실제 API)
├── vitest.extensions.config.ts ← 확장 테스트
└── vitest.gateway.config.ts  ← 게이트웨이 테스트
```

### 커버리지 임계값
```
Lines:      70%
Functions:  70%
Statements: 70%
Branches:   55%
```

### 테스트 설정
- Pool: `forks` (프로세스 격리)
- 타임아웃: 120초
- Workers: CI(2-3), 로컬(4-16, CPU 수 기반)
- 커버리지: V8 엔진 네이티브

---

## Docker 배포

### Dockerfile (메인)
```dockerfile
# Multi-stage 빌드
FROM node:22-bookworm AS builder
# → pnpm install + build
FROM node:22-bookworm-slim AS runtime
# → dist/ + node_modules/ 복사
# → 2GB 메모리, 비특권 사용자
```

### Dockerfile.sandbox (경량 격리)
```dockerfile
FROM debian:bookworm-slim
# bash, curl, git, python3, ripgrep
# 비루트 사용자 (sandbox)
# 에이전트 도구 실행용
```

### Dockerfile.sandbox-browser (브라우저 자동화)
```dockerfile
# sandbox 확장
# + Chromium, X11, VNC
# 포트: 9222 (DevTools), 5900 (VNC), 6080 (noVNC)
# 브라우저 기반 에이전트 태스크용
```

### docker-compose.yml
```yaml
services:
  gateway:
    build: .
    ports: ["3000:3000"]
    volumes: ["./data:/data"]
    environment:
      - NODE_ENV=production

  cli:
    build: .
    command: openclaw chat
    depends_on: [gateway]
```

---

## 클라우드 배포

### Fly.io (fly.toml)
```toml
[build]
  dockerfile = "Dockerfile"

[[vm]]
  size = "shared-cpu-2x"
  memory = "2gb"

[mounts]
  source = "data"
  destination = "/data"

[[services]]
  internal_port = 3000
  protocol = "tcp"
  [services.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20
```

### Fly.io Private (fly.private.toml)
```toml
# 퍼블릭 인그레스 없음 (아웃바운드 전용)
# 내부 게이트웨이로 사용
```

### Render.com (render.yaml)
```yaml
services:
  - type: web
    name: openclaw
    plan: starter
    disk:
      name: data
      mountPath: /data
      sizeGB: 1
```

---

## CI/CD (.github/workflows/ci.yml)

679줄, 13개 작업의 멀티 플랫폼 CI:

```yaml
jobs:
  lint:          # Oxlint + Oxfmt 검사
  typecheck:     # TypeScript 타입 체크
  build:         # 프로덕션 빌드
  test-unit:     # 유닛 테스트 + 커버리지
  test-e2e:      # E2E 테스트
  test-extensions: # 확장 테스트
  build-android: # Android APK 빌드
  build-ios:     # iOS IPA 빌드 (macOS runner)
  build-macos:   # macOS 앱 빌드
  security:      # 의존성 보안 스캔
  docker:        # Docker 이미지 빌드
  deploy-staging: # 스테이징 배포 (Fly.io)
  deploy-prod:   # 프로덕션 배포 (수동 승인)
```

### 자동 라벨링 (.github/labeler.yml)
- 30+ 라벨 자동 부여
- 파일 변경 기반 패턴 매칭
- 채널별 라벨 (20개), 앱 라벨 (3개), 코어 라벨 (10개)

### Dependabot (.github/dependabot.yml)
```yaml
# 4개 에코시스템:
- npm (주간, 프로덕션 minor+patch만)
- GitHub Actions (주간)
- Swift Package Manager (3개 위치)
- Gradle (Android)
```

---

## 환경 변수 (.env.example)

70+ 환경변수 카테고리:

| 카테고리 | 예시 |
|----------|------|
| Gateway | `GATEWAY_PORT`, `GATEWAY_SECRET`, `DATA_DIR` |
| LLM Providers | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |
| Channels | `SLACK_BOT_TOKEN`, `DISCORD_TOKEN`, `TELEGRAM_BOT_TOKEN` |
| Voice | `TWILIO_ACCOUNT_SID`, `ELEVENLABS_API_KEY` |
| Database | `SQLITE_PATH`, `VECTOR_DB_PATH` |
| Deployment | `FLY_APP_NAME`, `NODE_ENV` |

---

## 코드 품질 도구

### Pre-commit Hook (git-hooks/pre-commit)
```bash
# 스테이징된 파일에만 실행
oxlint --staged    # 린팅
oxfmt --staged     # 포맷팅
```

### Linting (.oxlintrc.json)
```json
{
  "rules": {
    "no-unused-vars": "error",
    "no-explicit-any": "warn"
  }
}
```

### Formatting (.oxfmtrc.jsonc)
```json
{
  "printWidth": 100,
  "tabWidth": 2,
  "singleQuote": false
}
```

### Secret Detection (.detect-secrets.cfg, .secrets.baseline)
- pre-commit으로 시크릿 노출 방지
- baseline 파일로 기존 시크릿 추적
