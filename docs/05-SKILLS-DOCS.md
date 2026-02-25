# skills/, docs/, scripts/, .agents/, .pi/, ui/, test/ - 보조 시스템
docs/agent-autonomous-architecture.md  
## skills/ - AI 스킬 정의 (53+개)

에이전트가 사용할 수 있는 "능력"을 YAML 프롬프트로 정의.
`src/agents/skills.ts`가 로딩하여 시스템 프롬프트에 주입.

### 구조 (실제 디렉토리 리스트, 카테고리 아닌 플랫 구조)
```
skills/
├── README.md
├── 1password/        ├── apple-notes/      ├── apple-reminders/
├── bear-notes/       ├── blogwatcher/      ├── blucli/
├── bluebubbles/      ├── camsnap/          ├── canvas/
├── clawhub/          ├── coding-agent/     ├── discord/
├── eightctl/         ├── food-order/       ├── gemini/
├── gifgrep/          ├── github/           ├── gog/
├── goplaces/         ├── healthcheck/      ├── himalaya/
├── imsg/             ├── local-places/     ├── mcporter/
├── model-usage/      ├── nano-banana-pro/  ├── nano-pdf/
├── notion/           ├── obsidian/         ├── openai-image-gen/
├── openai-whisper/   ├── openai-whisper-api/├── openhue/
├── oracle/           ├── ordercli/         ├── peekaboo/
├── sag/              ├── session-logs/     ├── sherpa-onnx-tts/
├── skill-creator/    ├── slack/            ├── songsee/
├── sonoscli/         ├── spotify-player/   ├── summarize/
├── things-mac/       ├── tmux/             ├── trello/
├── video-frames/     ├── voice-call/       ├── wacli/
└── weather/
```
**참고:** 스킬은 카테고리 서브폴더 없이 **플랫 구조**로 되어있다. 각 스킬은 자체 디렉토리.

### 스킬 파일 형식
```yaml
---
name: weather
description: Get current weather and forecasts
tools:
  - name: get_weather
    description: Fetch weather for a location
    parameters:
      location:
        type: string
        required: true
---
You are a weather assistant. When asked about weather,
use the get_weather tool to fetch real-time data.
Always include temperature, conditions, and forecast.
```

### 스킬 로딩 흐름
```
skills/*.yml → src/agents/skills.ts → SystemPrompt 빌더 → LLM 호출
```

---

## docs/ - 프로젝트 문서 (Mintlify)

```
docs/
├── index.md           ← 랜딩 페이지
├── getting-started/
│   ├── installation.md
│   ├── quick-start.md
│   └── configuration.md
├── guides/
│   ├── channels.md
│   ├── extensions.md
│   ├── skills.md
│   └── deployment.md
├── api/
│   ├── gateway-rpc.md
│   ├── plugin-sdk.md
│   └── cli-reference.md
├── i18n/
│   └── zh-CN/         ← 중국어 번역 (자동 생성)
└── mint.json          ← Mintlify 설정
```

### docs.acp.md - Agent Client Protocol 문서
```markdown
# ACP Bridge
- `openclaw acp` → stdio로 ACP 프로토콜 노출
- WebSocket을 통해 Gateway에 전달
- ACP session ID → Gateway session key 매핑
- Zed 에디터 통합 문서 포함
```

---

## .agents/ - AI 에이전트 워크플로우

AI 어시스턴트(GitHub Copilot, Claude 등)를 위한 지침 파일.

```
.agents/
├── skills/
│   └── PR_WORKFLOW.md     ← PR 리뷰/머지 워크플로우 (170줄)
└── prompts/
    └── system.md          ← 기본 시스템 프롬프트
```

### PR 워크플로우 (PR_WORKFLOW.md)
```
1. PR 오래된 순서대로 처리
2. 3개 스킬: review-pr → prepare-pr → merge-pr
3. 품질 기준: 엄격한 타입, 보안 평가
4. Rebase 필수
5. 통합 사전 검사: git pull, 메타데이터 쿼리, 문제 확인
```

---

## .pi/ - 프롬프트 엔지니어링

```
.pi/
└── prompts/
    ├── reviewpr.md        ← PR 리뷰 프롬프트 (106줄, 9단계)
    └── system.md          ← 시스템 프롬프트
```

### reviewpr.md 프로세스
```
1. PR diff 분석
2. 변경 범위 파악
3. 코드 품질 평가
4. 보안 취약점 검사
5. 테스트 커버리지 확인
6. 성능 영향 분석
7. 아키텍처 일관성
8. 문서 업데이트 필요성
9. 구조화된 리뷰 출력
```

---

## .agent/ - 에이전트 런타임 설정

```
.agent/
├── config.json        ← 에이전트 런타임 설정
└── state/             ← 에이전트 상태 저장
```

---

## ui/ - 웹 컨트롤 UI

```
ui/
├── package.json       ← Lit + Vite 프로젝트
├── vite.config.ts     ← Vite 빌드 설정
├── src/
│   ├── index.ts       ← UI 진입점
│   ├── components/
│   │   ├── chat-panel.ts     ← 채팅 패널 (Lit 웹 컴포넌트)
│   │   ├── settings-panel.ts ← 설정 패널
│   │   ├── agent-list.ts     ← 에이전트 목록
│   │   └── session-list.ts   ← 세션 목록
│   ├── services/
│   │   └── gateway-ws.ts     ← 게이트웨이 WebSocket 클라이언트
│   └── styles/
│       └── theme.css         ← 테마 스타일
└── public/
```

**기술 스택:**
- Lit (Web Components 라이브러리)
- Vite (번들러)
- A2UI 프로토콜 (vendor/a2ui)로 게이트웨이와 통신

---

## test/ - 통합 테스트

```
test/
├── setup.ts                          ← Vitest 설정 (스텁 채널 플러그인 등록)
├── test-env.ts                       ← 테스트 환경 변수
├── global-setup.ts                   ← 글로벌 설정
├── helpers/                          ← 테스트 헬퍼
├── fixtures/                         ← 테스트 데이터
├── mocks/                            ← 모의 객체
├── gateway.multi.e2e.test.ts         ← 멀티 게이트웨이 E2E
├── auto-reply.retry.test.ts          ← 자동 응답 재시도
├── inbound-contract.providers.test.ts ← 인바운드 계약 테스트
├── media-understanding.auto.e2e.test.ts ← 미디어 이해 E2E
└── provider-timeout.e2e.test.ts      ← 프로바이더 타임아웃
```
**참고:** 대부분의 유닛 테스트는 `src/` 내부에 colocated (`*.test.ts`). `test/`에는 통합/E2E 테스트만 있음.

### test/setup.ts 핵심
```typescript
// 모든 채널을 스텁으로 대체
const stubChannels = ["discord", "slack", "telegram", ...];
for (const ch of stubChannels) {
  registerStubChannel(ch);
}
// → 실제 API 호출 없이 통합 테스트 가능
```

---

## scripts/ - 빌드/릴리스/테스트 스크립트 (60+ 파일)

```
scripts/
├── run-node.mjs           ← 개발 실행 (pnpm openclaw, pnpm dev)
├── watch-node.mjs         ← Watch 모드
├── test-parallel.mjs      ← 병렬 테스트 실행
├── build-docs-list.mjs    ← 문서 빌드
├── bundle-a2ui.sh         ← A2UI 번들링
├── package-mac-app.sh     ← macOS 앱 패키징
├── protocol-gen.ts        ← 프로토콜 생성
├── protocol-gen-swift.ts  ← Swift 프로토콜 생성
├── committer              ← 커밋 헬퍼 스크립트
├── e2e/                   ← E2E 테스트 스크립트
├── docker/                ← Docker 관련
├── docs-i18n/             ← 문서 i18n 파이프라인
├── systemd/               ← systemd 서비스 파일
└── ... (60+ 파일)
```

## AGENTS.md - 개발자 가이드라인 (178줄)

프로젝트에 기여하는 AI 에이전트와 인간 개발자를 위한 규칙 (실제 내용 기반):

```
- Node 22+ (Bun도 지원), TypeScript ESM strict
- 파일 500줄 이하 권장 (~700 가이드라인)
- 테스트: colocated *.test.ts, Vitest + V8 coverage 70%
- 커밋: scripts/committer 사용, 수동 git add/commit 지양
- 보안: 시크릿 커밋 금지, 샌드박스 격리
- 멀티에이전트 안전: git stash 금지, branch 전환 금지, 자기 변경만 커밋
- 채널 작업시: 모든 내장+확장 채널을 함께 고려 (routing, allowlists, pairing, onboarding)
- 플러그인: plugin-only 의존성은 확장 자체 package.json에, 루트에 넣지 말 것
- 릴리스: version 변경 전 반드시 승인 받을 것
```

## CLAUDE.md
```
AGENTS.md
```
(AGENTS.md를 참조하라는 한 줄. 실제로는 AGENTS.md 심링크)
