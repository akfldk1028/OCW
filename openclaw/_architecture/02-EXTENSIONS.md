# extensions/ - 플러그인 시스템

> 36개 확장 플러그인. 채널/인증/도구 확장을 Plugin SDK로 등록.
> Plugin SDK (`openclaw/plugin-sdk`)를 통해 일관된 인터페이스로 등록.
> **주의:** 코어 채널(Discord, Slack 등)의 핵심 로직은 `src/`에 있고, extensions/는 Plugin SDK wrapper.

## 플러그인 등록 아키텍처

```
Extension (index.ts)
  │
  ├── export default { id, name, configSchema, register(api) }
  │
  └── register(api: OpenClawPluginApi) 에서:
      ├── api.registerChannel()        ← 메시징 채널 등록
      ├── api.registerTool()           ← AI 도구 등록
      ├── api.registerProvider()       ← 인증 프로바이더 등록
      ├── api.registerGatewayMethod()  ← RPC 메서드 등록
      ├── api.registerCommand()        ← 채팅 명령어 등록
      ├── api.registerCli()            ← CLI 명령어 등록
      ├── api.registerHttpHandler()    ← Webhook 핸들러 등록
      ├── api.registerService()        ← 백그라운드 서비스 등록
      └── api.addEventListener()       ← 이벤트 훅 등록
```

## 전체 확장 목록 (36개)

### A. 메시징 채널 (19개)

채팅 메시지를 주고받는 플랫폼 연동. 모두 `api.registerChannel()` 사용.
코어 채널(Discord, Slack 등)은 `src/`에 핵심 로직이 있고, 여기는 게이트웨이 등록 wrapper.

| Extension | 플랫폼 | 복잡도 | 특이사항 |
|-----------|--------|--------|---------|
| `slack` | Slack | 기본 | @slack/bolt 기반 |
| `discord` | Discord | 기본 | 스레드, 리액션, 임베드 |
| `telegram` | Telegram | 기본 | grammy, 인라인 키보드 |
| `signal` | Signal | 기본 | E2E 암호화 |
| `whatsapp` | WhatsApp | 기본 | Business API |
| `imessage` | iMessage | 기본 | macOS 전용 |
| `bluebubbles` | BlueBubbles | 중간 | + HttpHandler (webhook) |
| `irc` | IRC | 기본 | 클래식 IRC 프로토콜 |
| `line` | Line | 중간 | + Dock UI |
| `matrix` | Matrix | 기본 | 분산 프로토콜 |
| `mattermost` | Mattermost | 기본 | 오픈소스 Slack 대안 |
| `msteams` | MS Teams | 기본 | Microsoft 365 |
| `googlechat` | Google Chat | 중간 | + Dock UI |
| `feishu` | Feishu/Lark | 복잡 | + 문서/위키/드라이브 도구 |
| `nextcloud-talk` | Nextcloud | 기본 | 셀프호스팅 |
| `nostr` | Nostr | 중간 | + HttpHandler, 분산 |
| `tlon` | Tlon (Urbit) | 기본 | P2P |
| `twitch` | Twitch | 기본 | 스트리밍 채팅 |
| `zalo` | Zalo (OA) | 중간 | + HttpHandler + Dock UI |
| `zalouser` | Zalo (User) | 중간 | + Tool (action 기반) |

**채널 등록 패턴 (공통):**
```typescript
export default {
  id: "channel-name",
  name: "Channel Name",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    setChannelRuntime(api.runtime);
    api.registerChannel({ plugin: channelPlugin });
    // 선택: api.registerHttpHandler(webhookHandler);
    // 선택: dock: channelDock
  },
};
```

### B. 인증 프로바이더 (5개)

LLM API 접근을 위한 인증 플러그인. `api.registerProvider()` 사용.

| Extension | 대상 | 인증 방식 |
|-----------|------|----------|
| `google-antigravity-auth` | Google AI | OAuth2 PKCE (441줄) |
| `google-gemini-cli-auth` | Gemini CLI | OAuth2 |
| `copilot-proxy` | GitHub Copilot | 로컬 프록시 |
| `minimax-portal-auth` | Minimax | 포털 인증 |
| `qwen-portal-auth` | Qwen | Device Code Flow |

### C. 복합 도구 확장 (5개)

AI 에이전트가 사용하는 도구를 제공. `api.registerTool()` 사용.

#### trading-engine (539줄, 가장 핵심)
```
extensions/trading-engine/
├── index.ts              ← 플러그인 진입점 (5 tools + 5 gateway methods + CLI + chat cmd)
├── src/
│   └── python-bridge.ts  ← Python 프로세스 관리 (spawn, HTTP 통신)
└── python/
    ├── server.py         ← FastAPI 서버 (463줄, 5 엔드포인트)
    ├── ensemble_agent.py ← PPO/A2C/DDPG 앙상블 (625줄)
    ├── risk_manager.py   ← 리스크 관리 (401줄)
    ├── data_processor.py ← 데이터 파이프라인 (298줄)
    ├── sentiment_scorer.py ← 감성분석 (239줄)
    ├── config.py         ← 설정값
    └── requirements.txt  ← Python 의존성
```

**등록하는 도구 5개:**
| 도구 | 기능 |
|------|------|
| `trading_predict` | PPO+A2C+DDPG 앙상블 예측, 리스크 조정 |
| `trading_train` | 모델 학습 (최대 5M timesteps) |
| `trading_backtest` | 히스토리컬 백테스트 + 벤치마크 비교 |
| `trading_status` | 엔진 상태, 앙상블 가중치, 메트릭 |
| `trading_health` | 서버 헬스체크 + 자동 복구 |

**게이트웨이 메서드 5개:** `trading.predict`, `trading.train`, `trading.backtest`, `trading.status`, `trading.health`

#### voice-call (512줄, 가장 복잡한 설정)
```
- Provider: Twilio / Telnyx / Mock
- Tunnel: ngrok / Tailscale
- Streaming: OpenAI Realtime API
- TTS override: OpenAI / ElevenLabs
- 6개 게이트웨이 메서드 (voicecall.*)
- 1개 통합 도구 (voice_call, 6개 액션 Union)
```

#### memory-lancedb (626줄)
```
- 벡터 임베딩 (OpenAI 연동)
- 시맨틱 검색 (유사도 기반)
- 에이전트 입출력 자동 캡처/리콜
- 3개 도구: memory_recall, memory_store, memory_forget
- 이벤트 훅: before_agent_start, agent_end
```

#### memory-core
```
- 그래프 기반 메모리 시스템
- 런타임 주입 설정
```

#### llm-task
```
- LLM 작업 오케스트레이션
```

### D. 음성 확장 (1개)

| Extension | 역할 |
|-----------|------|
| `talk-voice` | 음성 대화 (/voice 명령어, 음성 목록/설정) |

### E. 서비스 확장 (2개)

| Extension | 역할 |
|-----------|------|
| `diagnostics-otel` | OpenTelemetry 관측성 (트레이싱, 메트릭) |
| `open-prose` | 텍스트 스킬 (Open Prose 프레임워크) |

### F. 유틸리티 확장 (3개)

| Extension | 역할 |
|-----------|------|
| `device-pair` | 장치 페어링 |
| `phone-control` | 전화 제어 |
| `lobster` | Lobster 프레임워크 통합 |

## 확장 설정 패턴

### 빈 설정 (대부분 채널)
```typescript
configSchema: emptyPluginConfigSchema()
// 런타임에 환경변수로 설정
```

### 커스텀 설정 (복잡 확장)
```typescript
// trading-engine
configSchema: {
  pythonPath?: string;     // Python 경로 (자동 탐지)
  serverPort: number;      // FastAPI 포트 (기본 8787)
  autoStart: boolean;      // 게이트웨이 시작시 자동 실행
}

// voice-call (40+ 필드)
configSchema: {
  provider: "twilio" | "telnyx" | "mock";
  tunnel: "ngrok" | "tailscale";
  streaming: { /* OpenAI Realtime config */ };
  tts: { /* TTS override */ };
  webhook: { port, bind, path };
}
```

## Plugin SDK API 요약

```typescript
interface OpenClawPluginApi {
  // 메타데이터
  logger: Logger;
  config: CoreConfig;
  pluginConfig: PluginConfig;
  runtime: {
    config: RuntimeConfig;
    tts: TTSRuntime;
    logger: Logger;
  };

  // 등록 메서드
  registerChannel(opts: { plugin: ChannelPlugin; dock?: DockConfig }): void;
  registerTool(tool: ToolDefinition): void;
  registerProvider(provider: ProviderAuth): void;
  registerGatewayMethod(name: string, handler: MethodHandler): void;
  registerCommand(cmd: CommandDefinition): void;
  registerCli(builder: CliBuilder, opts?: { commands: string[] }): void;
  registerHttpHandler(handler: HttpHandler): void;
  registerService(service: ServiceDefinition): void;
  addEventListener(event: string, handler: EventHandler): void;
}
```

## 복잡도별 분류

```
Minimal (16개)  : 순수 채널 플러그인 wrapper, ~15-30줄
Simple (6개)    : 채널 + HttpHandler 또는 기본 인증
Intermediate (6개): 도구 등록 또는 복잡한 설정
Complex (3개)   : 다기능, 400줄 이상 (trading-engine, voice-call, memory-lancedb)
Auth (5개)      : OAuth/Device Code/Proxy 인증
```

## 핵심 이해 포인트

**src/ vs extensions/ 채널 이중 구조:**
- `src/discord/` = accounts, monitor, 메시지 파싱, 상태 관리 등 핵심 구현
- `extensions/discord/index.ts` = `api.registerChannel({ plugin: discordPlugin })` 한 줄짜리 wrapper
- 이 분리 덕분에 코어 채널도 확장 채널과 동일한 Plugin SDK 인터페이스로 게이트웨이에 등록됨
