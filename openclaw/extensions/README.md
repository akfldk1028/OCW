# extensions/ - 채널/기능 확장 플러그인

각각 독립된 pnpm 워크스페이스 패키지로, `src/plugins/`에 의해 로드됩니다.

## 채널 확장
- `bluebubbles/` - iMessage (BlueBubbles API, 권장)
- `discord/` - Discord 확장 기능
- `googlechat/` - Google Chat (Chat API)
- `imessage/` - iMessage (레거시)
- `irc/` - IRC 채널
- `line/` - LINE 채널
- `matrix/` - Matrix 프로토콜
- `msteams/` - Microsoft Teams
- `slack/` - Slack 확장 기능
- `signal/` - Signal 확장 기능
- `telegram/` - Telegram 확장 기능
- `whatsapp/` - WhatsApp 확장 기능
- `zalo/` - Zalo (비즈니스)
- `zalouser/` - Zalo (개인)
- `feishu/` - 페이슈(라크)
- `mattermost/` - Mattermost
- `nextcloud-talk/` - Nextcloud Talk
- `nostr/` - Nostr 프로토콜
- `tlon/` - Tlon
- `twitch/` - Twitch

## 기능 확장
- `copilot-proxy/` - GitHub Copilot 프록시
- `device-pair/` - 디바이스 페어링
- `diagnostics-otel/` - OpenTelemetry 진단
- `llm-task/` - LLM 태스크 실행
- `lobster/` - Lobster 통합
- `memory-core/` - 메모리 코어
- `memory-lancedb/` - LanceDB 벡터 메모리
- `open-prose/` - 문서 작성 스킬
- `phone-control/` - 전화 제어
- `talk-voice/` - 음성 통화
- `voice-call/` - 음성 통화 (WebRTC)

## 인증 확장
- `google-antigravity-auth/` - Google Antigravity OAuth
- `google-gemini-cli-auth/` - Google Gemini CLI 인증
- `minimax-portal-auth/` - MiniMax 포털 인증
- `qwen-portal-auth/` - Qwen 포털 인증
