# skills/ - AI 어시스턴트 스킬

에이전트가 사용할 수 있는 번들/관리형 스킬 모음.
`src/agents/skills.ts`에 의해 로드되어 시스템 프롬프트에 주입됩니다.

각 스킬은 YAML frontmatter + 프롬프트 파일로 구성됩니다.

## 스킬 카테고리
- **생산성**: 1password, apple-notes, apple-reminders, bear-notes, notion, obsidian, things-mac, trello
- **미디어**: camsnap, gifgrep, songsee, video-frames, openai-image-gen, openai-whisper
- **커뮤니케이션**: discord, slack, bluebubbles, imsg, wacli, himalaya (이메일)
- **개발**: github, coding-agent, skill-creator
- **시스템**: healthcheck, model-usage, session-logs, peekaboo (스크린샷)
- **유틸리티**: weather, goplaces, local-places, food-order, canvas, summarize
- **음성**: voice-call, sherpa-onnx-tts, sonoscli, spotify-player
- **기타**: oracle (운세), nano-pdf, nano-banana-pro, tmux
