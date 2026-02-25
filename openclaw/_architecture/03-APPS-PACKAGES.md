# apps/, packages/, vendor/, Swabble/ - 클라이언트 & 공유 코드

## apps/ - 네이티브 앱 클라이언트

게이트웨이에 WebSocket/HTTP로 연결하는 네이티브 앱들.

```
apps/
├── android/          ← Android 앱 (Kotlin + Jetpack Compose)
├── ios/              ← iOS 앱 (Swift + SwiftUI)
├── macos/            ← macOS 메뉴바 앱 (Swift + AppKit)
└── OpenClawKit/      ← iOS/macOS 공유 Swift 패키지
```

### Android (apps/android/)

```
android/
├── build.gradle.kts         ← Gradle 빌드 설정
├── app/
│   └── src/main/
│       ├── java/.../
│       │   ├── MainActivity.kt       ← 메인 액티비티
│       │   ├── ChatScreen.kt         ← 채팅 UI (Compose)
│       │   ├── SettingsScreen.kt     ← 설정 화면
│       │   ├── GatewayClient.kt      ← 게이트웨이 WebSocket 클라이언트
│       │   └── MessageViewModel.kt   ← 메시지 상태 관리
│       └── AndroidManifest.xml
└── gradle/
```

**특징:**
- Kotlin + Jetpack Compose (선언형 UI)
- Material 3 디자인 시스템
- 게이트웨이와 WebSocket 실시간 통신
- 푸시 알림 (FCM)

### iOS (apps/ios/)

```
ios/
├── OpenClaw.xcodeproj
├── OpenClaw/
│   ├── App.swift               ← SwiftUI 앱 진입점
│   ├── ContentView.swift       ← 메인 뷰
│   ├── ChatView.swift          ← 채팅 인터페이스
│   ├── SettingsView.swift      ← 설정
│   └── GatewayConnection.swift ← 게이트웨이 연결
└── OpenClawTests/
```

**특징:**
- Swift + SwiftUI
- OpenClawKit 공유 라이브러리 의존
- 상태: 50% 구현 (기본 채팅 동작)

### macOS (apps/macos/)

```
macos/
├── OpenClaw.xcodeproj
├── OpenClaw/
│   ├── AppDelegate.swift       ← 메뉴바 앱
│   ├── StatusBarController.swift ← 시스템 트레이 아이콘
│   ├── PopoverView.swift       ← 팝오버 채팅 UI
│   └── GatewayManager.swift    ← 게이트웨이 관리
└── OpenClawTests/
```

**특징:**
- macOS 메뉴바 앱 (상시 상주)
- 팝오버로 빠른 채팅
- 시스템 트레이 통합

### OpenClawKit (apps/shared/OpenClawKit/)

```
apps/shared/OpenClawKit/
```

**역할:** iOS와 macOS 앱이 공유하는 Swift 패키지 (게이트웨이 통신, 모델, 유틸리티)
- 정확한 경로: `apps/shared/OpenClawKit/` (apps/ 직접 하위가 아님)
- Swift Package Manager로 관리

---

## packages/ - 공유 NPM 패키지

pnpm 워크스페이스로 관리되는 내부 패키지들.

```
packages/
├── clawdbot/         ← 호환성 심 (레거시 API 지원)
└── moltbot/          ← 호환성 심 (레거시 API 지원)
```

### clawdbot (packages/clawdbot/)
```json
{
  "name": "@openclaw/clawdbot",
  "description": "Compatibility shim for legacy clawdbot API"
}
```
- 이전 `clawdbot` API를 사용하던 코드와의 호환성 유지
- 새로운 OpenClaw API로 리다이렉트

### moltbot (packages/moltbot/)
```json
{
  "name": "@openclaw/moltbot",
  "description": "Compatibility shim for legacy moltbot API"
}
```
- `moltbot` 레거시 API 호환성 심

---

## vendor/ - 벤더 코드

외부 라이브러리를 프로젝트 내에 직접 포함.

```
vendor/
└── a2ui/
    ├── protocol/        ← A2UI 프로토콜 정의
    │   ├── messages.ts  ← 메시지 타입 (Canvas 통신)
    │   └── schema.ts    ← 프로토콜 스키마
    └── renderers/
        └── canvas/      ← Canvas 렌더러 구현
```

**A2UI (Agent-to-UI) 프로토콜:**
- 에이전트가 UI 요소를 동적으로 생성/업데이트하는 프로토콜
- Canvas 기반 렌더링
- 게이트웨이 ↔ 웹 UI 간 실시간 UI 업데이트에 사용
- 벤더로 관리하는 이유: 커스텀 포크 또는 아직 npm 배포 전

---

## Swabble/ - 웨이크워드 데몬

```
Swabble/
├── Package.swift           ← Swift 6.2 패키지 설정
├── Sources/
│   └── Swabble/
│       ├── SwabbleApp.swift      ← 데몬 메인
│       ├── WakeWordDetector.swift ← 음성 감지 (Speech.framework)
│       ├── AudioCapture.swift     ← 마이크 입력 캡처
│       └── GatewayNotifier.swift  ← 감지시 게이트웨이 알림
└── Tests/
```

**역할:**
- macOS에서 실행되는 **웨이크워드 감지 데몬**
- Apple Speech.framework 사용
- "Hey OpenClaw" 같은 트리거 워드 감지
- 감지되면 게이트웨이에 `voicewake` 이벤트 전송
- Swift 6.2 (최신) 기반

---

## patches/ - pnpm 패치

```
patches/
├── @anthropic-ai+sdk+0.32.1.patch
├── grammy+1.35.0.patch
└── ...
```

**역할:**
- 외부 의존성의 버그 수정이나 커스텀 변경
- `pnpm patch` 명령으로 생성
- `package.json`의 `pnpm.patchedDependencies`에 등록
- npm 패키지 새 버전 나올 때까지 임시 수정용

---

## 폴더간 관계도

```
OpenClawKit ──────────┐
(Swift 공유 라이브러리)  │
                      │ 의존
  iOS App ◄───────────┤
  macOS App ◄─────────┘
  Android App (독립)
       │
       │ WebSocket/HTTP
       ▼
  Gateway (src/gateway/)
       │
       │ Plugin SDK
       ▼
  Extensions (extensions/)
       │
       │ 호환 심
       ├── clawdbot (packages/)
       └── moltbot (packages/)

  Swabble ──── voicewake 이벤트 ──→ Gateway
  vendor/a2ui ─── 프로토콜 ──→ ui/ (Web UI)
```
