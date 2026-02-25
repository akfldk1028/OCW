# apps/ - 네이티브 앱

OpenClaw의 네이티브 클라이언트 앱 소스코드.

- `android/` - Android 앱 (Kotlin, Talk Mode/Canvas/카메라/화면녹화)
- `ios/` - iOS 앱 (Swift, Canvas/Voice Wake/Talk Mode/Bonjour 페어링)
- `macos/` - macOS 메뉴바 앱 (Swift, 제어 플레인/Voice Wake/PTT/WebChat)
- `shared/` - iOS/macOS 공유 코드 (OpenClawKit)

모든 앱은 Gateway에 WebSocket으로 연결되어 동작합니다.
