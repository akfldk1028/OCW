# Mac Mini M4 배포 가이드

## 환경 개요

- 하드웨어: Mac Mini M4, 16GB RAM
- 역할: 24/7 트레이딩 서버
- 방침: 네이티브 Python (Docker 미사용)

## Docker 미사용 이유

Docker Desktop은 ARM Mac에서 VM을 통해 동작하며 1-2GB RAM 오버헤드가 발생한다.
16GB 환경에서 FinBERT + PyTorch를 구동하려면 메모리 절약이 필수.

## 메모리 예산

| 컴포넌트 | 예상 사용량 |
|----------|------------|
| macOS 시스템 | ~3GB |
| FinBERT 모델 | ~500MB |
| PyTorch 런타임 | ~300MB |
| Python 프로세스 (트레이딩 엔진) | ~200MB |
| 여유 | ~12GB |

충분한 여유가 있으나, 모델을 추가로 로드할 경우 모니터링 필요.

## 설치 절차

### 1. Homebrew + Python

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

### 2. 가상환경 생성

```bash
python3.11 -m venv ~/trading-engine-venv
source ~/trading-engine-venv/bin/activate
pip install -r requirements.txt
```

### 3. launchd 자동시작 설정

`~/Library/LaunchAgents/com.trading-engine.plist` 파일 생성:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading-engine</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/USER/trading-engine-venv/bin/python</string>
        <string>/Users/USER/trading-engine/main.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/USER/trading-engine</string>
    <key>StandardOutPath</key>
    <string>/Users/USER/trading-engine/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/USER/trading-engine/logs/stderr.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.trading-engine.plist
launchctl start com.trading-engine
```

## 안정성 필수 사항

### UPS (무정전 전원장치)

정전 시 미체결 주문/포지션 관리 불가. 소형 UPS 필수.
- APC Back-UPS 600VA 수준이면 Mac Mini 30분+ 유지 가능
- 정전 감지 시 열린 포지션 정리 스크립트 연동 권장

### Tailscale VPN

외부에서 Mac Mini 접속용. SSH 포트 노출 없이 안전한 원격 접속.

```bash
brew install tailscale
sudo tailscale up
```

접속: `ssh user@mac-mini-hostname` (Tailscale 네트워크 내)

## 모니터링

### /health 엔드포인트

간단한 HTTP 서버로 헬스체크 제공:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, datetime

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            status = {
                "status": "ok",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "uptime_seconds": get_uptime(),
                "open_positions": get_position_count(),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
```

외부 모니터링 서비스(UptimeRobot 등)에서 주기적으로 호출.

### 로그 로테이션

```bash
# /etc/newsyslog.d/trading-engine.conf
/Users/USER/trading-engine/logs/stdout.log  644  7  1024  *  J
/Users/USER/trading-engine/logs/stderr.log  644  7  1024  *  J
```

7일 보관, 1MB 초과 시 로테이션.
