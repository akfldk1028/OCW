# Mac Mini Setup Guide

> Trading engine 24/7 운영을 위한 Mac Mini 설정 가이드.
> Fly.io 클라우드와 로컬 운영 모두 지원.

---

## Prerequisites

- macOS 14+ (Apple Silicon recommended)
- Homebrew installed

---

## 1. Install Dependencies

```bash
brew install python@3.11 flyctl git
```

Verify:
```bash
python3 --version   # 3.11.x
fly version          # flyctl v0.x
```

---

## 2. Clone Repository

```bash
cd ~
git clone https://github.com/akfldk1028/OCW.git
cd OCW/openclaw/extensions/trading-engine/python
```

---

## 3. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r binance/requirements.txt
```

---

## 4. Configuration

```bash
cp binance/.env.example binance/.env
```

Edit `binance/.env`:
```env
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
BINANCE_PAPER=true
LIVE_TRADING=false
ANTHROPIC_API_KEY=sk-ant-api03-...
LOG_LEVEL=INFO
```

### API Keys 발급

| Key | Source |
|-----|--------|
| Binance testnet | https://testnet.binance.vision |
| Binance live | https://www.binance.com/en/my/settings/api-management |
| Anthropic | https://console.anthropic.com/settings/keys |
| FRED (optional) | https://fred.stlouisfed.org/docs/api/api_key.html |

---

## 5. Manual Test

```bash
source .venv/bin/activate
python3 binance/main.py --testnet
```

Expected output:
```
[INFO] Loading config: SWING mode, testnet=True
[INFO] WS connected: btcusdt@kline_15m, ethusdt@kline_15m, solusdt@kline_15m
[INFO] Derivatives monitor started
[INFO] Heartbeat loop started
```

Stop with `Ctrl+C`.

### Futures mode (optional):
```bash
python3 binance/main.py --testnet --futures --leverage 3
```

---

## 6. Fly.io Cloud Deployment

```bash
fly auth login
cd ~/OCW/openclaw/extensions/trading-engine/python

# Set secrets (one-time)
fly secrets set \
  BINANCE_API_KEY=your_key \
  BINANCE_SECRET_KEY=your_secret \
  ANTHROPIC_API_KEY=sk-ant-api03-...

# Deploy
fly deploy

# Verify
fly logs --app ocw-trader
fly status --app ocw-trader
```

---

## 7. 24/7 Local Operation (launchd)

### Quick setup (automated):
```bash
cd ~/OCW/openclaw/extensions/trading-engine/python
bash binance/setup-macmini.sh
```

### Manual setup:

1. Copy plist:
```bash
cp binance/com.openclaw.trading-engine.plist ~/Library/LaunchAgents/
```

2. Edit paths in plist to match your install location (default: `~/openclaw/...`):
```bash
# If cloned to ~/OCW instead of ~/openclaw:
sed -i '' 's|~/openclaw|~/OCW/openclaw|g' \
  ~/Library/LaunchAgents/com.openclaw.trading-engine.plist
```

3. Create log directory:
```bash
mkdir -p ~/OCW/openclaw/extensions/trading-engine/python/data/logs
```

4. Load service:
```bash
launchctl load ~/Library/LaunchAgents/com.openclaw.trading-engine.plist
```

5. Verify:
```bash
launchctl list | grep openclaw
# PID should be non-zero
```

### Service Management

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.openclaw.trading-engine.plist

# Restart
launchctl unload ~/Library/LaunchAgents/com.openclaw.trading-engine.plist
launchctl load ~/Library/LaunchAgents/com.openclaw.trading-engine.plist

# View logs
tail -f ~/OCW/openclaw/extensions/trading-engine/python/data/logs/engine.stderr.log
```

### Auto-restart behavior:
- `RunAtLoad: true` — starts on login
- `KeepAlive.SuccessfulExit: false` — restarts on crash (not clean exit)
- `ThrottleInterval: 30` — min 30s between restarts

---

## 8. Monitoring

### Local logs:
```bash
tail -f data/logs/engine.stderr.log
```

### Fly.io logs:
```bash
fly logs --app ocw-trader
```

### Healthcheck (Fly.io):
Docker healthcheck reads `data/heartbeat` file (updated every 30s).
If stale > 120s, container restarts.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ccxt AuthenticationError` | Wrong API keys | Check `binance/.env` |
| `IP banned` (418/403) | REST rate limit | Already mitigated: REST->WS. Wait for ban expiry. |
| `Claude timeout` | API key missing/invalid | Set `ANTHROPIC_API_KEY` |
| OOM on Fly.io | Claude SDK ~250MB RSS | Use `memory = "2048mb"` in fly.toml |
| WS disconnect during Claude call | CPython SSL bug | Auto-reconnects, no action needed |
| launchd not starting | Wrong paths in plist | Check `ProgramArguments` and `WorkingDirectory` |

---

## 10. Updating

```bash
cd ~/OCW
git pull origin master
cd openclaw/extensions/trading-engine/python
source .venv/bin/activate
pip install -r binance/requirements.txt

# If using launchd:
launchctl unload ~/Library/LaunchAgents/com.openclaw.trading-engine.plist
launchctl load ~/Library/LaunchAgents/com.openclaw.trading-engine.plist
```
