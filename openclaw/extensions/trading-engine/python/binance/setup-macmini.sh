#!/bin/bash
# One-time setup for Mac Mini M4 deployment
# Run from: openclaw/extensions/trading-engine/python/
set -e

echo "=== Trading Engine — Mac Mini Setup ==="

# 1. Python venv
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python venv..."
    python3 -m venv .venv
else
    echo "[1/5] Venv already exists"
fi
source .venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install -r binance/requirements.txt

# 3. Data directories
echo "[3/5] Creating data directories..."
mkdir -p data/models data/logs

# 4. Environment file
if [ ! -f "binance/.env" ]; then
    echo "[4/5] Creating .env from example..."
    cp binance/.env.example binance/.env
    echo "  !! Edit binance/.env with your API keys before running !!"
else
    echo "[4/5] .env already exists"
fi

# 5. Launchd service (optional)
echo "[5/5] Launchd setup..."
PLIST_SRC="binance/com.openclaw.trading-engine.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.openclaw.trading-engine.plist"
if [ ! -f "$PLIST_DST" ]; then
    # Update paths in plist to absolute
    PROJ_DIR="$(pwd)"
    sed "s|~/openclaw/extensions/trading-engine/python|$PROJ_DIR|g" "$PLIST_SRC" > "$PLIST_DST"
    echo "  Installed launchd plist to $PLIST_DST"
    echo "  Load with: launchctl load $PLIST_DST"
else
    echo "  Launchd plist already installed"
fi

echo ""
echo "=== Setup Complete ==="
echo "Quick test:  source .venv/bin/activate && python3 binance/main.py --testnet"
echo "Service:     launchctl load ~/Library/LaunchAgents/com.openclaw.trading-engine.plist"
echo "Logs:        tail -f data/logs/engine.stderr.log"
