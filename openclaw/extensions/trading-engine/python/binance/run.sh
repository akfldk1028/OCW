#!/bin/bash
# Trading engine launcher for Mac Mini M4
# Usage: ./run.sh [--testnet] [--futures] [--leverage N]
#
# First time setup:
#   cd openclaw/extensions/trading-engine/python
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r binance/requirements.txt
#   cp binance/.env.example binance/.env
#   # Edit binance/.env with your API keys

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ensure data directories exist
mkdir -p data/models data/logs

# Default args: testnet spot
ARGS="${@:---testnet}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting trading engine: python3 binance/main.py $ARGS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID: $$"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log: data/logs/engine.log"

# Run with output to both console and log file
exec python3 binance/main.py $ARGS 2>&1 | tee -a "data/logs/engine.log"
