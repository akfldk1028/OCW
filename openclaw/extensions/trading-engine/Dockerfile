# Trading Engine v3 — Multi-agent Python server
# Standalone deployment: includes FastAPI + WebSocket + cron scheduler
#
# Build:  docker build -t trading-engine .
# Run:    docker run -p 8787:8787 -e ENABLE_CRON=true trading-engine

FROM python:3.11-slim AS base

# System deps for building numpy/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~4GB vs GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps (cache layer)
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
COPY python/ ./
COPY config/ ./config/

# Create required directories
RUN mkdir -p logs models data/raw data/processed config

# Expose server port
EXPOSE 8787

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/health')" || exit 1

# Environment defaults
ENV TRADING_SERVER_PORT=8787
ENV TRADING_SERVER_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run server (bind 0.0.0.0 for container networking)
CMD ["python", "server.py"]
