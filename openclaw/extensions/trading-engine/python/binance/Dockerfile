FROM python:3.11.11-slim

WORKDIR /app

# Install binance/ requirements
COPY binance/requirements.txt /app/binance/requirements.txt
RUN pip install --no-cache-dir -r /app/binance/requirements.txt

# Copy parent modules (shared code) and binance package
COPY *.py /app/
COPY agents/ /app/agents/
COPY core/ /app/core/
COPY analysis/ /app/analysis/
COPY brokers/ /app/brokers/
COPY binance/ /app/binance/

WORKDIR /app/binance

# Create entrypoint that links persistent volume dirs + writes Claude credentials
RUN echo '#!/bin/bash\n\
set -e\n\
# Link persistent state if /state volume is mounted\n\
if [ -d /state ]; then\n\
  mkdir -p /state/data /state/models /state/logs\n\
  rm -rf /app/binance/data /app/binance/models /app/binance/logs\n\
  ln -sf /state/data /app/binance/data\n\
  ln -sf /state/models /app/binance/models\n\
  ln -sf /state/logs /app/binance/logs\n\
fi\n\
# Write Claude credentials (base64-encoded to preserve JSON quotes)\n\
if [ -n "$CLAUDE_CREDENTIALS_B64" ]; then\n\
  mkdir -p /root/.claude\n\
  echo "$CLAUDE_CREDENTIALS_B64" | base64 -d > /root/.claude/.credentials.json\n\
  chmod 600 /root/.claude/.credentials.json\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import time,sys,os;f='data/heartbeat';sys.exit(1)if not os.path.exists(f)else sys.exit(0 if time.time()-float(open(f).read())<120 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-u", "main.py", "--testnet"]
