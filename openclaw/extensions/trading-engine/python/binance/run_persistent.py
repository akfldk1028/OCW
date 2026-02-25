"""Persistent runner — auto-restarts main.py on crash with exponential backoff.

Usage:
    python run_persistent.py [--testnet] [--futures] [--leverage N] [--live]

Works on Windows (MSYS/Git Bash), macOS, and Linux.
On Mac Mini, prefer launchd (com.openclaw.trading-engine.plist) instead.
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Force unbuffered output (critical for MSYS/Git Bash)
os.environ["PYTHONUNBUFFERED"] = "1"

MAIN_PY = Path(__file__).resolve().parent / "main.py"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_BACKOFF = 600  # 10 minutes max between restarts (handles IP bans)
MIN_RUNTIME = 60   # If process ran < 60s, it's a fast crash (increase backoff)
MAX_FAST_CRASHES = 10  # After 10 fast crashes, give up (IP bans need ~10min)


def main():
    args = sys.argv[1:]
    backoff = 5
    fast_crashes = 0
    run_count = 0

    # Forward signals to child
    child_proc = None

    def forward_signal(signum, frame):
        if child_proc and child_proc.poll() is None:
            child_proc.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, forward_signal)

    print(f"[persistent] Starting persistent runner for {MAIN_PY}")
    print(f"[persistent] Args: {args}")
    print(f"[persistent] Max fast crashes before exit: {MAX_FAST_CRASHES}")

    while True:
        run_count += 1
        start_time = time.time()

        log_file = LOG_DIR / f"trading_{time.strftime('%Y%m%d_%H%M%S')}.log"
        print(f"\n[persistent] Run #{run_count} starting... (log: {log_file.name})")

        try:
            with open(log_file, "w") as lf:
                cmd = [sys.executable, str(MAIN_PY)] + args
                child_proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    cwd=str(MAIN_PY.parent),
                )
                exit_code = child_proc.wait()
        except KeyboardInterrupt:
            print("\n[persistent] KeyboardInterrupt — stopping.")
            if child_proc and child_proc.poll() is None:
                child_proc.terminate()
                child_proc.wait(timeout=10)
            break

        runtime = time.time() - start_time
        print(f"[persistent] Process exited with code {exit_code} after {runtime:.0f}s")

        # Check for IP ban in log (don't waste retries — each retry extends ban!)
        ip_ban = False
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            if "IP banned until" in content or "418" in content:
                ip_ban = True
                import re
                m = re.search(r"IP banned until (\d+)", content)
                if m:
                    ban_until = int(m.group(1)) / 1000
                    wait_s = ban_until - time.time() + 60  # +60s margin (Binance escalates)
                    if wait_s > 0:
                        backoff = min(wait_s, 3600)  # allow up to 1h wait
                        print(f"[persistent] IP BAN detected — waiting {backoff:.0f}s ({backoff/60:.0f}min) until ban expires")
                    else:
                        backoff = 60  # ban expired but be cautious
                        print(f"[persistent] IP BAN expired {-wait_s:.0f}s ago — cautious restart in {backoff}s")
                else:
                    # 418 but no timestamp — wait conservatively
                    backoff = 600
                    print(f"[persistent] IP BAN (no timestamp) — waiting {backoff}s")
                # Reset fast crash counter — ban is not a code bug
                fast_crashes = 0
        except Exception:
            pass

        if runtime < MIN_RUNTIME:
            if not ip_ban:
                fast_crashes += 1
                backoff = min(backoff * 2, MAX_BACKOFF)
                print(f"[persistent] Fast crash #{fast_crashes} (runtime < {MIN_RUNTIME}s)")

                if fast_crashes >= MAX_FAST_CRASHES:
                    print(f"[persistent] FATAL: {MAX_FAST_CRASHES} fast crashes — giving up.")
                    print(f"[persistent] Check log: {log_file}")
                    sys.exit(1)
        else:
            # Healthy run — reset backoff
            fast_crashes = 0
            backoff = 5

        print(f"[persistent] Restarting in {backoff}s...")
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            print("\n[persistent] KeyboardInterrupt during backoff — stopping.")
            break

    print("[persistent] Bye.")


if __name__ == "__main__":
    main()
