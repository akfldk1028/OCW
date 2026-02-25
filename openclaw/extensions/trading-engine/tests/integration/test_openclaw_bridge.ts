/**
 * Integration tests for the Node.js ↔ Python bridge.
 *
 * These tests verify:
 * 1. Python server can be started and stopped
 * 2. Health check endpoint works
 * 3. Predict endpoint works (with/without trained model)
 * 4. Status endpoint returns expected structure
 *
 * Prerequisites:
 * - Python 3.10+ with venv at ../python/venv/
 * - Dependencies installed: pip install -r requirements.txt
 *
 * Run with: npx vitest run tests/integration/test_openclaw_bridge.ts
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";

const PYTHON_DIR = resolve(__dirname, "../../python");
const SERVER_PORT = 18787; // Use non-default port for testing
const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}`;
const STARTUP_TIMEOUT_MS = 30_000;

let serverProcess: ChildProcess | null = null;

async function waitForServer(url: string, timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${url}/health`);
      if (res.ok) return true;
    } catch {
      // Server not ready yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

describe("OpenClaw ↔ Python Bridge Integration", () => {
  beforeAll(async () => {
    // Determine Python executable (venv or system)
    const isWin = process.platform === "win32";
    const pythonPath = isWin
      ? resolve(PYTHON_DIR, "venv/Scripts/python.exe")
      : resolve(PYTHON_DIR, "venv/bin/python");

    serverProcess = spawn(pythonPath, ["server.py"], {
      cwd: PYTHON_DIR,
      env: {
        ...process.env,
        TRADING_SERVER_PORT: String(SERVER_PORT),
      },
      stdio: "pipe",
    });

    serverProcess.stderr?.on("data", (data: Buffer) => {
      // Uncomment for debugging:
      // console.error(`[python stderr] ${data.toString()}`);
    });

    const ready = await waitForServer(SERVER_URL, STARTUP_TIMEOUT_MS);
    if (!ready) {
      throw new Error(`Python server failed to start within ${STARTUP_TIMEOUT_MS}ms`);
    }
  }, STARTUP_TIMEOUT_MS + 5000);

  afterAll(() => {
    if (serverProcess) {
      serverProcess.kill("SIGTERM");
      serverProcess = null;
    }
  });

  describe("Health Check", () => {
    it("should return ok status", async () => {
      const res = await fetch(`${SERVER_URL}/health`);
      expect(res.status).toBe(200);

      const data = await res.json();
      expect(data.status).toBe("ok");
      expect(data).toHaveProperty("models_loaded");
      expect(data).toHaveProperty("server_uptime");
    });
  });

  describe("Status", () => {
    it("should return model and risk config info", async () => {
      const res = await fetch(`${SERVER_URL}/status`);
      expect(res.status).toBe(200);

      const data = await res.json();
      expect(data).toHaveProperty("models_loaded");
      expect(data).toHaveProperty("ensemble_weights");
      expect(data).toHaveProperty("risk_config");
    });
  });

  describe("Predict", () => {
    it("should accept prediction request", async () => {
      const res = await fetch(`${SERVER_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: ["AAPL"] }),
      });

      // Without trained model, may return 200 with zeros or 503
      expect([200, 503]).toContain(res.status);
    });
  });

  describe("Train", () => {
    it("should accept training request with minimal params", async () => {
      const res = await fetch(`${SERVER_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: ["AAPL"],
          lookback_days: 30,
          total_timesteps: 100,
        }),
      });

      expect([200, 202]).toContain(res.status);
      const data = await res.json();
      expect(data).toHaveProperty("status");
    });
  });

  describe("Backtest", () => {
    it("should accept backtest request", async () => {
      const res = await fetch(`${SERVER_URL}/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: ["AAPL"],
          start_date: "2025-01-01",
          end_date: "2025-06-01",
        }),
      });

      // Without trained model, may return error
      expect([200, 400, 503]).toContain(res.status);
    });
  });
});
