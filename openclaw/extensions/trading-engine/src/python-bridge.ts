import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import { WebSocket as WsClient } from "ws";

export type PythonBridgeLogger = {
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

export type StreamEvent = {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
  seq: number;
};

type StreamEventHandler = (event: StreamEvent) => void;

let pythonProcess: ChildProcess | null = null;
let serverPort = 8787;
let serverHost = "127.0.0.1";
let serverScheme: "http" | "https" = "http";
let remoteMode = false; // true = connect to external server, don't spawn subprocess
let bridgeLogger: PythonBridgeLogger = {
  info: () => {},
  warn: () => {},
  error: () => {},
};

// WebSocket client state
let wsClient: WsClient | null = null;
let wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let wsEventHandlers: StreamEventHandler[] = [];
let wsConnected = false;

export function setLogger(logger: PythonBridgeLogger): void {
  bridgeLogger = logger;
}

/**
 * Resolve the Python executable path.
 * If a custom path is provided, use it directly.
 * Otherwise, try common Python executable names.
 */
async function resolvePythonPath(customPath?: string): Promise<string> {
  if (customPath && customPath.trim()) {
    return customPath.trim();
  }

  // Try common Python executable names in order of preference
  const candidates = process.platform === "win32"
    ? ["python", "python3", "py"]
    : ["python3", "python"];

  for (const candidate of candidates) {
    try {
      const result = await new Promise<boolean>((resolve) => {
        const proc = spawn(candidate, ["--version"], {
          stdio: "pipe",
          shell: process.platform === "win32",
        });
        proc.on("close", (code) => resolve(code === 0));
        proc.on("error", () => resolve(false));
        // Kill after 5s if hanging
        setTimeout(() => {
          proc.kill();
          resolve(false);
        }, 5000);
      });
      if (result) {
        return candidate;
      }
    } catch {
      // continue to next candidate
    }
  }

  throw new Error(
    "Could not find Python executable. Install Python 3.10+ or set pythonPath in plugin config.",
  );
}

/**
 * Configure the bridge to connect to a remote (pre-existing) Python server
 * instead of spawning a local subprocess.
 *
 * Use this when the Python server runs on Cloud Run, Railway, or a VM.
 */
export function setRemoteServer(host: string, port: number, scheme: "http" | "https" = "http"): void {
  serverHost = host;
  serverPort = port;
  serverScheme = scheme;
  remoteMode = true;
  bridgeLogger.info(
    `[trading-engine] Remote mode: connecting to ${scheme}://${host}:${port}`,
  );
}

/**
 * Check whether the bridge is in remote mode.
 */
export function isRemoteMode(): boolean {
  return remoteMode;
}

/**
 * Start the Python FastAPI server as a child process.
 *
 * In remote mode, this is a no-op — the server is already running externally.
 * The server script is expected at `extensions/trading-engine/python/server.py`.
 */
export async function startPythonServer(
  pythonPath: string | undefined,
  port: number,
): Promise<void> {
  if (remoteMode) {
    bridgeLogger.info("[trading-engine] Remote mode — skipping subprocess spawn");
    serverPort = port;
    return;
  }

  if (pythonProcess) {
    bridgeLogger.warn("[trading-engine] Python server is already running");
    return;
  }

  serverPort = port;
  const resolvedPython = await resolvePythonPath(pythonPath);

  // Resolve the server.py path relative to this module.
  // This file lives at extensions/trading-engine/src/python-bridge.ts
  // The Python server lives at extensions/trading-engine/python/server.py
  const thisFile = import.meta.url.replace("file:///", "").replace("file://", "");
  const extensionDir = path.dirname(path.dirname(thisFile));
  const serverScript = path.join(extensionDir, "python", "server.py");

  bridgeLogger.info(
    `[trading-engine] Starting Python server: ${resolvedPython} ${serverScript} --port ${port}`,
  );

  pythonProcess = spawn(
    resolvedPython,
    [serverScript],
    {
      stdio: ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
      cwd: path.join(extensionDir, "python"),
      env: {
        ...process.env,
        TRADING_SERVER_PORT: String(port),
      },
    },
  );

  pythonProcess.stdout?.on("data", (data: Buffer) => {
    const text = data.toString().trim();
    if (text) {
      bridgeLogger.info(`[trading-engine:py] ${text}`);
    }
  });

  pythonProcess.stderr?.on("data", (data: Buffer) => {
    const text = data.toString().trim();
    if (text) {
      // Python logs INFO to stderr via uvicorn, so not all stderr is errors
      if (text.includes("ERROR") || text.includes("Traceback")) {
        bridgeLogger.error(`[trading-engine:py] ${text}`);
      } else {
        bridgeLogger.info(`[trading-engine:py] ${text}`);
      }
    }
  });

  pythonProcess.on("close", (code) => {
    bridgeLogger.info(`[trading-engine] Python server exited with code ${code}`);
    pythonProcess = null;
  });

  pythonProcess.on("error", (err) => {
    bridgeLogger.error(`[trading-engine] Failed to start Python server: ${err.message}`);
    pythonProcess = null;
  });

  // Unref so the process doesn't prevent Node from exiting
  pythonProcess.unref();
}

/**
 * Gracefully stop the Python server process.
 */
export async function stopPythonServer(): Promise<void> {
  if (!pythonProcess) {
    return;
  }

  bridgeLogger.info("[trading-engine] Stopping Python server...");

  return new Promise<void>((resolve) => {
    const proc = pythonProcess;
    if (!proc) {
      resolve();
      return;
    }

    const timeout = setTimeout(() => {
      bridgeLogger.warn("[trading-engine] Python server did not exit gracefully, force-killing");
      proc.kill("SIGKILL");
      pythonProcess = null;
      resolve();
    }, 5000);

    proc.on("close", () => {
      clearTimeout(timeout);
      pythonProcess = null;
      resolve();
    });

    // Try graceful shutdown first
    proc.kill("SIGTERM");
  });
}

/**
 * Wait for the Python server to become ready by polling the /health endpoint.
 */
export async function waitForReady(
  port: number,
  timeoutMs: number = 30_000,
): Promise<boolean> {
  const start = Date.now();
  const interval = 500;

  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(`${serverScheme}://${serverHost}:${port}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(2000),
      });
      if (response.ok) {
        bridgeLogger.info(`[trading-engine] Python server is ready on port ${port}`);
        return true;
      }
    } catch {
      // Server not ready yet, retry
    }

    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  bridgeLogger.error(
    `[trading-engine] Python server did not become ready within ${timeoutMs}ms`,
  );
  return false;
}

/**
 * Call an API endpoint on the Python FastAPI server.
 */
export async function callApi<T = unknown>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  urlPath: string,
  body?: unknown,
): Promise<{ ok: boolean; status: number; data: T | null; error?: string }> {
  const url = `${serverScheme}://${serverHost}:${serverPort}${urlPath}`;

  try {
    const options: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(120_000), // 2 minute timeout for long-running operations
    };

    if (body !== undefined && method !== "GET") {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") ?? "";

    let data: T | null = null;
    if (contentType.includes("application/json")) {
      data = (await response.json()) as T;
    } else {
      const text = await response.text();
      data = text as unknown as T;
    }

    if (!response.ok) {
      const errMsg = data && typeof data === "object" && "detail" in data
        ? String((data as Record<string, unknown>).detail)
        : `HTTP ${response.status}`;
      return { ok: false, status: response.status, data, error: errMsg };
    }

    return { ok: true, status: response.status, data };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);

    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return {
        ok: false,
        status: 0,
        data: null,
        error: "Python server is not running. Start it with the trading_health tool or enable autoStart.",
      };
    }

    return { ok: false, status: 0, data: null, error: message };
  }
}

/**
 * Check whether the Python server process is currently alive.
 * In remote mode, returns true (we assume the external server is running).
 */
export function isServerRunning(): boolean {
  if (remoteMode) return true;
  return pythonProcess !== null && !pythonProcess.killed;
}

/**
 * Return the current server port.
 */
export function getServerPort(): number {
  return serverPort;
}

// ---------------------------------------------------------------------------
// WebSocket streaming client
// ---------------------------------------------------------------------------

/**
 * Connect to the Python WebSocket stream at /ws/stream.
 *
 * Automatically reconnects on disconnect with exponential backoff.
 */
export function connectWsStream(port: number): void {
  if (wsClient && wsConnected) {
    bridgeLogger.warn("[trading-engine:ws] Already connected to stream");
    return;
  }

  const wsScheme = serverScheme === "https" ? "wss" : "ws";
  const url = `${wsScheme}://${serverHost}:${port}/ws/stream`;
  bridgeLogger.info(`[trading-engine:ws] Connecting to ${url}`);

  try {
    wsClient = new WsClient(url);
  } catch (err) {
    bridgeLogger.error(
      `[trading-engine:ws] Failed to create WebSocket: ${err instanceof Error ? err.message : String(err)}`,
    );
    scheduleWsReconnect(port);
    return;
  }

  wsClient.on("open", () => {
    wsConnected = true;
    bridgeLogger.info("[trading-engine:ws] Connected to trading stream");

    // Clear any pending reconnect
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = null;
    }
  });

  wsClient.on("message", (raw: Buffer | string) => {
    try {
      const event = JSON.parse(raw.toString()) as StreamEvent;
      for (const handler of wsEventHandlers) {
        try {
          handler(event);
        } catch (handlerErr) {
          bridgeLogger.error(
            `[trading-engine:ws] Event handler error: ${handlerErr instanceof Error ? handlerErr.message : String(handlerErr)}`,
          );
        }
      }
    } catch {
      // Ignore non-JSON messages
    }
  });

  wsClient.on("close", () => {
    wsConnected = false;
    bridgeLogger.info("[trading-engine:ws] Disconnected from trading stream");
    scheduleWsReconnect(port);
  });

  wsClient.on("error", (err: Error) => {
    bridgeLogger.error(`[trading-engine:ws] WebSocket error: ${err.message}`);
    // close event will fire after this
  });
}

/**
 * Disconnect from the WebSocket stream.
 */
export function disconnectWsStream(): void {
  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }

  if (wsClient) {
    wsClient.removeAllListeners();
    if (wsClient.readyState === WsClient.OPEN) {
      wsClient.close();
    }
    wsClient = null;
  }

  wsConnected = false;
  bridgeLogger.info("[trading-engine:ws] WebSocket stream disconnected");
}

/**
 * Register a handler for stream events.
 */
export function onStreamEvent(handler: StreamEventHandler): void {
  wsEventHandlers.push(handler);
}

/**
 * Remove a previously registered stream event handler.
 */
export function offStreamEvent(handler: StreamEventHandler): void {
  wsEventHandlers = wsEventHandlers.filter((h) => h !== handler);
}

/**
 * Check if the WebSocket stream is connected.
 */
export function isWsConnected(): boolean {
  return wsConnected;
}

function scheduleWsReconnect(port: number): void {
  if (wsReconnectTimer) return;

  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    // In remote mode, always try to reconnect (no local process to check)
    // In local mode, only reconnect if subprocess is alive
    const shouldReconnect = remoteMode || (pythonProcess && !pythonProcess.killed);
    if (!wsConnected && shouldReconnect) {
      bridgeLogger.info("[trading-engine:ws] Attempting reconnect...");
      connectWsStream(port);
    }
  }, 5000);
}
