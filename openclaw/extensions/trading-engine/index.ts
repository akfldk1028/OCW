import type { OpenClawPluginApi, OpenClawPluginService } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import {
  startPythonServer,
  stopPythonServer,
  waitForReady,
  callApi,
  isServerRunning,
  getServerPort,
  setLogger,
  setRemoteServer,
  isRemoteMode,
  connectWsStream,
  disconnectWsStream,
  onStreamEvent,
  isWsConnected,
  type StreamEvent,
} from "./src/python-bridge.js";

// ---------------------------------------------------------------------------
// Plugin config
// ---------------------------------------------------------------------------

type TradingEngineConfig = {
  pythonPath?: string;
  serverPort: number;
  autoStart: boolean;
  /** Remote server URL (e.g. "https://trading-engine-xxx.run.app" or "http://my-vm:8787") */
  serverUrl?: string;
};

function parseConfig(raw: unknown): TradingEngineConfig {
  const obj =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, unknown>)
      : {};

  return {
    pythonPath: typeof obj.pythonPath === "string" ? obj.pythonPath : undefined,
    serverPort:
      typeof obj.serverPort === "number" && obj.serverPort > 0
        ? obj.serverPort
        : 8787,
    autoStart: typeof obj.autoStart === "boolean" ? obj.autoStart : true,
    serverUrl: typeof obj.serverUrl === "string" && obj.serverUrl.trim()
      ? obj.serverUrl.trim()
      : undefined,
  };
}

const tradingEngineConfigSchema = {
  parse(value: unknown): TradingEngineConfig {
    return parseConfig(value);
  },
  uiHints: {
    pythonPath: {
      label: "Python Path",
      help: "Leave empty for auto-detection",
      placeholder: "/usr/bin/python3",
    },
    serverPort: {
      label: "Server Port",
      help: "Port for the FinRL backend server",
    },
    autoStart: {
      label: "Auto Start",
      help: "Start Python server automatically when gateway starts",
    },
    serverUrl: {
      label: "Remote Server URL",
      help: "Connect to a remote Python server (Cloud Run, Railway, VM). Leave empty for local subprocess.",
      placeholder: "https://trading-engine-xxx.run.app",
    },
  },
};

// ---------------------------------------------------------------------------
// Tool parameter schemas (TypeBox)
// ---------------------------------------------------------------------------

const PredictParams = Type.Object({
  tickers: Type.Array(Type.String(), {
    description: "List of stock tickers, e.g. [\"AAPL\", \"MSFT\", \"GOOGL\"]",
  }),
  start_date: Type.Optional(
    Type.String({ description: "Start date for prediction context (YYYY-MM-DD)" }),
  ),
  end_date: Type.Optional(
    Type.String({ description: "End date for prediction context (YYYY-MM-DD)" }),
  ),
  initial_amount: Type.Optional(
    Type.Number({ description: "Initial portfolio amount in USD (default: 1000000)" }),
  ),
  risk_level: Type.Optional(
    Type.Union(
      [Type.Literal("conservative"), Type.Literal("moderate"), Type.Literal("aggressive")],
      { description: "Risk management level" },
    ),
  ),
});

const TrainParams = Type.Object({
  tickers: Type.Array(Type.String(), {
    description: "List of stock tickers to train on",
  }),
  start_date: Type.String({ description: "Training start date (YYYY-MM-DD)" }),
  end_date: Type.String({ description: "Training end date (YYYY-MM-DD)" }),
  models: Type.Optional(
    Type.Array(
      Type.Union([Type.Literal("ppo"), Type.Literal("a2c"), Type.Literal("sac")]),
      { description: "Models to train (default: all three)" },
    ),
  ),
  timesteps: Type.Optional(
    Type.Number({ description: "Total training timesteps per model (default: 50000)" }),
  ),
});

const BacktestParams = Type.Object({
  tickers: Type.Array(Type.String(), {
    description: "List of stock tickers to backtest",
  }),
  start_date: Type.String({ description: "Backtest start date (YYYY-MM-DD)" }),
  end_date: Type.String({ description: "Backtest end date (YYYY-MM-DD)" }),
  initial_amount: Type.Optional(
    Type.Number({ description: "Initial portfolio amount in USD (default: 1000000)" }),
  ),
  benchmark: Type.Optional(
    Type.String({ description: "Benchmark ticker for comparison (default: SPY)" }),
  ),
});

const StatusParams = Type.Object({
  verbose: Type.Optional(
    Type.Boolean({ description: "Include detailed model and position information" }),
  ),
});

const ScanParams = Type.Object({
  top_sectors: Type.Optional(
    Type.Number({ description: "Number of top sectors to select (default: 3)", minimum: 1, maximum: 14 }),
  ),
  stocks_per_sector: Type.Optional(
    Type.Number({ description: "Number of stocks to pick per sector (default: 5)", minimum: 1, maximum: 10 }),
  ),
  include_sentiment: Type.Optional(
    Type.Boolean({ description: "Include news sentiment in stock scoring (default: true)" }),
  ),
});

const AutoDecideParams = Type.Object({
  portfolio_value: Type.Optional(
    Type.Number({ description: "Portfolio value in USD (default: 100000)", minimum: 1 }),
  ),
  top_sectors: Type.Optional(
    Type.Number({ description: "Number of top sectors (default: 3)", minimum: 1, maximum: 14 }),
  ),
  stocks_per_sector: Type.Optional(
    Type.Number({ description: "Stocks per sector (default: 5)", minimum: 1, maximum: 10 }),
  ),
  include_sentiment: Type.Optional(
    Type.Boolean({ description: "Include FinBERT sentiment (default: true)" }),
  ),
});

const ScanBacktestParams = Type.Object({
  months: Type.Optional(
    Type.Number({ description: "Number of months to backtest (default: 12)", minimum: 3, maximum: 36 }),
  ),
});

const ExecuteParams = Type.Object({
  dry_run: Type.Optional(
    Type.Boolean({ description: "Dry run mode - no real orders (default: true for safety)" }),
  ),
  portfolio_value: Type.Optional(
    Type.Number({ description: "Portfolio value for auto-decide (default: 100000)" }),
  ),
});

const HealthParams = Type.Object({
  start_if_down: Type.Optional(
    Type.Boolean({ description: "Attempt to start the server if it is not running (default: true)" }),
  ),
});

const StreamStatusParams = Type.Object({});

const QuantRankParams = Type.Object({});

const QuantRetrainParams = Type.Object({});

const RiskCheckParams = Type.Object({});

const PipelineBacktestParams = Type.Object({
  start_date: Type.Optional(
    Type.String({ description: "Backtest start date (YYYY-MM-DD, default: 2024-06-01)" }),
  ),
  end_date: Type.Optional(
    Type.String({ description: "Backtest end date (YYYY-MM-DD, default: 2026-02-01)" }),
  ),
  rebalance_days: Type.Optional(
    Type.Number({ description: "Rebalance period in trading days (default: 21)", minimum: 5, maximum: 63 }),
  ),
  initial_cash: Type.Optional(
    Type.Number({ description: "Initial cash in USD (default: 100000)", minimum: 1 }),
  ),
  use_xgboost: Type.Optional(
    Type.Boolean({ description: "Use XGBoost ranking vs momentum proxy (default: true)" }),
  ),
});

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function jsonResult(payload: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload, null, 2) }],
    details: payload,
  };
}

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

const tradingEnginePlugin = {
  id: "trading-engine",
  name: "AI Trading Engine",
  description: "Real-time multi-agent trading engine with WebSocket streaming, XGBoost cross-sectional ranking, and ensemble risk management",
  configSchema: tradingEngineConfigSchema,

  register(api: OpenClawPluginApi) {
    const config = parseConfig(api.pluginConfig);
    setLogger(api.logger);

    // ------------------------------------------------------------------
    // Service: manage the Python subprocess lifecycle
    // ------------------------------------------------------------------

    const service: OpenClawPluginService = {
      id: "trading-engine",

      start: async (_ctx) => {
        try {
          // Remote mode: connect to external server (Cloud Run, Railway, VM)
          if (config.serverUrl) {
            const url = new URL(config.serverUrl);
            const host = url.hostname;
            const port = url.port ? parseInt(url.port, 10) : (url.protocol === "https:" ? 443 : 80);
            const scheme = url.protocol === "https:" ? "https" as const : "http" as const;
            setRemoteServer(host, port, scheme);
            api.logger.info(`[trading-engine] Remote mode: ${config.serverUrl}`);
          } else if (!config.autoStart) {
            api.logger.info("[trading-engine] autoStart is disabled; skipping server launch");
            return;
          } else {
            // Local mode: spawn Python subprocess
            await startPythonServer(config.pythonPath, config.serverPort);
          }

          const readyPort = config.serverUrl
            ? getServerPort()
            : config.serverPort;
          const ready = await waitForReady(readyPort, 30_000);
          if (!ready) {
            api.logger.warn(
              "[trading-engine] Python server did not become ready in time",
            );
            return;
          }

          // Connect WebSocket stream for real-time events
          connectWsStream(readyPort);
          onStreamEvent((event: StreamEvent) => {
            // Broadcast to OpenClaw gateway for UI / Telegram / Discord
            api.broadcast?.("trading." + event.type, event.data);
          });
          api.logger.info("[trading-engine] WebSocket stream bridge connected");
        } catch (err) {
          api.logger.error(
            `[trading-engine] Failed to start Python server: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
        }
      },

      stop: async (_ctx) => {
        disconnectWsStream();
        if (!isRemoteMode()) {
          await stopPythonServer();
        }
      },
    };

    api.registerService(service);

    // ------------------------------------------------------------------
    // Tool: trading_predict
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_predict",
      label: "Trading Predict",
      description:
        "Generate ensemble trading predictions for given tickers using the FinRL engine " +
        "(PPO + A2C + DDPG). Returns recommended actions (buy/sell/hold) with confidence " +
        "scores and risk-adjusted position sizes.",
      parameters: PredictParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/predict", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_train
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_train",
      label: "Trading Train",
      description:
        "Train or retrain the ensemble FinRL models (PPO, A2C, DDPG) on historical " +
        "market data for the specified tickers and date range. This is a long-running " +
        "operation -- the tool returns once training is complete.",
      parameters: TrainParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          // Training can take a long time, use a longer internal timeout
          const result = await callApi("POST", "/train", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_backtest
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_backtest",
      label: "Trading Backtest",
      description:
        "Run a historical backtest of the ensemble trading strategy over a date range " +
        "and compare performance against a benchmark. Returns portfolio value curve, " +
        "Sharpe ratio, max drawdown, and other risk metrics.",
      parameters: BacktestParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/backtest", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_status
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_status",
      label: "Trading Status",
      description:
        "Get the current status of the trading engine including server health, " +
        "loaded models, active positions, and recent performance metrics.",
      parameters: StatusParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const verbose = params?.verbose === true;
          const urlPath = verbose ? "/status?verbose=true" : "/status";
          const result = await callApi("GET", urlPath);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_scan
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_scan",
      label: "Trading Scan",
      description:
        "Scan 14 sector ETFs for relative momentum vs SPY, rank sectors by composite " +
        "score, and pick top stocks within the strongest sectors using momentum, volume, " +
        "RSI, and news sentiment. Does NOT require a trained model.",
      parameters: ScanParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/scan", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_decide
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_decide",
      label: "Trading Multi-Agent Decide",
      description:
        "Multi-agent trading decision (v3 pipeline). Runs MarketAgent " +
        "(regime + sector), QuantAgent (XGBoost cross-sectional ranking), " +
        "and Synthesizer (weighted voting with EXIT management, FinBERT " +
        "sentiment, and position awareness). Auto-fetches current Alpaca " +
        "positions if not provided. This is the BRAIN: call this to get " +
        "'what should I trade right now?'",
      parameters: AutoDecideParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/decide/v3", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_regime
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_regime",
      label: "Trading Regime",
      description:
        "Detect the current market regime using a Hidden Markov Model (HMM). " +
        "Returns regime state (low_vol/high_vol), confidence level, realised " +
        "volatility, transition probabilities, and strategy adjustments.",
      parameters: Type.Object({}),

      async execute(_toolCallId: string, _params: Record<string, unknown>) {
        try {
          const result = await callApi("GET", "/regime");
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_scan_backtest
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_scan_backtest",
      label: "Trading Scan Backtest",
      description:
        "Backtest the sector scanner over N months. Simulates rolling monthly " +
        "scanner recommendations and measures forward returns vs SPY. Reports " +
        "hit rate, excess return, information ratio, and Moskowitz-Grinblatt benchmark.",
      parameters: ScanBacktestParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/scan/backtest", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_execute
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_execute",
      label: "Trading Execute",
      description:
        "Execute trading decisions via Alpaca broker. Runs v3 multi-agent " +
        "pipeline first, then submits orders. DEFAULT is dry_run=true (no real " +
        "orders). Set dry_run=false only when you want to actually trade. " +
        "Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.",
      parameters: ExecuteParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/execute", {
            auto_decide: true,
            dry_run: params.dry_run !== false,  // default true for safety
            portfolio_value: params.portfolio_value || 100000,
          });
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_health
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_health",
      label: "Trading Health",
      description:
        "Check whether the Python trading engine server is running and healthy. " +
        "Optionally attempt to start the server if it is not running.",
      parameters: HealthParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const startIfDown = params?.start_if_down !== false;

          // First, try to reach the existing server
          const healthResult = await callApi("GET", "/health");
          if (healthResult.ok) {
            return jsonResult({
              healthy: true,
              server_running: true,
              port: getServerPort(),
              ...(healthResult.data && typeof healthResult.data === "object"
                ? (healthResult.data as Record<string, unknown>)
                : {}),
            });
          }

          // Server is not reachable
          if (!startIfDown) {
            return jsonResult({
              healthy: false,
              server_running: isServerRunning(),
              port: getServerPort(),
              error: healthResult.error,
            });
          }

          // Attempt to start the server
          api.logger.info("[trading-engine] Server not reachable, attempting to start...");
          await startPythonServer(config.pythonPath, config.serverPort);
          const ready = await waitForReady(config.serverPort, 30_000);

          if (ready) {
            const freshHealth = await callApi("GET", "/health");
            return jsonResult({
              healthy: true,
              server_running: true,
              port: getServerPort(),
              started_now: true,
              ...(freshHealth.data && typeof freshHealth.data === "object"
                ? (freshHealth.data as Record<string, unknown>)
                : {}),
            });
          }

          return jsonResult({
            healthy: false,
            server_running: isServerRunning(),
            port: getServerPort(),
            error: "Server started but did not become ready within 30 seconds",
          });
        } catch (err) {
          return jsonResult({
            healthy: false,
            server_running: isServerRunning(),
            port: getServerPort(),
            error: err instanceof Error ? err.message : String(err),
          });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_stream_status
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_stream_status",
      label: "Trading Stream Status",
      description:
        "Check WebSocket real-time stream status: connected clients, " +
        "bridge connection, recent events.",
      parameters: StreamStatusParams,

      async execute(_toolCallId: string, _params: Record<string, unknown>) {
        try {
          const wsStatus = await callApi("GET", "/ws/status");
          const eventsResult = await callApi("GET", "/events/recent?n=10");
          return jsonResult({
            bridge_connected: isWsConnected(),
            server_stream: wsStatus.ok ? wsStatus.data : { error: wsStatus.error },
            recent_events: eventsResult.ok ? eventsResult.data : { error: eventsResult.error },
          });
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_quant_rank
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_quant_rank",
      label: "Trading Quant Rank",
      description:
        "Run QuantAgent cross-sectional XGBoost ranking on all ~64 stocks. " +
        "Returns P(top-quartile) for each stock, sorted by probability. " +
        "Uses the validated v8 z-scored strategy.",
      parameters: QuantRankParams,

      async execute(_toolCallId: string, _params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/quant/rank", {});
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_quant_retrain
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_quant_retrain",
      label: "Trading Quant Retrain",
      description:
        "Retrain the QuantAgent XGBoost classifier on an 18-month rolling " +
        "window. Should be called weekly for optimal performance.",
      parameters: QuantRetrainParams,

      async execute(_toolCallId: string, _params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/quant/retrain", {});
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_risk_check
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_risk_check",
      label: "Trading Risk Check",
      description:
        "Manually trigger risk check for all held Alpaca positions. " +
        "Evaluates take-profit (+4%), stop-loss (-2.5%), and trailing stop " +
        "triggers. Auto-submits SELL orders when triggered (dry_run by default). " +
        "Requires Alpaca broker connection.",
      parameters: RiskCheckParams,

      async execute(_toolCallId: string, _params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/risk/check", {});
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Tool: trading_pipeline_backtest
    // ------------------------------------------------------------------

    api.registerTool({
      name: "trading_pipeline_backtest",
      label: "Trading Pipeline Backtest",
      description:
        "Run the full v3 pipeline backtest: sector scan → XGBoost ranking → " +
        "weighted signal synthesis → position management with EXIT rules. " +
        "Reports total return, Sharpe, max drawdown, win rate, and alpha vs SPY. " +
        "Long-running operation (~1-3 min).",
      parameters: PipelineBacktestParams,

      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const result = await callApi("POST", "/backtest/pipeline", params);
          if (!result.ok) {
            return jsonResult({ error: result.error, status: result.status });
          }
          return jsonResult(result.data);
        } catch (err) {
          return jsonResult({ error: err instanceof Error ? err.message : String(err) });
        }
      },
    });

    // ------------------------------------------------------------------
    // Gateway methods (for programmatic / Lobster workflow access)
    // ------------------------------------------------------------------

    api.registerGatewayMethod(
      "trading.predict",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/predict", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.train",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/train", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.backtest",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/backtest", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.status",
      async ({ params, respond }) => {
        try {
          const verbose =
            params && typeof params === "object" && (params as Record<string, unknown>).verbose === true;
          const urlPath = verbose ? "/status?verbose=true" : "/status";
          const result = await callApi("GET", urlPath);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.scan",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/scan", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.decide",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/decide/v3", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.regime",
      async ({ params: _params, respond }) => {
        try {
          const result = await callApi("GET", "/regime");
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.scan_backtest",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/scan/backtest", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.health",
      async ({ params: _params, respond }) => {
        try {
          const result = await callApi("GET", "/health");
          if (!result.ok) {
            respond(false, { error: result.error, server_running: isServerRunning() });
            return;
          }
          respond(true, { healthy: true, port: getServerPort(), ...((result.data as object) ?? {}) });
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.quant_rank",
      async ({ params: _params, respond }) => {
        try {
          const result = await callApi("POST", "/quant/rank", {});
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.stream_status",
      async ({ params: _params, respond }) => {
        try {
          const wsResult = await callApi("GET", "/ws/status");
          respond(true, {
            bridge_connected: isWsConnected(),
            server_stream: wsResult.ok ? wsResult.data : null,
          });
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.risk_check",
      async ({ params: _params, respond }) => {
        try {
          const result = await callApi("POST", "/risk/check", {});
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    api.registerGatewayMethod(
      "trading.pipeline_backtest",
      async ({ params, respond }) => {
        try {
          const result = await callApi("POST", "/backtest/pipeline", params);
          if (!result.ok) {
            respond(false, { error: result.error });
            return;
          }
          respond(true, result.data);
        } catch (err) {
          respond(false, { error: err instanceof Error ? err.message : String(err) });
        }
      },
    );

    // ------------------------------------------------------------------
    // CLI: trading engine subcommands
    // ------------------------------------------------------------------

    api.registerCli(
      ({ program }) => {
        const cmd = program
          .command("trading")
          .description("AI Trading Engine commands");

        cmd
          .command("health")
          .description("Check Python server health")
          .action(async () => {
            const result = await callApi("GET", "/health");
            if (result.ok) {
              console.log("Trading engine is healthy");
              console.log(JSON.stringify(result.data, null, 2));
            } else {
              console.error(`Trading engine is not healthy: ${result.error}`);
              process.exitCode = 1;
            }
          });

        cmd
          .command("status")
          .description("Get trading engine status")
          .option("--verbose", "Include detailed information")
          .action(async (opts: { verbose?: boolean }) => {
            const urlPath = opts.verbose ? "/status?verbose=true" : "/status";
            const result = await callApi("GET", urlPath);
            if (result.ok) {
              console.log(JSON.stringify(result.data, null, 2));
            } else {
              console.error(`Failed to get status: ${result.error}`);
              process.exitCode = 1;
            }
          });
      },
      { commands: ["trading"] },
    );

    // ------------------------------------------------------------------
    // Chat command: /trading
    // ------------------------------------------------------------------

    api.registerCommand({
      name: "trading",
      description: "Check trading engine status and server health",
      acceptsArgs: true,
      handler: async (ctx) => {
        const args = ctx.args?.trim() ?? "";

        if (!args || args === "status") {
          const result = await callApi("GET", "/status");
          if (result.ok) {
            return { text: `Trading Engine Status:\n\`\`\`json\n${JSON.stringify(result.data, null, 2)}\n\`\`\`` };
          }
          return { text: `Trading engine is not reachable: ${result.error}` };
        }

        if (args === "health") {
          const result = await callApi("GET", "/health");
          if (result.ok) {
            return { text: "Trading engine is healthy and running." };
          }
          return { text: `Trading engine is not healthy: ${result.error}` };
        }

        return {
          text: [
            "Trading engine commands:",
            "",
            "/trading status - Show engine status",
            "/trading health - Check server health",
          ].join("\n"),
        };
      },
    });
  },
};

export default tradingEnginePlugin;
