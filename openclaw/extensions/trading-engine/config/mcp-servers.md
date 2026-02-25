# MCP Server Configuration Guide

Required and optional MCP servers for the Trading Engine.

## Required MCP Servers

### 1. Graphiti (Memory)

Already configured in your OpenClaw setup. Used for:
- Trade history storage (`group_id: "trading-portfolio"`)
- Learned patterns (`group_id: "trading-patterns"`)
- Research insights (`group_id: "trading-research"`)

No additional setup needed.

## Optional MCP Servers (Recommended)

### 2. Alpha Vantage MCP (Real-time Data)

Provides real-time and historical stock data.

```json
{
  "mcpServers": {
    "alpha-vantage": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-alpha-vantage"],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Get API key: https://www.alphavantage.co/support/#api-key (free tier: 25 req/day)

### 3. Alpaca MCP (Broker - Paper Trading)

Provides paper/live trading execution.

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-alpaca"],
      "env": {
        "ALPACA_API_KEY": "your-paper-key",
        "ALPACA_SECRET_KEY": "your-paper-secret",
        "ALPACA_PAPER": "true"
      }
    }
  }
}
```

Get API keys: https://app.alpaca.markets/paper/dashboard/overview

### 4. Financial Datasets MCP (News & Fundamentals)

Provides financial news and fundamental data.

```json
{
  "mcpServers": {
    "financial-datasets": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-financial-datasets"],
      "env": {
        "FINANCIAL_DATASETS_API_KEY": "your-api-key"
      }
    }
  }
}
```

### 5. MaverickMCP (Advanced Analysis)

29+ tools for technical analysis and VectorBT backtesting.

```json
{
  "mcpServers": {
    "maverick": {
      "command": "uvx",
      "args": ["maverick-mcp"],
      "env": {
        "MAVERICK_API_KEY": "your-api-key"
      }
    }
  }
}
```

### 6. arxiv-mcp (Research)

Already configured. Used for monitoring latest trading/ML research papers.

## Configuration Location

Add MCP server configs to your OpenClaw config:
- **Global**: `~/.openclaw/config.json` under `"mcpServers"` key
- **Project**: `.openclaw/config.json` in project root

## Verification

After adding MCP servers, verify they're connected:

```bash
# In OpenClaw chat
/mcp status

# Should show all servers as "connected"
```
