"""Claude Agent — PRIMARY decision maker for the trading engine.

The trading engine collects market data 24/7. Claude (Sonnet) makes ALL
trading decisions: BUY, SELL, HOLD, position sizing, and exit timing.

Architecture:
    Market Events → Data Collection → Claude Agent → Execute
                                        ↑                ↓
                                        └── TS feedback ←┘

Claude receives a full MarketSnapshot containing:
- OHLCV summary (price action, volume, recent returns)
- Derivatives context (funding, OI, CVD, L/S ratio)
- Regime state (crypto regime + FRED macro quadrant)
- Thompson Sampling posteriors (which signal categories work in current regime)
- Recent trade history (last 10 trades with PnL, for learning context)
- Open positions (entry price, PnL, held hours)
- Portfolio state (cash, total value)

Rule-based pipeline is NOT used in production. It exists for backtesting only.

Authentication: ANTHROPIC_API_KEY (server) or OAuth (local dev):
    configure_sdk_authentication() → get_sdk_env_vars() → ClaudeAgentOptions(env=...)
If no auth available, falls back to rule-based pipeline (backtesting mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading-engine.claude_agent")

# Lazy SDK import
_SDK_AVAILABLE: Optional[bool] = None


def _check_sdk() -> bool:
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            from claude_agent_sdk import ClaudeSDKClient  # noqa: F401
            _SDK_AVAILABLE = True
        except ImportError:
            _SDK_AVAILABLE = False
            logger.warning("[agent] claude-agent-sdk not installed — falling back to rule-based")
    return _SDK_AVAILABLE


# ------------------------------------------------------------------
# Market Snapshot — single input contract for Claude
# ------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """Everything Claude needs to make a trading decision."""

    # Price action
    ticker_prices: Dict[str, float] = field(default_factory=dict)  # {ticker: last_price}
    ticker_returns_4h: Dict[str, float] = field(default_factory=dict)  # {ticker: 4h % return}
    ticker_returns_24h: Dict[str, float] = field(default_factory=dict)  # {ticker: 24h % return}
    ticker_volumes: Dict[str, float] = field(default_factory=dict)  # {ticker: 24h volume USD}
    btc_price: float = 0.0
    btc_change_1h: float = 0.0
    btc_change_24h: float = 0.0

    # Derivatives
    funding_rates: Dict[str, float] = field(default_factory=dict)  # {ticker: rate}
    open_interest: Dict[str, Any] = field(default_factory=dict)
    taker_delta: Dict[str, Any] = field(default_factory=dict)  # CVD data
    long_short_ratio: Dict[str, Any] = field(default_factory=dict)
    basis_spreads: Dict[str, float] = field(default_factory=dict)  # {ticker: annualized basis %}

    # Regime
    crypto_regime: str = "unknown"  # trending/ranging/volatile
    macro_regime: str = "unknown"  # goldilocks/reflation/stagflation/deflation
    combined_regime: str = "unknown"  # crypto_macro combined
    macro_exposure_scale: float = 1.0  # from MacroRegimeDetector
    macro_trail_multiplier: float = 1.0

    # Thompson Sampling posteriors (RL context)
    ts_mean_weights: Dict[str, float] = field(default_factory=dict)  # {agent: mean_weight}
    ts_regime_info: str = ""  # human-readable TS summary for the current regime
    ts_total_trades: int = 0
    ts_cumulative_pnl_pct: float = 0.0

    # Recent trades (learning context)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)  # last 10

    # Open positions
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {ticker: {qty, entry_price, pnl_pct, held_hours}}

    # Portfolio
    portfolio_value: float = 0.0
    cash: float = 0.0
    candidates: List[str] = field(default_factory=list)  # tradeable tickers

    # Event trigger
    trigger: str = "manual"  # candle_close / significant_move / oi_spike / manual

    # Sentiment & News
    fear_greed_index: int = 0  # 0-100, from alternative.me
    fear_greed_label: str = ""  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    news_summary: str = ""  # trending coins / market buzz
    etf_daily_flow_usd: float = 0.0  # BTC ETF net daily flow in USD
    etf_flow_label: str = ""  # "inflow" / "outflow" / "neutral"

    # Liquidation & Stablecoin
    liquidation_usd_1h: float = 0.0  # total forced liquidation volume in 1h
    liquidation_cascade: bool = False  # >$5M liquidated in 1h
    stablecoin_mcap_b: float = 0.0  # total stablecoin market cap in $B
    stablecoin_change_pct: float = 0.0  # supply change since last check

    # Multi-timeframe analysis (from MultiTFAggregator)
    multi_tf_summary: Dict[str, str] = field(default_factory=dict)  # {ticker: formatted_text}

    # Agent-autonomous gate context
    gate_wake_reasons: List[str] = field(default_factory=list)  # why the gate woke Claude
    agent_memory: str = ""  # Claude's note from previous wake

    # Historical insights (from Graphiti memory)
    historical_insights: str = ""

    def to_prompt(self) -> str:
        """Serialize into a Claude-readable prompt section."""
        lines = []

        # Price action
        lines.append("## Current Prices & Returns")
        for tic in self.candidates:
            px = self.ticker_prices.get(tic, 0)
            r4h = self.ticker_returns_4h.get(tic, 0)
            r24h = self.ticker_returns_24h.get(tic, 0)
            vol = self.ticker_volumes.get(tic, 0)
            lines.append(f"  {tic}: ${px:,.2f} (1bar: {r4h:+.2%}, 24h: {r24h:+.2%}, vol: ${vol:,.0f})")
        lines.append(f"  BTC: ${self.btc_price:,.2f} (1h: {self.btc_change_1h:+.2%}, 24h: {self.btc_change_24h:+.2%})")

        # Multi-timeframe analysis
        if self.multi_tf_summary:
            lines.append("\n## Multi-Timeframe Analysis")
            for tic in self.candidates:
                tf_text = self.multi_tf_summary.get(tic)
                if tf_text:
                    lines.append(f"  {tic}:")
                    for tf_line in tf_text.split("\n"):
                        lines.append(f"  {tf_line}")

        # Derivatives
        lines.append("\n## Derivatives")
        for tic in self.candidates:
            fr = self.funding_rates.get(tic, 0)
            td = self.taker_delta.get(tic, {})
            ls = self.long_short_ratio.get(tic, {})
            basis = self.basis_spreads.get(tic)
            parts = [f"funding={fr:+.4f}",
                     f"CVD_ratio={td.get('buy_sell_ratio', 'N/A')}",
                     f"L/S_zscore={ls.get('z_score', 'N/A')}"]
            if basis is not None:
                parts.append(f"basis={basis:+.1%} ann.")
            lines.append(f"  {tic}: {', '.join(parts)}")

        # Liquidation & Stablecoin
        if self.liquidation_usd_1h > 0 or self.stablecoin_mcap_b > 0:
            lines.append("\n## Market Structure")
            if self.liquidation_usd_1h > 0:
                lines.append(f"  Liquidations (1h): ${self.liquidation_usd_1h/1e6:.1f}M")
                if self.liquidation_cascade:
                    lines.append("  !! LIQUIDATION CASCADE — forced exits accelerating, expect volatility spike")
            if self.stablecoin_mcap_b > 0:
                lines.append(f"  Stablecoin Supply: ${self.stablecoin_mcap_b:.1f}B (chg: {self.stablecoin_change_pct:+.2%})")
                if self.stablecoin_change_pct > 0.01:
                    lines.append("  Stablecoin inflow — fresh buying power entering crypto")
                elif self.stablecoin_change_pct < -0.01:
                    lines.append("  Stablecoin outflow — capital leaving crypto ecosystem")

        # Regime
        lines.append(f"\n## Market Regime")
        lines.append(f"  Crypto: {self.crypto_regime}")
        lines.append(f"  Macro: {self.macro_regime} (exposure_scale={self.macro_exposure_scale:.2f})")
        lines.append(f"  Combined: {self.combined_regime}")

        # TS posteriors
        lines.append(f"\n## Reinforcement Learning Context (Thompson Sampling)")
        lines.append(f"  Total trades: {self.ts_total_trades}, Cumulative PnL: {self.ts_cumulative_pnl_pct:+.2f}%")
        if self.ts_mean_weights:
            lines.append(f"  Signal reliability in {self.combined_regime}:")
            for agent, w in sorted(self.ts_mean_weights.items(), key=lambda x: -x[1]):
                pct = w * 100
                lines.append(f"    {agent}: {pct:.1f}% (higher = more reliable in this regime)")
        if self.ts_regime_info:
            lines.append(f"  {self.ts_regime_info}")

        # Recent trades
        if self.recent_trades:
            lines.append(f"\n## Recent Trade History (last {len(self.recent_trades)})")
            for t in self.recent_trades:
                lines.append(f"  {t.get('ticker','?')}: PnL {t.get('pnl_pct',0):+.2%}, "
                             f"regime={t.get('regime','?')}, held {t.get('held_hours',0):.0f}h")

        # Positions
        lines.append(f"\n## Open Positions")
        if self.positions:
            for tic, pos in self.positions.items():
                peak_pnl = 0.0
                if pos.get("trailing_high", 0) > 0 and pos.get("entry_price", 0) > 0:
                    peak_pnl = (pos["trailing_high"] - pos["entry_price"]) / pos["entry_price"]
                lines.append(f"  {tic}: qty={pos.get('qty',0):.4f}, "
                             f"entry=${pos.get('entry_price',0):,.2f}, "
                             f"PnL={pos.get('pnl_pct',0):+.2%}, "
                             f"peak={peak_pnl:+.2%}, "
                             f"held={pos.get('held_hours',0):.0f}h")
            lines.append("  Safety net: hard_SL=-5%, trail activates at +8%, trail width=12%")
            lines.append("  YOU must exit BEFORE safety triggers. Safety = your failure.")
        else:
            lines.append("  No open positions")

        # Portfolio
        lines.append(f"\n## Portfolio")
        lines.append(f"  Value: ${self.portfolio_value:,.2f}, Cash: ${self.cash:,.2f}")
        lines.append(f"  Trigger: {self.trigger}")

        # Sentiment
        if self.fear_greed_index > 0 or self.etf_daily_flow_usd != 0:
            lines.append(f"\n## Market Sentiment")
            if self.fear_greed_index > 0:
                lines.append(f"  Fear & Greed Index: {self.fear_greed_index}/100 ({self.fear_greed_label})")
                if self.fear_greed_index <= 20:
                    lines.append("  ⚠ EXTREME FEAR — historically a contrarian buy signal")
                elif self.fear_greed_index >= 80:
                    lines.append("  ⚠ EXTREME GREED — historically a contrarian sell signal")
            if self.etf_daily_flow_usd != 0:
                lines.append(f"  BTC ETF Daily Flow: ${self.etf_daily_flow_usd/1e6:+.1f}M ({self.etf_flow_label})")

        if self.news_summary:
            lines.append(f"\n## Recent Crypto News")
            lines.append(f"  {self.news_summary}")

        # Wake reasons (from AdaptiveGate)
        if self.gate_wake_reasons:
            lines.append(f"\n## Wake Reasons")
            for r in self.gate_wake_reasons:
                lines.append(f"  - {r}")

        # Agent memory (from previous Claude response)
        if self.agent_memory:
            lines.append(f"\n## Your Previous Note")
            lines.append(f"  {self.agent_memory}")

        # Historical insights (from Graphiti memory)
        if self.historical_insights:
            lines.append(f"\n## Historical Insights")
            lines.append(f"  {self.historical_insights}")

        return "\n".join(lines)


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous crypto swing trading agent running 24/7 on Binance.
You make ALL trading decisions. There are no rules — only your judgment informed by data.

**Trading frequency target: 1-3 trades per day.** You are a swing trader, not a scalper.
Look for high-conviction setups with favorable risk/reward.

You receive a MarketSnapshot with:
- Multi-timeframe analysis (15m/1h/4h candles with RSI, EMA cross, volume trends)
- Price action (returns, volume)
- Derivatives (funding rate, CVD, long/short ratio)
- Market regime (crypto + macro)
- Thompson Sampling RL posteriors (which signal types are reliable in current regime)
- Recent trade history (learn from past wins and losses)
- Open positions and portfolio state

**Multi-timeframe approach:**
- 4h: Identify the big picture trend and major support/resistance
- 1h: Confirm trend direction and find swing entry zones
- 15m: Fine-tune entry/exit timing within the swing

Your job:
1. Analyze ALL timeframes holistically — higher TF trend > lower TF noise
2. Decide: BUY, SELL, or HOLD for each candidate ticker
3. For BUY: specify position_pct (0.05 to 0.20 of portfolio)
4. For SELL: specify which position to close and why
5. Learn from the TS posteriors — if momentum signals are unreliable in this regime, weight them down

**EXIT MANAGEMENT (CRITICAL):**
You MUST proactively manage exits. The safety layer (hard SL -5%, trailing stop, time stop) is the LAST RESORT — if safety exits your position, YOU failed.
Core exit principles (adapt these through experience, learn from /trade-reflect):
- Thesis invalidated (support broke, regime changed) → SELL immediately
- Target +3-8% profit on swings. Protect gains: +2% peak dropping to +0.5% → SELL
- Losing with no catalyst: PnL < -3% → cut early, don't wait for -5% safety stop
- Dead money: >12h with PnL < +1% in ranging market → SELL
Your R:R ratio matters more than win rate. Cut losers fast, let winners run.

Rules you MUST follow:
- Never risk more than 20% of portfolio on a single position (smaller size for higher frequency)
- Respect the macro exposure_scale (reduce sizing in stagflation/deflation)
- If Thompson Sampling shows a signal type has <40% reliability, be skeptical of that signal
- Provide clear reasoning for every decision
- Use 15m RSI for timing, 1h for trend confirmation, 4h for big picture

## Self-scheduling
After each decision, you MUST set your own monitoring schedule:

**next_check_seconds**: How many seconds until you want to be woken again.
- Quiet, ranging market: 600-3600 (10 min - 1 hour)
- Active trend: 300-600 (5-10 min)
- High volatility / near entry/exit: 120-300 (2-5 min)
- Choose based on how fast conditions are changing.

**wake_conditions**: Conditions that should wake you BEFORE the timer.
Each condition has: metric (feature name), operator (gt/lt/crosses_above/crosses_below/abs_change_pct_gt), threshold, reason.
Available metrics: btc_price, price_change_pct, volume, funding_rate, oi_change_pct.
Set 0-3 conditions. Example: wake me if BTC drops below 95000.

**memory_update**: A short note (max 500 chars) for yourself. You'll see it as "Your Previous Note" next time you wake up. Use it to track key levels, developing patterns, or pending decisions.

**signal_weights** (per decision): Rate how much each signal category influenced your decision.
Values from -1.0 (strong contrarian) to +1.0 (strong confirming). Only include signals you actually used.
Categories: momentum, funding_rate, oi_signal, regime, macro, sentiment, market, quant.
This feeds Thompson Sampling RL — honest assessment helps learn which signals work in which regimes.

Respond ONLY with valid JSON:
{
  "decisions": [
    {"ticker": "BTC/USDT:USDT", "action": "BUY"|"SELL"|"HOLD", "position_pct": 0.0-0.20, "confidence": 0.0-1.0, "reasoning": "...", "signal_weights": {"momentum": 0.7, "funding_rate": -0.3}},
    ...
  ],
  "market_assessment": "one sentence overall market view",
  "regime_agreement": true|false,
  "learning_note": "what I learned from recent trade history",
  "next_check_seconds": 600,
  "wake_conditions": [{"metric": "btc_price", "operator": "lt", "threshold": 95000, "reason": "support break"}],
  "memory_update": "BTC testing 95K support, watching for breakdown"
}"""


# ------------------------------------------------------------------
# Claude Agent
# ------------------------------------------------------------------

class ClaudeAgent:
    """Primary trading decision maker using Claude Sonnet via SDK.

    NOT an advisor. NOT optional. This IS the trader.
    Falls back to rule-based pipeline ONLY if no auth is available.

    Auth: ANTHROPIC_API_KEY (server) or OAuth (local dev).
        configure_sdk_authentication() → get_sdk_env_vars() → ClaudeAgentOptions(env=...)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._available: Optional[bool] = None
        self._auth_configured: bool = False
        self._consecutive_failures: int = 0
        self._max_failures: int = 3  # circuit breaker
        self._circuit_open_time: float = 0.0
        self._circuit_cooldown: float = 900.0  # 15 min half-open retry

    def _ensure_auth(self) -> bool:
        """Configure SDK authentication once (OpenClaw pattern)."""
        if self._auth_configured:
            return True
        try:
            from core.claude_auth import configure_sdk_authentication
            configure_sdk_authentication()
            self._auth_configured = True
            return True
        except ValueError as exc:
            logger.warning("[agent] Auth failed: %s", exc)
            return False
        except Exception as exc:
            logger.error("[agent] Auth error: %s", exc)
            return False

    @property
    def is_available(self) -> bool:
        """Check if Claude agent is available (SDK + OAuth + circuit breaker).

        Circuit breaker uses half-open pattern:
        - 3 consecutive failures → open (block all calls)
        - After 15 min cooldown → half-open (allow one retry, re-auth)
        - If retry succeeds → closed (normal operation)
        - If retry fails → open again (restart cooldown)
        """
        if self._consecutive_failures >= self._max_failures:
            if self._circuit_open_time <= 0:
                self._circuit_open_time = time.time()
                logger.warning("[agent] Circuit breaker OPEN (%d failures)", self._consecutive_failures)
            elapsed = time.time() - self._circuit_open_time
            if elapsed < self._circuit_cooldown:
                return False
            # Half-open: reset and force re-authentication (token may have refreshed)
            logger.info("[agent] Circuit breaker HALF-OPEN after %.0fs, retrying", elapsed)
            self._consecutive_failures = 0
            self._circuit_open_time = 0.0
            self._auth_configured = False
            self._available = None

        if self._available is None:
            if not _check_sdk():
                self._available = False
            elif not self._ensure_auth():
                self._available = False
            else:
                self._available = True
                auth_mode = "API key" if os.environ.get("ANTHROPIC_API_KEY") else "OAuth"
                logger.info("[agent] Claude agent ready (Sonnet, %s)", auth_mode)
        return self._available

    def reset_circuit_breaker(self) -> None:
        """Reset after a successful call."""
        self._consecutive_failures = 0
        self._circuit_open_time = 0.0

    async def _call_sdk(self, system_prompt: str, prompt: str) -> Optional[str]:
        """Single SDK call — shared by decide() and evaluate_position().

        Uses OpenClaw's auth pattern: env vars passed to ClaudeAgentOptions,
        NOT manually set on os.environ per-call.
        """
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ClaudeSDKClient,
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )
        from core.claude_auth import get_sdk_env_vars

        sdk_env = get_sdk_env_vars()
        # Allow nested sessions (when launched from within Claude Code)
        os.environ.pop("CLAUDECODE", None)
        response_text = ""

        async def _run():
            nonlocal response_text
            client = ClaudeSDKClient(options=ClaudeAgentOptions(
                model=self._model,
                system_prompt=system_prompt,
                allowed_tools=[],
                max_turns=1,
                env=sdk_env,
            ))
            await client.connect()
            try:
                await client.query(prompt)
                try:
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            # Check for API errors returned as messages (Issue #472)
                            if hasattr(msg, "error") and msg.error:
                                logger.warning("[agent] API error: %s", msg.error)
                                continue
                            for block in msg.content:
                                if isinstance(block, TextBlock):
                                    response_text += block.text
                        # receive_response() stops at ResultMessage automatically;
                        # do NOT break early — causes asyncio cleanup issues
                except Exception as inner_exc:
                    # SDK may not handle all event types (e.g. rate_limit_event)
                    if response_text:
                        logger.warning("[agent] Stream interrupted (%s), using partial response", inner_exc)
                    else:
                        raise
            finally:
                await client.disconnect()

        await asyncio.wait_for(_run(), timeout=self._timeout)
        return response_text or None

    async def decide(self, snapshot: MarketSnapshot) -> Optional[Dict[str, Any]]:
        """Make trading decisions from a full MarketSnapshot.

        Returns:
            {
                "decisions": [{"ticker", "action", "position_pct", "confidence", "reasoning"}],
                "market_assessment": str,
                "regime_agreement": bool,
                "learning_note": str,
            }
            or None if unavailable (caller should use rule-based fallback).
        """
        if not self.is_available:
            return None

        prompt = f"Here is the current market state. Make your trading decisions.\n\n{snapshot.to_prompt()}"

        try:
            response_text = await self._call_sdk(SYSTEM_PROMPT, prompt)
            if not response_text:
                self._consecutive_failures += 1
                return None

            result = self._parse_json(response_text)
            if result and "decisions" in result:
                self.reset_circuit_breaker()
                logger.info("[agent] Decision: %s", result.get("market_assessment", ""))
                return result
            else:
                logger.warning("[agent] Invalid response format: %s", response_text[:300])
                self._consecutive_failures += 1
                return None

        except asyncio.TimeoutError:
            logger.warning("[agent] Timeout after %.0fs", self._timeout)
            self._consecutive_failures += 1
            return None
        except Exception as exc:
            logger.error("[agent] Call failed: %s", exc)
            self._consecutive_failures += 1
            return None

    async def evaluate_position(
        self,
        position_data: Dict[str, Any],
        market_context: Dict[str, Any],
        ts_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Ask Claude to evaluate an open position (for PositionTracker).

        Returns:
            {"verdict": "HOLD"|"EXIT"|"ADD", "confidence": float, "reasoning": str}
        """
        if not self.is_available:
            return None

        prompt = (
            "Evaluate this open position. Should I HOLD, EXIT, or ADD?\n\n"
            f"## Position\n{json.dumps(position_data, default=str)}\n\n"
            f"## Market Context\n{json.dumps(market_context, default=str)}\n\n"
            f"## RL Context (Thompson Sampling)\n{json.dumps(ts_context, default=str)}\n\n"
            'Respond with JSON: {"verdict": "HOLD"|"EXIT"|"ADD", '
            '"confidence": 0.0-1.0, "reasoning": "..."}'
        )

        try:
            system = "You are a crypto position evaluator. Respond ONLY with JSON."
            response_text = await self._call_sdk(system, prompt)
            if not response_text:
                self._consecutive_failures += 1
                return None

            result = self._parse_json(response_text)
            if result:
                self.reset_circuit_breaker()
            return result

        except Exception as exc:
            logger.warning("[agent] Position eval failed: %s", exc)
            self._consecutive_failures += 1
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from Claude's response."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
