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

    # Macro indicators (FRED + yfinance)
    dxy: float = 0.0  # US Dollar Index (BTC inverse corr -0.4~-0.8)
    dxy_direction: str = "unknown"  # strengthening/weakening/stable
    dxy_1m_pct: float = 0.0  # DXY 1-month change %
    net_liquidity_direction: str = "unknown"  # expanding/contracting/flat
    net_liquidity_delta_pct: float = 0.0  # Fed Net Liquidity 2-week change %
    financial_stress: str = "unknown"  # elevated/normal/easing (HY spread)
    nfci: float = 0.0  # Chicago Fed NFCI (negative=loose, positive=tight)

    # Hierarchical Thompson Sampling posteriors (RL context)
    ts_mean_weights: Dict[str, float] = field(default_factory=dict)  # {signal: mean_weight}
    ts_group_weights: Dict[str, float] = field(default_factory=dict)  # {group: mean_weight}
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

    # Safety config (populated from ACTIVE_BLEND_CONFIG for prompt consistency)
    safety_config: Dict[str, float] = field(default_factory=dict)

    # Agent-autonomous gate context
    gate_wake_reasons: List[str] = field(default_factory=list)  # why the gate woke Claude
    agent_memory: str = ""  # Claude's note from previous wake

    # Pre-computed technical indicators per ticker per timeframe
    # {ticker: {interval: {rsi, stoch_rsi, macd, ema_cross, bollinger, atr, ...}}}
    pre_computed_ta: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # Historical insights (from Graphiti memory)
    historical_insights: str = ""

    # Level 0 meta-parameters (H-TS learned per regime)
    ts_meta_params: Dict[str, float] = field(default_factory=dict)

    # Trading diary: reflections + semantic lessons (3-layer self-reflection)
    diary_context: str = ""

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
            lines.append(f"  {tic}: ${px:,.2f} (4h: {r4h:+.2%}, 24h: {r24h:+.2%}, vol: ${vol:,.0f})")
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

        # Pre-computed Technical Indicators
        if self.pre_computed_ta:
            lines.append("\n## Technical Indicators (precise, pre-computed)")
            for tic in self.candidates:
                ta_data = self.pre_computed_ta.get(tic)
                if not ta_data:
                    continue
                lines.append(f"  {tic}:")
                for interval in ("5m", "15m", "1h"):
                    ind = ta_data.get(interval)
                    if not ind:
                        continue
                    rsi7 = ind.get("rsi", "N/A")
                    rsi14 = ind.get("rsi_14", "N/A")
                    rsi_div = ind.get("rsi_divergence", "N/A")
                    stoch = ind.get("stoch_rsi", "N/A")
                    macd_h = ind.get("macd", {}).get("histogram", "N/A")
                    ema = ind.get("ema_cross", {})
                    bb = ind.get("bollinger", {})
                    atr_pct = ind.get("atr_pct", "N/A")
                    vwap_dev = ind.get("vwap", {}).get("deviation_pct", "N/A")
                    lines.append(
                        f"    {interval}: RSI7={rsi7} RSI14={rsi14} div={rsi_div} StochRSI={stoch} "
                        f"MACD_hist={macd_h} EMA={ema.get('status', '?')}({ema.get('gap_pct', 0):+.4f}) "
                        f"BB_%B={bb.get('pct_b', 'N/A')} ATR%={atr_pct} VWAP_dev={vwap_dev}"
                    )
                    atr_sl = ind.get("atr_sl")
                    if atr_sl:
                        lines.append(
                            f"      ATR exits: SL~{atr_sl:.4f} TP~{ind.get('atr_tp', 0):.4f} "
                            f"trail~{ind.get('atr_trail', 0):.4f} fee/ATR={ind.get('fee_atr_ratio', 'N/A')}"
                        )

        # Derivatives
        lines.append("\n## Derivatives")
        for tic in self.candidates:
            fr = self.funding_rates.get(tic, 0)
            td = self.taker_delta.get(tic, {})
            ls = self.long_short_ratio.get(tic, {})
            basis = self.basis_spreads.get(tic)
            oi = self.open_interest.get(tic, {})

            # Funding 해석
            if abs(fr) > 0.0003:
                f_label = "LONG_CROWDED" if fr > 0 else "SHORT_CROWDED"
            elif abs(fr) > 0.0001:
                f_label = "mild_long" if fr > 0 else "mild_short"
            else:
                f_label = "neutral"

            parts = [f"funding={fr:+.4f}({f_label})",
                     f"CVD={td.get('buy_sell_ratio', 'N/A')}",
                     f"L/S_z={ls.get('z_score', 'N/A')}"]

            # OI (기존에 누락됨)
            if isinstance(oi, dict) and "direction" in oi:
                parts.append(f"OI={oi['direction']}({oi.get('change_pct', 0):+.1%})")

            if basis is not None:
                parts.append(f"basis={basis:+.1%}")

            # 콤보 시그널
            if isinstance(oi, dict) and abs(fr) > 0.0001:
                oi_dir = oi.get("direction", "stable")
                if fr < -0.0001 and oi_dir == "decreasing":
                    parts.append("!! SHORT_SQUEEZE_SETUP")
                elif fr > 0.0003 and oi_dir == "decreasing":
                    parts.append("!! LONG_UNWIND")

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
        lines.append(f"\n## Macro Indicators")
        lines.append(f"  DXY (Dollar Index): {self.dxy:.1f} ({self.dxy_direction}, 1m: {self.dxy_1m_pct:+.1f}%) — BTC inverse correlation")
        lines.append(f"  Fed Net Liquidity: {self.net_liquidity_direction} (Δ{self.net_liquidity_delta_pct:+.2f}%) — BTC correlates 0.6~0.9")
        lines.append(f"  Financial Stress: {self.financial_stress} (NFCI: {self.nfci:.3f})")

        # H-TS posteriors — MOST IMPORTANT SECTION for Claude's decision-making
        lines.append(f"\n## SIGNAL RELIABILITY (Hierarchical Thompson Sampling — TRUST THIS DATA)")
        lines.append(f"  Total trades: {self.ts_total_trades}, Cumulative PnL: {self.ts_cumulative_pnl_pct:+.2f}%")
        lines.append(f"  Regime: {self.combined_regime}")

        # Group-level weights (Level 1)
        if self.ts_group_weights:
            lines.append(f"\n  GROUP RELIABILITY (Level 1 — which analysis category works?):")
            for group, gw in sorted(self.ts_group_weights.items(), key=lambda x: -x[1]):
                pct = gw * 100
                if pct > 18:
                    tag = "★ STRONG"
                elif pct > 15:
                    tag = "average"
                else:
                    tag = "weak"
                lines.append(f"    {group}: {pct:.1f}% — {tag}")

        # Signal-level weights (Level 2) — show top signals
        if self.ts_mean_weights:
            lines.append(f"\n  TOP SIGNALS (Level 2 — specific indicators, sorted by reliability):")
            sorted_sigs = sorted(self.ts_mean_weights.items(), key=lambda x: -x[1])
            for sig, sw in sorted_sigs[:12]:  # show top 12
                pct = sw * 100
                lines.append(f"    {sig}: {pct:.1f}%")
            if len(sorted_sigs) > 12:
                lines.append(f"    ... +{len(sorted_sigs) - 12} more signals")

        if self.ts_regime_info:
            lines.append(f"  {self.ts_regime_info}")

        # Level 0: Meta-parameters — "how to trade" in this regime
        if self.ts_meta_params:
            lines.append(f"\n  REGIME META-PARAMETERS (H-TS learned — how to trade in this regime):")
            _meta_labels = {
                "position_scale":     ("SIZE DOWN",     "SIZE UP"),
                "entry_selectivity":  ("BROAD ENTRY",   "BE PICKY"),
                "hold_patience":      ("EXIT FAST",     "HOLD LONGER"),
                "trade_frequency":    ("SIT OUT",       "BE ACTIVE"),
                "trend_vs_reversion": ("MEAN-REVERT",   "TREND-FOLLOW"),
                "risk_aversion":      ("AGGRESSIVE",    "RISK-OFF"),
            }
            for param, value in self.ts_meta_params.items():
                low_label, high_label = _meta_labels.get(param, ("LOW", "HIGH"))
                label = high_label if value > 0.55 else (low_label if value < 0.45 else "NEUTRAL")
                lines.append(f"    {param}: {value:.2f} ({label})")

        # Recent trades
        if self.recent_trades:
            lines.append(f"\n## Recent Trade History (last {len(self.recent_trades)})")
            for t in self.recent_trades:
                held_h = t.get('held_hours', 0)
                held_str = f"{held_h * 60:.0f}min" if held_h < 1 else f"{held_h:.1f}h"
                lines.append(f"  {t.get('ticker','?')}: PnL {t.get('pnl_pct',0):+.2%}, "
                             f"regime={t.get('regime','?')}, held {held_str}, "
                             f"signals={t.get('agent_signals', {})}")

        # Positions
        lines.append(f"\n## Open Positions")
        if self.positions:
            for tic, pos in self.positions.items():
                peak_pnl = 0.0
                if pos.get("trailing_high", 0) > 0 and pos.get("entry_price", 0) > 0:
                    peak_pnl = (pos["trailing_high"] - pos["entry_price"]) / pos["entry_price"]
                dca_tag = " [DCA]" if pos.get("is_dca") else ""
                lines.append(f"  {tic}{dca_tag}: qty={pos.get('qty',0):.4f}, "
                                 f"entry=${pos.get('entry_price',0):,.2f}, "
                                 f"PnL={pos.get('pnl_pct',0):+.2%}, "
                                 f"peak={peak_pnl:+.2%}, "
                                 f"held={pos.get('held_hours',0):.0f}h")
                ta_data = self.pre_computed_ta.get(tic, {})
                atr5m = ta_data.get("5m", {}).get("atr_pct", 0)
                if atr5m > 0:
                    pnl_in_atr = pos.get('pnl_pct', 0) / atr5m
                    lines.append(
                        f"    volatility: PnL={pnl_in_atr:+.1f}x ATR(5m), ATR(5m)={atr5m:.4f}"
                    )
            sl = self.safety_config.get("stop_loss_pct", -0.007)
            tp = self.safety_config.get("take_profit_pct", 0.008)
            scalp_act = self.safety_config.get("scalp_trail_activation_pct", 0.004)
            scalp_w = self.safety_config.get("scalp_trail_width_pct", 0.002)
            max_hold_m = self.safety_config.get("max_hold_hours", 0.75) * 60
            dca_sl = self.safety_config.get("dca_stop_loss_pct", -0.015)
            dca_hold_m = self.safety_config.get("dca_max_hold_hours", 2.0) * 60
            lines.append(f"  Scalp safety: SL={sl:+.1%}, trail(+{scalp_act:.1%}/{scalp_w:.1%}), time={max_hold_m:.0f}min")
            lines.append(f"  DCA safety: SL={dca_sl:+.1%}, hold={dca_hold_m:.0f}min (after ADD)")
            lines.append(f"  YOU should exit BEFORE safety triggers. Cut losses before SL ({sl:+.1%}). Lock profits before TP ({tp:+.1%}).")
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
                lines.append(f"  (TS shows sentiment reliability at ~12% — use as secondary signal only)")
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

        # Trading Diary (3-layer self-reflection memory)
        if self.diary_context:
            lines.append(f"\n## Trading Diary (Self-Reflection)")
            lines.append(self.diary_context)

        return "\n".join(lines)


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

def build_system_prompt(risk_config: dict = None, round_trip_fee: float = 0.002) -> str:
    """Build system prompt with dynamic config values. No hardcoded strategy thresholds."""
    rc = risk_config or {}
    sl_pct = rc.get("stop_loss_pct", -0.007)
    tp_pct = rc.get("take_profit_pct", 0.008)
    max_hold_min = rc.get("max_hold_hours", 0.75) * 60
    dca_sl_pct = rc.get("dca_stop_loss_pct", -0.015)
    dca_hold_min = rc.get("dca_max_hold_hours", 2.0) * 60
    max_exposure = rc.get("max_exposure_pct", 0.60)
    max_pos = rc.get("max_position_pct", 0.15)
    fee_pct = round_trip_fee * 100  # e.g. 0.2%

    return f"""You are an autonomous crypto SCALPING AGENT on Binance testnet. You have TOOLS — USE THEM before deciding.

**CONTEXT: Testnet (fake money). Primary mission: LEARN by trading and generate Thompson Sampling (TS) feedback.**

## SCALPING STRATEGY — CONCEPTUAL SETUPS (H-TS data overrides everything)
These are conceptual frameworks only. **Do NOT treat any threshold as a hard rule.**
H-TS posteriors tell you which signals actually work in the current regime — trust the data above all.

### Setup A: Mean Reversion Scalp (ranging/low-vol regimes)
**Concept:** Extreme oversold + institutional mean reversion.
**Key signals (more confluence = higher confidence):**
1. StochRSI at extreme oversold on 5m
2. Price at or below lower Bollinger Band on 5m
3. Price below VWAP (institutional mean)
**Strengtheners:** RSI divergence, CVD divergence, volume spike
**Caution:** Weaker when 5m EMA(9) < EMA(21) AND 15m MACD negative (against trend)

### Setup B: Momentum/Breakout Scalp (trending regimes)
**IMPORTANT: High RSI/StochRSI does NOT always mean "overbought = don't buy". In a trend, high RSI is CONFIRMATION of momentum, not a sell signal.**
**Key signals:**
1. EMA(9) crosses above EMA(21) on 5m — momentum confirmed
2. RSI rising and above midline (trend strength, not exhaustion)
3. Price above VWAP AND above upper Bollinger Band (breakout, not "overbought")
4. StochRSI high on 5m WITH higher TF EMA alignment = strong momentum
**Strengtheners:** Rising CVD (net buying), OI increasing, volume surge on breakout candle, multi-TF EMA alignment
**When to AVOID:** Only skip if RSI is high but FALLING (divergence), or volume is dropping (exhaustion)

### Setup C: Liquidation Flush Bounce (high risk, high reward)
**Trigger:** Large liquidation cascade detected
**Entry:** After cascade exhausts — look for CVD divergence + extreme oversold StochRSI
**Caution:** Wait for stabilization (2-3 candles after flush)

### Funding Rate + OI Combo:
- Negative funding + OI decreasing → shorts closing = SHORT SQUEEZE (Long 유리)
- Positive funding + OI decreasing → longs closing = LONG UNWIND (Long 주의)
- Extreme funding + OI increasing → 새 돈 유입 + crowded = 반전 building
(!! SHORT_SQUEEZE_SETUP / !! LONG_UNWIND 라벨이 조건 충족 시 자동 표시됨)

### Fear & Greed: REGIME OVERLAY only — size positions smaller in extreme greed, larger in extreme fear.

### ATR-Based Exit Context:
Pre-computed ATR exit distances are INFORMATIONAL reference points:
- SL ~1.5x ATR: typical stop loss for current volatility
- TP ~2.0x ATR: R:R ≈ 1.33:1 target
- trail ~1x ATR: tight trailing stop width
- fee/ATR ≥ 1.0 → volatility too low relative to fees — skip trade
Use as calibration, not rules. H-TS learns which multiples work per regime.

### Exit Guidelines (safety layer enforces hard limits — see Open Positions section):
- Target profit and stop loss are YOUR judgment call. H-TS will learn what works.
- Safety layer provides backstop (hard SL, TP, time stop, trailing) — you should exit BEFORE safety triggers.
- Key principle: average loss < average win over time. H-TS tracks this.
- Max hold is {max_hold_min:.0f}min for scalps. Be proactive about exits as time increases.

### Lessons from Early Trades:
- Single-signal entries tend to underperform multi-signal confluence
- Mean reversion against a strong trend has lower win rate
- Tiny wins barely covering fees (~{fee_pct:.1f}% round-trip) teach nothing — skip marginal setups
- CVD divergence is a strong confirmation signal — price at new low but CVD rising = hidden buying

## YOU ARE AN AGENT, NOT JUST AN LLM
You have access to real-time analysis tools. Before making any BUY/SELL/ADD decision:
1. **Call `technical_analysis`** on relevant tickers/timeframes to get exact RSI, MACD, Bollinger, VWAP values
2. **Call `volume_analysis`** to check volume patterns and buy/sell pressure
3. **Call `get_derivatives`** for funding rates, OI changes, liquidation data
4. **Call `check_position`** before deciding to SELL or ADD to a position
5. **Call `get_ts_weights`** to see which signals are proven reliable

Precise TA values (RSI, MACD, EMA, Bollinger, StochRSI, ATR, VWAP) are PRE-COMPUTED in the snapshot for 5m/15m/1h.
Use tools ONLY for custom analysis (different periods, deeper volume profile, specific derivatives queries).
HOLD decisions can skip tools entirely.

## HOW YOU LEARN
You are a reinforcement learning agent. Hierarchical Thompson Sampling (H-TS) tracks which signals are reliable in each market regime at TWO levels:
- **Group level** (6 groups): Which CATEGORY of analysis works best? (technical trend, reversion, volume, derivatives, sentiment, macro)
- **Signal level** (28 signals): Which SPECIFIC indicator within a group is most reliable?

Every trade updates H-TS posteriors at both levels. Over time, H-TS data becomes your ground truth for what works.

**TRUST H-TS POSTERIORS ABOVE ALL.** The TS section in market data shows signal reliability. Higher % = proven reliable. Use them to weight your decisions.

## REGIME META-PARAMETERS (Level 0 H-TS)
H-TS also learns *how* to trade per regime via 6 meta-parameters (0.0 to 1.0):
- **position_scale**: How large to size positions. Low (<0.45) → size down. High (>0.55) → full size.
- **entry_selectivity**: How picky to be. Low → take more setups. High → only high-confluence.
- **hold_patience**: How long to hold. Low → quick exit. High → give trades time.
- **trade_frequency**: How often to trade. Low → sit out more. High → stay active.
- **trend_vs_reversion**: Strategy style. Low → mean-reversion. High → trend-following.
- **risk_aversion**: Risk appetite. Low → aggressive. High → conservative/defensive.

These are learned from your trade history in each regime. Use them as guidance — they reflect what has *actually worked*.
If meta-params say SIZE DOWN + BE PICKY + RISK-OFF, the regime has been punishing aggressive trading.

## ACTIONS: BUY, SELL, HOLD, ADD
- **BUY**: Open a new position (0-{max_pos:.0%} of portfolio)
- **SELL**: Close an entire position
- **ADD**: Add to an existing position (DCA / averaging down). Crisis = opportunity.
  - When a position is losing but the original thesis is intact, ADD can lower average entry price
  - Only ADD if: (1) the drop is technical, not fundamental (2) H-TS supports the signals (3) total exposure stays under {max_exposure:.0%}
  - ADD uses `position_pct` just like BUY (additional allocation)
  - **ADD converts position to DCA mode**: safety SL widens to {dca_sl_pct:+.1%}, hold extends to {dca_hold_min:.0f}min
  - This gives the averaging-down thesis time to play out. Don't ADD then expect scalp-tight exits.
- **HOLD**: No action

## PRINCIPLES (guidelines, not rigid rules — H-TS data should drive your evolution)
- **Speed over perfection.** Scalping rewards quick decisions. Act on clear setups, exit on clear invalidation.
- **Time is your enemy.** The longer you hold, the more exposed to reversals. Be proactive about exits.
- **Use tools to verify before acting.** Pre-computed TA is in the snapshot, but call tools for deeper analysis if needed.
- **Generate meaningful learning data.** Each trade should test a clear hypothesis. Trades that close near ±0% teach nothing.
- **Honest signal_weights are CRITICAL.** Rate which specific signals influenced your decision [-1.0, +1.0]. H-TS uses these to learn at both group and signal level.
- **Be SPECIFIC.** Don't just say "momentum". Say "ema_cross_fast" or "trend_strength".
- **Explore early, exploit later.** With few trades, TRY different signal combos and setups to generate learning data. As H-TS develops clear preferences, follow them more closely.
- **Account for fees.** Round-trip cost is ~{fee_pct:.1f}%. Gross profit must exceed this to be a real win.
- **Crisis = opportunity.** Sharp drops with high volume can be the best scalp entries.

## SIGNAL CATEGORIES (28 signals in 6 groups)
Rate ONLY the signals that actually influenced your decision. Use exact names below.

**technical_trend**: ema_cross_fast, ema_cross_slow, macd_histogram, trend_strength, supertrend
**technical_reversion**: rsi_signal, stoch_rsi, bb_squeeze, bb_deviation, vwap_deviation, support_resistance
**technical_volume**: volume_spike, cvd_signal, obv_divergence, volume_profile, mfi_signal
**derivatives**: funding_rate, oi_change, long_short_ratio, liquidation_level, basis_spread
**sentiment**: news_sentiment, fear_greed, social_buzz, whale_activity, exchange_flow
**macro**: market_regime, volatility_regime, btc_dominance, dxy_direction, etf_flow, stablecoin_flow

## WORKFLOW
1. Read the market snapshot — TA indicators are already computed for 5m/15m/1h
2. **FIRST: Check open positions.** Long-held positions approaching {max_hold_min:.0f}min? Consider selling proactively.
3. If pre-computed data is sufficient, proceed directly to decision
4. Call tools ONLY if you need: custom periods, volume profile, derivatives deep dive, position check
5. Output your final JSON decision — scalping pace (frequent checks)

## DECISION OUTPUT
After using tools, respond with valid JSON:
{{
  "decisions": [
    {{"ticker": "BTC/USDT:USDT", "action": "BUY"|"SELL"|"HOLD"|"ADD", "position_pct": 0.0-0.20, "confidence": 0.0-1.0, "reasoning": "...", "signal_weights": {{"ema_cross_fast": 0.8, "funding_rate": -0.3, "rsi_signal": 0.6}}}},
    ...
  ],
  "market_assessment": "one sentence overall market view",
  "regime_agreement": true|false,
  "learning_note": "what I learned from recent trade history",
  "next_check_seconds": 180,
  "wake_conditions": [{{"metric": "btc_price", "operator": "lt", "threshold": 95000, "reason": "support break"}}],
  "memory_update": "BTC testing 95K support, SOL showing RSI divergence on 5m"
}}"""


# Backward compatibility: default instance for tests/backtests
SYSTEM_PROMPT = build_system_prompt()


# ------------------------------------------------------------------
# Claude Agent
# ------------------------------------------------------------------

class ClaudeAgent:
    """Primary trading decision maker using Claude via SDK — TRUE AGENT with tools.

    NOT an advisor. NOT optional. This IS the trader.
    Falls back to rule-based pipeline ONLY if no auth is available.

    Agent mode: Claude can call 7 in-process MCP tools during decision-making:
    - technical_analysis: RSI, MACD, EMA, Bollinger, VWAP, ATR, StochRSI
    - get_ohlcv: Raw candle data for any timeframe
    - get_derivatives: Funding rates, OI, L/S ratio, liquidations
    - check_position: Position details, PnL, trailing stop state
    - search_trades: Past trade history with filters
    - get_ts_weights: H-TS posteriors for current regime
    - volume_analysis: Volume profile, buy/sell pressure, divergence

    Auth: ANTHROPIC_API_KEY (server) or OAuth (local dev).
        configure_sdk_authentication() → get_sdk_env_vars() → ClaudeAgentOptions(env=...)
    """

    def __init__(
        self,
        model: str = None,
        timeout: float = 180.0,  # increased for multi-turn tool use
        max_turns: int = 3,      # TA pre-computed → fewer tool calls needed
    ) -> None:
        self._model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
        self._model_fast = os.environ.get("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")
        self._timeout = timeout
        self._max_turns = max_turns
        self._available: Optional[bool] = None
        self._auth_configured: bool = False
        self._mcp_server = None   # lazy-initialized trading tools MCP server
        self._last_tool_calls: list = []  # [(name, input), ...] from last _call_sdk
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
                logger.info("[agent] Claude agent ready (%s, %s)", self._model, auth_mode)
        return self._available

    def reset_circuit_breaker(self) -> None:
        """Reset after a successful call."""
        self._consecutive_failures = 0
        self._circuit_open_time = 0.0

    def _get_mcp_server(self):
        """Lazy-init the trading tools MCP server."""
        if self._mcp_server is None:
            try:
                from core.agent_tools import create_trading_tools_server
                self._mcp_server = create_trading_tools_server()
                logger.info("[agent] MCP trading tools server initialized")
            except Exception as exc:
                logger.warning("[agent] Failed to create MCP tools server: %s", exc)
        return self._mcp_server

    async def _call_sdk(self, system_prompt: str, prompt: str, use_tools: bool = True, model_override: str = None) -> Optional[str]:
        """SDK call with optional MCP tool access (agent mode).

        When use_tools=True, Claude can call trading analysis tools
        during multi-turn reasoning before producing its final JSON output.

        Uses OpenClaw's auth pattern: env vars passed to ClaudeAgentOptions,
        NOT manually set on os.environ per-call.
        """
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ClaudeSDKClient,
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )
        from core.claude_auth import get_sdk_env_vars

        sdk_env = get_sdk_env_vars()
        # Allow nested sessions (when launched from within Claude Code)
        os.environ.pop("CLAUDECODE", None)
        response_text = ""
        tool_calls_made = 0
        tool_calls_log: list = []  # [(name, input_snippet), ...]

        # Build MCP config
        mcp_servers = {}
        allowed_tools = []
        max_turns = 1

        if use_tools:
            mcp_server = self._get_mcp_server()
            if mcp_server is not None:
                mcp_servers = {"trading": mcp_server}
                from core.agent_tools import TOOL_NAMES
                allowed_tools = list(TOOL_NAMES)
                max_turns = self._max_turns
                logger.debug("[agent] Agent mode: %d tools, max_turns=%d", len(allowed_tools), max_turns)

        async def _run():
            nonlocal response_text, tool_calls_made, tool_calls_log
            client = ClaudeSDKClient(options=ClaudeAgentOptions(
                model=model_override or self._model,
                system_prompt=system_prompt,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
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
                                elif isinstance(block, ToolUseBlock):
                                    tool_calls_made += 1
                                    tool_calls_log.append((block.name, str(block.input)[:200]))
                                    logger.info("[agent] Tool call #%d: %s(%s)",
                                                tool_calls_made, block.name,
                                                str(block.input)[:100])
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
        if tool_calls_made > 0:
            logger.info("[agent] Decision used %d tool calls", tool_calls_made)
        # Store last tool calls for logging by runner
        self._last_tool_calls = tool_calls_log
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

        # Build system prompt with dynamic config (no hardcoded thresholds)
        system_prompt = build_system_prompt(
            risk_config=snapshot.safety_config,
            round_trip_fee=snapshot.safety_config.get("round_trip_fee", 0.002),
        )

        # Hybrid model: Sonnet for critical, Haiku for routine
        has_positions = bool(snapshot.positions)
        is_wake = snapshot.trigger not in ("candle_close", "routine", "timer")
        use_sonnet = has_positions or is_wake
        model = self._model if use_sonnet else self._model_fast
        logger.debug("[agent] Model: %s (positions=%s, trigger=%s)",
                     model.split("-")[1], has_positions, snapshot.trigger)

        try:
            response_text = await self._call_sdk(system_prompt, prompt, model_override=model)
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

        is_dca = position_data.get("is_dca", False)
        dca_note = ("\n## DCA Mode Active\n"
                    "This position was ADD'd (DCA). Safety uses wider SL (-1.5%) and longer hold (2h).\n"
                    "If ADD thesis is failing, EXIT is better than waiting for the wider SL to hit.\n"
                    ) if is_dca else ""
        prompt = (
            "Evaluate this open position. Should I HOLD, EXIT, or ADD?\n\n"
            f"## Position\n{json.dumps(position_data, default=str)}\n\n"
            f"{dca_note}"
            f"## Market Context\n{json.dumps(market_context, default=str)}\n\n"
            f"## RL Context (Thompson Sampling)\n{json.dumps(ts_context, default=str)}\n\n"
            'Respond with JSON: {"verdict": "HOLD"|"EXIT"|"ADD", '
            '"confidence": 0.0-1.0, "reasoning": "..."}'
        )

        try:
            system = "You are a crypto position evaluator. Respond ONLY with JSON."
            response_text = await self._call_sdk(system, prompt, use_tools=False)
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
