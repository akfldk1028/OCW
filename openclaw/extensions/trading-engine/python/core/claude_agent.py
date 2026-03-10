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

Authentication: Claude Code CLI with Max subscription (unlimited).
    Previously used Agent SDK with API key, but switched to CLI to avoid per-call billing.
If CLI not available, falls back to rule-based pipeline (backtesting mode).
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

# ------------------------------------------------------------------
# Sparkline helper — ASCII mini-charts for LLM visual reasoning
# Reference: Agent Trading Arena (EMNLP 2025, arXiv:2502.17967)
# ------------------------------------------------------------------

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list, width: int = 20) -> str:
    """Convert a list of floats to a Unicode sparkline string.

    Uses 8 block characters (▁-█) to represent relative values.
    Returns empty string if insufficient data.
    """
    if not values or len(values) < 2:
        return ""
    # Use only the last `width` values
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return _SPARK_BLOCKS[3] * len(vals)  # flat line → mid-height
    out = []
    for v in vals:
        idx = int((v - lo) / span * 7)
        idx = max(0, min(7, idx))
        out.append(_SPARK_BLOCKS[idx])
    return "".join(out)


def _rsi_label(values: list) -> str:
    """Generate a short directional label from RSI trajectory."""
    if not values or len(values) < 2:
        return ""
    last = values[-1]
    first = values[0]
    if last < 30:
        zone = "oversold"
    elif last > 70:
        zone = "overbought"
    else:
        zone = "neutral"
    if last > first + 5:
        return f"{zone}, rising"
    elif last < first - 5:
        return f"{zone}, falling"
    return zone

# CLI availability check
_CLI_AVAILABLE: Optional[bool] = None
_CLI_PATH = "/Users/bijeuaien-epeu/.local/bin/claude"


def _check_cli() -> bool:
    """Check if Claude Code CLI is available."""
    global _CLI_AVAILABLE
    if _CLI_AVAILABLE is None:
        import shutil
        if os.path.isfile(_CLI_PATH) and os.access(_CLI_PATH, os.X_OK):
            _CLI_AVAILABLE = True
        elif shutil.which("claude"):
            _CLI_AVAILABLE = True
        else:
            _CLI_AVAILABLE = False
            logger.warning("[agent] claude CLI not found — falling back to rule-based")
    return _CLI_AVAILABLE


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
    btc_dominance_pct: float = 0.0  # BTC market cap dominance %
    social_buzz_score: float = 0.0  # trending coins avg |price_change| (excitement level)
    news_sentiment_score: float = 0.0  # headline sentiment [-1, +1]
    whale_activity_score: float = 0.0  # large BTC transaction activity [0, 1]
    exchange_netflow_score: float = 0.0  # exchange netflow [-1, +1], negative=outflow=bullish

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

    # Liquidation heatmap context per ticker (from LiquidationTracker)
    liquidation_contexts: Dict[str, Dict] = field(default_factory=dict)

    # OOD scores per ticker (Mahalanobis distance, normal=2-4, OOD>6)
    ood_scores: Dict[str, float] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Serialize into a Claude-readable prompt section."""
        lines = []

        # Price action + sparklines (Agent Trading Arena, EMNLP 2025)
        lines.append("## Current Prices & Returns")
        import math
        for tic in self.candidates:
            px = self.ticker_prices.get(tic, 0)
            r4h = self.ticker_returns_4h.get(tic, 0)
            r24h = self.ticker_returns_24h.get(tic, 0)
            vol = self.ticker_volumes.get(tic, 0)
            # Guard against NaN/Inf corrupting prompt (would confuse Claude)
            if not math.isfinite(px) or px <= 0:
                lines.append(f"  {tic}: PRICE UNAVAILABLE")
                continue
            if not math.isfinite(r4h):
                r4h = 0
            if not math.isfinite(r24h):
                r24h = 0
            lines.append(f"  {tic}: ${px:,.2f} (4h: {r4h:+.2%}, 24h: {r24h:+.2%}, vol: ${vol:,.0f})")
            # ASCII sparklines from 5m pre-computed data
            ta_5m = self.pre_computed_ta.get(tic, {}).get("5m", {})
            closes_spark = _sparkline(ta_5m.get("recent_closes", []))
            if closes_spark:
                lines.append(f"    5m price: {closes_spark}  (last 20 candles)")
            rsi_hist = ta_5m.get("rsi_history", [])
            rsi_spark = _sparkline(rsi_hist)
            if rsi_spark:
                lines.append(f"    RSI(14):  {rsi_spark}  ({_rsi_label(rsi_hist)})")
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
                # 1h trend direction summary (most important signal)
                ind_1h = ta_data.get("1h", {})
                ema_1h = ind_1h.get("ema_cross", {}).get("status", "?")
                st_1h = ind_1h.get("supertrend_dir", 0)
                st_1h_label = "bull" if st_1h > 0 else ("bear" if st_1h < 0 else "?")
                if ema_1h == "bearish" and st_1h < 0:
                    trend_tag = "⬇ DOWNTREND — prefer SHORT/HOLD"
                elif ema_1h == "bullish" and st_1h > 0:
                    trend_tag = "⬆ UPTREND — prefer BUY"
                else:
                    trend_tag = "↔ MIXED/RANGING"
                lines.append(f"  {tic}: [1h trend: {trend_tag}]")
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
                    # Core indicators
                    ema_50 = ind.get("ema_50")
                    st_dir = ind.get("supertrend_dir", 0)
                    st_label = "bull" if st_dir > 0 else ("bear" if st_dir < 0 else "?")
                    mfi = ind.get("mfi", "N/A")
                    obv_dir = ind.get("obv_direction", 0)
                    obv_label = "bull_div" if obv_dir > 0 else ("bear_div" if obv_dir < 0 else "aligned")
                    bb_bw = bb.get("bandwidth", "N/A")
                    lines.append(
                        f"    {interval}: RSI7={rsi7} RSI14={rsi14} div={rsi_div} StochRSI={stoch} "
                        f"MACD_hist={macd_h} EMA9/21={ema.get('status', '?')}({ema.get('gap_pct', 0):+.4f}) "
                        f"EMA50={ema_50 or 'N/A'} Supertrend={st_label} "
                        f"BB_%B={bb.get('pct_b', 'N/A')} BB_bw={bb_bw} MFI={mfi} OBV={obv_label} "
                        f"ATR%={atr_pct} VWAP_dev={vwap_dev}"
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
        has_liq_context = any(
            abs(ctx.get("signal", 0)) > 0.2
            for ctx in self.liquidation_contexts.values()
        )
        if self.liquidation_usd_1h > 0 or self.stablecoin_mcap_b > 0 or has_liq_context:
            lines.append("\n## Market Structure")
            if self.liquidation_usd_1h > 0:
                lines.append(f"  Liquidations (1h): ${self.liquidation_usd_1h/1e6:.1f}M")
                if self.liquidation_cascade:
                    lines.append("  !! LIQUIDATION CASCADE — forced exits accelerating, expect volatility spike")
            # Liquidation Heatmap (per-ticker, from LiquidationTracker)
            if self.liquidation_contexts:
                for tic in self.candidates:
                    ctx = self.liquidation_contexts.get(tic)
                    if not ctx:
                        continue
                    signal = ctx.get("signal", 0)
                    if abs(signal) <= 0.2:
                        continue  # skip insignificant signals
                    long_usd = ctx.get("long_liqs_1h_usd", 0)
                    short_usd = ctx.get("short_liqs_1h_usd", 0)
                    above = ctx.get("nearest_cluster_above")
                    below = ctx.get("nearest_cluster_below")
                    direction = "shorts above" if signal > 0 else "longs below"
                    lines.append(f"  {tic} Liquidation Map:")
                    lines.append(f"    Signal: {signal:+.2f} ({direction} dominant)")
                    lines.append(f"    1h Liqs: Long ${long_usd/1e6:.1f}M | Short ${short_usd/1e6:.1f}M")
                    if above:
                        lines.append(f"    Short cluster above: ${above[0]:,.0f} (~${above[1]/1e6:.0f}M)")
                    if below:
                        lines.append(f"    Long cluster below: ${below[0]:,.0f} (~${below[1]/1e6:.0f}M)")
                    if ctx.get("cascade_risk"):
                        lines.append("    !! CASCADE RISK — rapid liquidations detected")
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
        if self.btc_dominance_pct > 0:
            lines.append(f"  BTC Dominance: {self.btc_dominance_pct:.1f}%")
        if self.etf_daily_flow_usd != 0:
            lines.append(f"  BTC ETF Flow: ${self.etf_daily_flow_usd/1e6:+.1f}M ({self.etf_flow_label})")
        if self.social_buzz_score > 0:
            lines.append(f"  Social Buzz: {self.social_buzz_score:.1f} (trending coins avg |change|)")
        if abs(self.news_sentiment_score) > 0.05:
            sent_label = "bullish" if self.news_sentiment_score > 0 else "bearish"
            lines.append(f"  News Sentiment: {self.news_sentiment_score:+.2f} ({sent_label})")
        if self.whale_activity_score > 0.1:
            lines.append(f"  Whale Activity: {self.whale_activity_score:.2f} (large BTC txs)")
        if abs(self.exchange_netflow_score) > 0.1:
            flow_label = "outflow=bullish" if self.exchange_netflow_score > 0 else "inflow=bearish"
            lines.append(f"  Exchange Netflow: {self.exchange_netflow_score:+.2f} ({flow_label})")

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

        # Signal-level weights (Level 2) — actionable ranking
        if self.ts_mean_weights:
            sorted_sigs = sorted(self.ts_mean_weights.items(), key=lambda x: -x[1])
            # Split into tiers for clear decision guidance
            proven = [(s, w) for s, w in sorted_sigs if w > 0.55]
            neutral = [(s, w) for s, w in sorted_sigs if 0.45 <= w <= 0.55]
            unreliable = [(s, w) for s, w in sorted_sigs if w < 0.45]

            if proven:
                lines.append(f"\n  ★ PROVEN SIGNALS — base your decisions primarily on these:")
                for sig, sw in proven:
                    lines.append(f"    {sig}: {sw*100:.1f}%")
            if neutral:
                lines.append(f"\n  ~ NEUTRAL SIGNALS — use as confirmation only, not as primary trigger:")
                for sig, sw in neutral[:8]:
                    lines.append(f"    {sig}: {sw*100:.1f}%")
            if unreliable:
                lines.append(f"\n  ✗ UNRELIABLE SIGNALS — these have LOST money. Ignore or fade them:")
                for sig, sw in unreliable:
                    lines.append(f"    {sig}: {sw*100:.1f}%")

        if self.ts_regime_info:
            lines.append(f"  {self.ts_regime_info}")

        # Level 0: Meta-parameters — "how to trade" in this regime
        if self.ts_meta_params:
            lines.append(f"\n  REGIME META-PARAMETERS (H-TS learned — how to trade in this regime):")
            _meta_labels = {
                "position_scale":       ("SIZE DOWN",     "SIZE UP"),
                "entry_selectivity":    ("BROAD ENTRY",   "BE PICKY"),
                "hold_patience":        ("TIGHT TARGETS", "HOLD LONGER"),
                "trade_frequency":      ("REDUCE LONGS",  "BE ACTIVE"),
                "trend_vs_reversion":   ("MEAN-REVERT",   "TREND-FOLLOW"),
                "risk_aversion":        ("AGGRESSIVE",    "RISK-OFF"),
                "profit_target_width":  ("TIGHT TP",      "WIDE TP"),
                "loss_tolerance":       ("CUT FAST",      "GIVE ROOM"),
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
                side = pos.get("side", "long")
                entry_px = pos.get("entry_price", 0)
                mfe = pos.get("mfe", 0)
                mae = pos.get("mae", 0)
                capture = pos.get("capture_ratio", 0)
                side_tag = " [SHORT]" if side == "short" else ""
                dca_tag = " [DCA]" if pos.get("is_dca") else ""
                exit_action = "COVER" if side == "short" else "SELL"
                lines.append(f"  {tic}{side_tag}{dca_tag}: qty={pos.get('qty',0):.4f}, "
                                 f"entry=${entry_px:,.2f}, "
                                 f"PnL={pos.get('pnl_pct',0):+.2%}, "
                                 f"MFE={mfe:+.2%}, MAE={mae:+.2%}, capture={capture*100:.0f}%, "
                                 f"held={pos.get('held_hours',0):.0f}h "
                                 f"(exit={exit_action})")
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
            lines.append(f"  Safety is a backstop — aim to exit on your own judgment before it triggers.")
            lines.append(f"  MFE/MAE guide: MFE=best unrealized profit, MAE=worst drawdown, capture=PnL/MFE.")
            lines.append(f"  → MFE>0 + PnL dropping: normal pullback if small. Only exit if thesis INVALIDATED.")
            lines.append(f"  → MFE=0% after 10+min: thesis never worked. Cut immediately.")
            _fee_pct = self.safety_config.get("round_trip_fee", 0.0004) * 100
            lines.append(f"  → Do NOT exit just because capture% is falling. Target profit > {_fee_pct:.1f}% (fees).")
        else:
            lines.append("  No open positions — you have ZERO positions. Do NOT output SELL or COVER.")
            lines.append("  Only valid actions now: BUY, SHORT, or HOLD.")

        # Portfolio
        lines.append(f"\n## Portfolio")
        lines.append(f"  Value: ${self.portfolio_value:,.2f}, Cash: ${self.cash:,.2f}")
        lines.append(f"  Trigger: {self.trigger}")

        # Sentiment
        if self.fear_greed_index > 0 or self.etf_daily_flow_usd != 0:
            lines.append(f"\n## Market Sentiment")
            if self.fear_greed_index > 0:
                lines.append(f"  Fear & Greed Index: {self.fear_greed_index}/100 ({self.fear_greed_label})")
                lines.append(f"  (Check H-TS signal reliability for sentiment_index to see current effectiveness)")
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

        # OOD Warning (Mahalanobis distance, arXiv:2512.23773)
        ood_tickers = {t: d for t, d in self.ood_scores.items() if d > 6.0}
        if ood_tickers:
            lines.append(f"\n## ⚠ OOD Warning (Out-of-Distribution Market State)")
            lines.append("The following tickers show STATISTICALLY UNUSUAL market conditions (Mahalanobis distance > 6):")
            for tic, dist in sorted(ood_tickers.items(), key=lambda x: -x[1]):
                lines.append(f"  {tic}: distance={dist:.1f} — H-TS data may not apply well to this regime")
            lines.append("H-TS posteriors were learned from normal conditions. Weigh this information in your decision.")

        return "\n".join(lines)


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

def build_system_prompt(risk_config: dict = None, round_trip_fee: float = 0.0004) -> str:
    """Build system prompt with dynamic config values. No hardcoded strategy thresholds."""
    rc = risk_config or {}
    sl_pct = rc.get("stop_loss_pct", -0.06)
    tp_pct = rc.get("take_profit_pct", 0.08)
    max_hold_hrs = rc.get("max_hold_hours", 24.0)
    dca_sl_pct = rc.get("dca_stop_loss_pct", -0.06)
    dca_hold_hrs = rc.get("dca_max_hold_hours", 24.0)
    max_exposure = rc.get("max_exposure_pct", 0.60)
    max_pos = rc.get("max_position_pct", 0.15)
    fee_pct = round_trip_fee * 100  # e.g. 0.2%

    return f"""You are an autonomous crypto TRADING AGENT on Binance Futures.

**Dual-Timeframe Strategy: 1h candle for DIRECTION decisions, 5m/15m data for ENTRY TIMING.**
You analyze the 1h chart to decide direction (BUY/SHORT/HOLD), then use 5m/15m TA data to refine entry precision.

**Your performance is measured by net P&L. Every loss reduces future trading capital. Missing a trade costs nothing; losing on a trade costs real capital.**

## SCALPING STRATEGY — CONCEPTUAL SETUPS (H-TS data overrides everything)
These are conceptual frameworks only. **Do NOT treat any threshold as a hard rule.**
H-TS posteriors tell you which signals actually work in the current regime — trust the data above all.

### CRITICAL — 1h DIRECTION FILTER (check BEFORE any entry):
**Look at the 1h EMA9/21 and Supertrend for each ticker FIRST.**
- 1h EMA=bearish + Supertrend=bear → **DOWNTREND. SHORT or HOLD only. Do NOT buy.**
- 1h EMA=bullish + Supertrend=bull → **UPTREND. BUY or HOLD only. Do NOT short.**
- Mixed/neutral → Range-bound. Mean reversion setups valid.
**"Oversold in a downtrend" is NOT a buy signal — it's a trend confirmation. RSI can stay at 20 for days.**

### Setup A: Mean Reversion Scalp (ONLY in ranging/neutral regimes)
**PREREQUISITE: 1h trend must be neutral or mixed.** If 1h EMA AND Supertrend both point one direction, this setup has very low win rate.
**Concept:** Extreme oversold + institutional mean reversion.
**Key signals (more confluence = higher confidence):**
1. StochRSI at extreme oversold on 5m
2. Price at or below lower Bollinger Band on 5m
3. Price below VWAP (institutional mean)
**Strengtheners:** RSI divergence, CVD divergence, volume spike
**AVOID when:** 1h EMA bearish AND Supertrend bear — use Setup E (short) instead

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
**Caution:** Look for stabilization after flush before entry

### Setup D: Short Scalp — Overbought Reversal
**Concept:** Momentum exhaustion at extended levels.
**Key signals:** RSI divergence, declining volume, StochRSI turning down, bearish EMA cross forming
**Strengtheners:** Positive funding rate (longs paying shorts), MACD divergence
**Exit:** COVER when reversal thesis invalidated (new highs with volume) or support bounce

### Setup E: Momentum Short — Downtrend Continuation
**Concept:** Trend continuation on pullbacks in a downtrend.
**Key signals:** EMA alignment bearish on higher TF, price below VWAP, pullback to resistance
**Strengtheners:** Increasing OI + negative CVD (net selling), fear sentiment
**Exit:** COVER at next support level or trailing stop locks profit

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

### Exit Guidelines — R:R is EVERYTHING:
**Your avg win MUST exceed your avg loss.** Exiting at +0.05% while losing -0.20% means you need 80% WR just to break even.
- **Minimum profit target: 2x ATR(5m) or +0.15%, whichever is larger.** Exits below this are fee losses.
- **Do NOT exit a winner just because MFE is declining.** Small MFE pullbacks (< 0.1%) are noise. Only exit when your exit THESIS is invalidated.
- **MFE/MAE in Open Positions are INFORMATIONAL, not exit triggers.** Low capture on a small move means "hold longer", not "take what you have".
- **Let winners run. Cut losers.** If MFE=+0.00% after 10+ min → thesis failed, exit. If MFE > +0.10% → the direction was right, hold for target.
- Target profit and stop loss are YOUR judgment call. H-TS will learn what works.
- Safety layer provides backstop (hard SL, TP, time stop, trailing) — see Open Positions section.
- Max hold is {max_hold_hrs:.0f}h (safety backstop).

## DATA
All technical data (RSI, MACD, EMA, Bollinger, StochRSI, ATR, VWAP) is PRE-COMPUTED in the snapshot for 5m/15m/1h.
**1h data = direction analysis. 5m/15m data = entry timing & precision.** Base your decisions on the provided data.

## HOW YOU LEARN
You are a reinforcement learning agent. Hierarchical Thompson Sampling (H-TS) tracks which signals are reliable in each market regime at TWO levels:
- **Group level** (6 groups): Which CATEGORY of analysis works best? (technical trend, reversion, volume, derivatives, sentiment, macro)
- **Signal level** (28 signals): Which SPECIFIC indicator within a group is most reliable?

Every trade updates H-TS posteriors at both levels. Over time, H-TS data becomes your ground truth for what works.

**TRUST H-TS POSTERIORS ABOVE ALL.** The TS section shows signal reliability from real trade data:
- **★ PROVEN (>55%)**: These signals have made money historically.
- **~ NEUTRAL (45-55%)**: Insufficient data to judge — could improve with more trades.
- **✗ UNRELIABLE (<45%)**: These have lost money historically.
Use these as information, not constraints. H-TS learns over time — today's UNRELIABLE signal may become PROVEN in a different regime.

## REGIME META-PARAMETERS (Level 0 H-TS)
H-TS also learns *how* to trade per regime via 6 meta-parameters (0.0 to 1.0):
- **position_scale**: How large to size positions. Low (<0.45) → size down. High (>0.55) → full size.
- **entry_selectivity**: How picky to be. Low → take more setups. High → only high-confluence.
- **hold_patience**: Profit target sizing. Low → tighter targets and stops. High → give trades room to develop.
- **trade_frequency**: How often to trade. Low → be selective. High → stay active.
- **trend_vs_reversion**: Strategy style. Low → mean-reversion. High → trend-following.
- **risk_aversion**: Risk appetite. Low → aggressive. High → conservative/defensive.
- **profit_target_width**: Exit timing for winners. Low → take profit quickly (tight TP). High → let winners run (wide TP).
- **loss_tolerance**: Cut-loss speed. Low → cut losses fast. High → give losing trades room to recover.

These are learned from your trade history in each regime. Use them as guidance — they reflect what has *actually worked*.
If meta-params say SIZE DOWN + BE PICKY + RISK-OFF, the regime has been punishing aggressive trading.

## ACTIONS: BUY, SELL, SHORT, COVER, ADD, HOLD
- **BUY**: Open a new LONG position (0-{max_pos:.0%} of portfolio) — profit when price rises
- **SELL**: Close a LONG position
- **SHORT**: Open a new SHORT position (0-{max_pos:.0%} of portfolio) — profit when price drops
  - Use when bearish signals align: downtrend, overbought reversal, funding rate extreme positive, liquidation cascade building
  - Same safety rules apply (SL/TP/trailing) but direction-inverted
- **COVER**: Close a SHORT position
- **ADD**: Add to an existing position (DCA / averaging down). Crisis = opportunity.
  - When a position is losing but the original thesis is intact, ADD can lower average entry price
  - Only ADD if: (1) the drop is technical, not fundamental (2) H-TS supports the signals (3) total exposure stays under {max_exposure:.0%}
  - ADD uses `position_pct` just like BUY (additional allocation)
  - **ADD converts position to DCA mode**: safety SL widens to {dca_sl_pct:+.1%}, hold extends to {dca_hold_hrs:.0f}h
  - This gives the averaging-down thesis time to play out. Don't ADD then expect scalp-tight exits.
- **HOLD**: No action

## PRINCIPLES (guidelines, not rigid rules — H-TS data should drive your evolution)
- **R:R > 1.0 is non-negotiable.** Each trade should risk X to make at least X. Exiting at +0.03% profit is WORSE than holding — it's a fee loss that counts as a "win" but loses money.
- **Let winners breathe.** Small pullbacks from MFE are normal. Only exit when your thesis is clearly invalidated (e.g., key level broken, momentum reversal on higher TF).
- **Cut losers decisively.** If MFE stays near zero after 10+ min, the thesis was wrong — exit without waiting.
- **Quality over quantity.** Skip marginal setups — they just generate fee losses. HOLD is free.
- **Honest signal_weights matter.** Rate which specific signals influenced your decision [-1.0, +1.0]. H-TS uses these to learn at both group and signal level.
- **Be SPECIFIC.** Don't just say "momentum". Say "ema_cross_fast" or "trend_strength".
- **Explore early, exploit later.** With few trades, TRY different signal combos and setups to generate learning data. As H-TS develops clear preferences, follow them more closely.
- **Account for fees.** Round-trip cost is ~{fee_pct:.1f}%. Gross profit must exceed this to be a real win. A +0.05% exit is a NET LOSS after fees.
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
1. Read the market snapshot — all TA indicators are pre-computed for 5m/15m/1h
2. **FIRST: Check "Open Positions" section.** If it says "No open positions", you MUST NOT output SELL or COVER actions — only BUY, SHORT, or HOLD are valid. SELL/COVER without a position will be rejected.
3. If positions exist and are long-held (approaching {max_hold_hrs:.0f}h), consider exiting proactively.
4. Analyze confluence across timeframes and signal groups
5. Output your final JSON decision

## DECISION OUTPUT
Respond with valid JSON:
{{
  "decisions": [
    {{"ticker": "BTC/USDT:USDT", "action": "BUY"|"SELL"|"SHORT"|"COVER"|"HOLD"|"ADD", "position_pct": 0.0-0.20, "confidence": 0.0-1.0, "reasoning": "...", "signal_weights": {{"ema_cross_fast": 0.8, "funding_rate": -0.3, "rsi_signal": 0.6}}}},
    ...
  ],
  "market_assessment": "one sentence overall market view",
  "regime_agreement": true|false,
  "learning_note": "what I learned from recent trade history",
  "next_check_seconds": 180,
  "wake_conditions": [{{"metric": "btc_price", "operator": "lt", "threshold": 95000, "reason": "support break"}}],
  "memory_update": "BTC testing 95K support, SOL showing RSI divergence on 5m"
}}

All market data is pre-computed in the snapshot below. Make your decision based on the provided data."""


# Backward compatibility: default instance for tests/backtests
SYSTEM_PROMPT = build_system_prompt()


# ------------------------------------------------------------------
# Claude Agent
# ------------------------------------------------------------------

class ClaudeAgent:
    """Primary trading decision maker using Claude Code CLI.

    NOT an advisor. NOT optional. This IS the trader.
    All market data is pre-computed and passed via prompt snapshot.

    Auth: Claude Code CLI with Max subscription (unlimited, no per-call billing).
    """

    def __init__(
        self,
        model: str = None,
        timeout: float = 300.0,
    ) -> None:
        self._model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
        self._model_fast = os.environ.get("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")
        self._timeout = timeout
        self._cli_path = _CLI_PATH
        self._available: Optional[bool] = None
        self._consecutive_failures: int = 0
        self._max_failures: int = 3  # circuit breaker
        self._circuit_open_time: float = 0.0
        self._circuit_cooldown: float = 900.0  # 15 min half-open retry

    @property
    def is_available(self) -> bool:
        """Check if Claude CLI is available + circuit breaker.

        Circuit breaker uses half-open pattern:
        - 3 consecutive failures → open (block all calls)
        - After 15 min cooldown → half-open (allow one retry)
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
            logger.info("[agent] Circuit breaker HALF-OPEN after %.0fs, retrying", elapsed)
            self._consecutive_failures = 0
            self._circuit_open_time = 0.0
            self._available = None
            # Reset global cache so _check_cli() actually re-checks
            global _CLI_AVAILABLE
            _CLI_AVAILABLE = None

        if self._available is None:
            if not _check_cli():
                self._available = False
            else:
                self._available = True
                logger.info("[agent] Claude agent ready (CLI, %s)", self._model)
        return self._available

    def reset_circuit_breaker(self) -> None:
        """Reset after a successful call."""
        self._consecutive_failures = 0
        self._circuit_open_time = 0.0

    async def _call_cli(self, system_prompt: str, prompt: str,
                        model_override: str = None) -> Optional[str]:
        """Call Claude Code CLI in --print mode (Max subscription, no per-call billing).

        Uses subprocess to invoke `claude --print` with system prompt and user prompt.
        Environment is sanitized to prevent API key usage and nesting guard conflicts.
        """
        model = model_override or self._model

        cmd = [
            self._cli_path,
            "--print",
            "--output-format", "text",
            "--model", model,
            "--no-session-persistence",
            "--system-prompt", system_prompt,
        ]

        # Env: remove API key (use Max subscription), remove CLAUDECODE (nesting guard)
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "CLAUDECODE",
                            "ANTHROPIC_AUTH_TOKEN")}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[agent] CLI timeout after %.0fs", self._timeout)
            try:
                proc.kill()
            except Exception:
                pass
            raise

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            logger.warning("[agent] CLI error (rc=%d): %s", proc.returncode, err)
            return None

        response_text = stdout.decode("utf-8").strip()
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
            round_trip_fee=snapshot.safety_config.get("round_trip_fee", 0.0004),
        )

        # Always use Sonnet — Haiku too unreliable for price reading / decisions
        model = self._model
        prompt_chars = len(system_prompt) + len(prompt)
        logger.info("[agent] Model: %s (trigger=%s, prompt=%dk chars)", model, snapshot.trigger, prompt_chars // 1000)

        try:
            response_text = await self._call_cli(system_prompt, prompt, model_override=model)
            if not response_text:
                self._consecutive_failures += 1
                return None

            result = self._parse_json(response_text)
            if result and "decisions" in result:
                # Validate decision fields — reject NaN/Inf/negative values
                result["decisions"] = self._validate_decisions(result["decisions"])
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
            response_text = await self._call_cli(system, prompt)
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

    @staticmethod
    def _validate_decisions(decisions: list) -> list:
        """Sanitize Claude's decision values — reject NaN/Inf/negative."""
        import math
        valid = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            action = d.get("action", "HOLD").upper()
            if action == "HOLD":
                valid.append(d)
                continue
            # Validate position_pct
            try:
                pct = float(d.get("position_pct", 0.10))
                if not math.isfinite(pct) or pct < 0:
                    pct = 0.10
                d["position_pct"] = pct
            except (TypeError, ValueError):
                d["position_pct"] = 0.10
            # Validate confidence
            try:
                conf = float(d.get("confidence", 0.5))
                if not math.isfinite(conf):
                    conf = 0.5
                d["confidence"] = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                d["confidence"] = 0.5
            valid.append(d)
        return valid
