"""Constants for Hierarchical Thompson Sampling."""

from __future__ import annotations

from typing import Dict, List

# Default regime when none detected
_GLOBAL_REGIME = "_global"

# Minimum trades per regime before trusting regime-specific weights
# Thompson (1933), CADTS (arXiv:2410.04217): 30-50 trades for meaningful adaptation.
# 10 = practical minimum for crypto scalping (fast regime cycles).
_MIN_REGIME_TRADES = 10

# 6 groups × 5-6 signals each = 32 total signals
SIGNAL_GROUPS: Dict[str, List[str]] = {
    "technical_trend": [
        "ema_cross_fast",       # EMA(9) vs EMA(21) crossover
        "ema_cross_slow",       # EMA(21) vs EMA(50) crossover
        "macd_histogram",       # MACD line vs signal divergence
        "trend_strength",       # ADX > 25 trending detection
        "supertrend",           # ATR-based trend direction
    ],
    "technical_reversion": [
        "rsi_signal",           # RSI(14) overbought/oversold
        "stoch_rsi",            # Stochastic RSI crossovers
        "bb_squeeze",           # Bollinger Band width contraction
        "bb_deviation",         # Price distance from BB middle
        "vwap_deviation",       # Price vs session VWAP
        "support_resistance",   # Key S/R level proximity
    ],
    "technical_volume": [
        "volume_spike",         # Volume > 2x 20-period average
        "cvd_signal",           # Cumulative Volume Delta (buy vs sell pressure)
        "obv_divergence",       # OBV vs price divergence
        "volume_profile",       # Volume at price (POC proximity)
        "mfi_signal",           # Money Flow Index extremes
    ],
    "derivatives": [
        "funding_rate",         # Binance perp funding rate
        "oi_change",            # Open interest change rate
        "long_short_ratio",     # Top trader long/short ratio
        "liquidation_level",    # Nearby liquidation cluster density
        "basis_spread",         # Futures premium vs spot (annualized)
    ],
    "sentiment": [
        "news_sentiment",       # FinBERT / headline NLP score
        "fear_greed",           # Crypto Fear & Greed Index
        "social_buzz",          # Social media volume/sentiment
        "whale_activity",       # Large transfer alerts (>$10M)
        "exchange_flow",        # Net exchange inflow/outflow
    ],
    "macro": [
        "market_regime",        # HMM/rule-based regime label
        "volatility_regime",    # ATR/BB-based vol state
        "btc_dominance",        # BTC.D percentage trend
        "dxy_direction",        # Dollar index direction
        "etf_flow",             # BTC ETF daily net flow
        "stablecoin_flow",      # Stablecoin supply change
    ],
}

# Flat list of all signal names (for iteration)
ALL_SIGNALS: List[str] = []
for _sigs in SIGNAL_GROUPS.values():
    ALL_SIGNALS.extend(_sigs)

# Reverse lookup: signal_name → group_name
SIGNAL_TO_GROUP: Dict[str, str] = {}
for _g, _sigs in SIGNAL_GROUPS.items():
    for _s in _sigs:
        SIGNAL_TO_GROUP[_s] = _g

GROUP_NAMES: List[str] = list(SIGNAL_GROUPS.keys())

# Level 0: Meta-parameters — "how to trade" per regime
# Each is a Beta(2,2) prior, mean=0.5. Interpretation depends on param:
#   position_scale:      low mean → size down, high mean → size up
#   entry_selectivity:   low mean → broad entry, high mean → picky
#   hold_patience:       low mean → quick exit, high mean → hold longer
#   trade_frequency:     low mean → sit out, high mean → active
#   trend_vs_reversion:  low mean → mean-revert, high mean → trend-follow
#   risk_aversion:       low mean → aggressive, high mean → conservative
META_PARAMS: List[str] = [
    "position_scale",
    "entry_selectivity",
    "hold_patience",
    "trade_frequency",
    "trend_vs_reversion",
    "risk_aversion",
    "profit_target_width",   # low=tight TP (take profit fast), high=wide TP (let winners run)
    "loss_tolerance",        # low=cut fast, high=give room
]
