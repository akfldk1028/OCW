"""Evidence-based backtest v2 -- LATEST research (2024-2025).

Previous approach (backtest_evidence.py) failed:
    - ETF Rotation: +16.61% vs SPY +54.02% (Alpha -37%)
    - Crypto Momentum: +125.59% vs BTC +372.91% (Alpha -247%)

LATEST RESEARCH FINDINGS:

    1. Momentum + Mean Reversion COMBINED (2025 systematic crypto study)
       - Sharpe 1.71, Annual 56% when 50/50 blended
       - Momentum alone: Sharpe ~1.0 (strong in trends)
       - Mean Reversion alone: Sharpe ~2.3 (strong post-2021)
       - KEY: Combine BOTH, not pick one

    2. Regime Switching (Price Action Lab, Jan 2024)
       - SPY/QQQ/TLT/GLD, Sharpe 1.15-1.21
       - Mean reversion in bear (short-squeeze rallies)
       - Momentum in bull (trend following)
       - Monthly rebalance, 53.6% win rate

    3. RSI Momentum works on CRYPTO but NOT stocks
       - RSI as momentum (not reversal) in crypto
       - Best results: short lookback periods
       - Crypto has stronger momentum persistence

    4. Talyxion Framework (arXiv 2511.13239, Nov 2025)
       - 30-day live Binance Futures: Sharpe 5.72, MDD 4.56%
       - Universe selection -> Alpha test -> Vol-aware optimization -> DD risk mgmt
       - Win rate 57.71%

    5. Dynamic Grid Trading (arXiv 2506.11921, Jun 2025)
       - BTC/ETH: IRR 60-70%, MDD ~50% (vs 80% B&H)
       - Grid resets when price breaks bounds
       - Works because crypto is range-bound 70% of time

    6. Slow Momentum + Fast Reversion (Oxford, JFDS 2022)
       - Changepoint detection to switch between momentum and reversion
       - Outperforms pure TSMOM especially at turning points

STRATEGY DESIGN (based on ALL above):

    A. CRYPTO REGIME BLEND (main strategy)
       - Regime detection: 20d volatility + 50d SMA trend
       - TRENDING regime: RSI momentum (buy RSI > 50 + positive 14d momentum)
       - RANGING regime: Bollinger Band mean reversion (buy lower band, sell upper)
       - Position sizing: Kelly fraction capped at 25%
       - Risk: Max 80% invested, trailing stop at -12%

    B. EQUITY REGIME SWITCH
       - QQQ + TLT (tech stocks vs bonds)
       - 12-month momentum: QQQ > 0 -> hold QQQ, else TLT
       - Add RSI oversold filter for entry timing
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ===================================================================
# Helpers
# ===================================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI series (not just last value)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.where(loss > 0, 1e-10)
    return 100 - 100 / (1 + rs)


def calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) bands."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + num_std * std, sma - num_std * std


def detect_regime(close: pd.Series, date, vol_window: int = 20, trend_window: int = 50) -> str:
    """Detect market regime: 'trending' or 'ranging'.

    Uses realized volatility percentile + trend strength.
    - High directional move + moderate vol = TRENDING
    - Low directional move + high vol = RANGING
    """
    hist = close.loc[:date].dropna()
    if len(hist) < max(vol_window, trend_window) + 5:
        return "unknown"

    # Realized vol (annualized)
    returns = hist.pct_change().dropna()
    recent_vol = returns.iloc[-vol_window:].std() * (365 ** 0.5)

    # Trend strength: price vs SMA
    sma = hist.iloc[-trend_window:].mean()
    current = hist.iloc[-1]
    trend_strength = abs(current - sma) / sma

    # ADX-like: ratio of directional move to total movement
    total_range = hist.iloc[-vol_window:].max() - hist.iloc[-vol_window:].min()
    net_move = abs(hist.iloc[-1] - hist.iloc[-vol_window])
    efficiency = net_move / total_range if total_range > 0 else 0

    if efficiency > 0.5 and trend_strength > 0.03:
        return "trending"
    else:
        return "ranging"


# ===================================================================
# Strategy A: Crypto Regime Blend
# ===================================================================

def run_crypto_regime_blend(
    start_date: str = "2023-01-01",
    end_date: str = "2026-02-01",
    initial_cash: float = 5_000.0,
) -> Dict[str, Any]:
    """Combined momentum + mean reversion with regime switching.

    Based on:
    - 2025 systematic crypto study: Sharpe 1.71 blended
    - RSI momentum works on crypto (not stocks)
    - Bollinger mean reversion for ranging markets
    - Talyxion-style drawdown risk management
    """

    tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

    print(f"\n{'='*70}")
    print(f"  STRATEGY A: Crypto Regime Blend (Momentum + Mean Reversion)")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Universe: {tickers}")
    print(f"  Trending: RSI momentum (buy when RSI > 50 + positive mom)")
    print(f"  Ranging: Bollinger mean reversion (buy lower band)")
    print(f"  Risk: trailing -12%, max 80% invested")
    print(f"  Cash: ${initial_cash:,.0f}")
    print(f"{'='*70}\n")

    dl_start = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
    data = yf.download(tickers + ["BTC-USD"], start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    btc = close["BTC-USD"] if "BTC-USD" in close.columns else None
    if btc is None:
        return {"error": "No BTC data"}

    all_dates = close.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    # State
    cash = initial_cash
    positions: Dict[str, Dict] = {}
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []
    regime_log: List[str] = []

    tx_cost = 0.001          # 0.1% (Binance maker)
    # Tuned params (2026-02-23): dd_trigger 8->15%, trail_activation 3->8%
    trail_pct = 0.12         # 12% trailing stop (tighter after activation)
    dd_trigger = 0.15        # 15% per-position drawdown exit
    trail_activation_pct = 0.08  # activate trail after 8% gain
    max_exposure = 0.70      # 70% max invested
    position_pct = 0.30      # 30% per position
    rebalance_days = 3       # check every 3 days

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        # Update prices + check trailing stops
        pv = cash
        exits_today = []
        for tic, pos in list(positions.items()):
            if tic not in close.columns:
                continue
            px = float(close[tic].iloc[date_idx])
            if pd.isna(px) or px <= 0:
                continue

            pos["current_price"] = px
            if px > pos.get("high", px):
                pos["high"] = px
            pv += pos["qty"] * px

            pnl = (px - pos["entry_price"]) / pos["entry_price"]

            # Trailing stop (activate after trail_activation_pct gain)
            trail_stop = pos["high"] * (1 - trail_pct)
            if pos["high"] > pos["entry_price"] * (1 + trail_activation_pct) and px <= trail_stop:
                exits_today.append((tic, f"trail_stop", pnl))
            # Per-position drawdown exit
            elif pnl < -dd_trigger:
                exits_today.append((tic, f"dd_exit", pnl))

        for tic, reason, pnl in exits_today:
            pos = positions[tic]
            cash += pos["qty"] * pos["current_price"] * (1 - tx_cost)
            trade_log.append({
                "date": str(date)[:10], "ticker": tic, "side": "SELL",
                "qty": pos["qty"], "price": pos["current_price"],
                "pnl_pct": pnl, "reason": reason,
            })
            del positions[tic]

        pv = cash + sum(
            p["qty"] * p.get("current_price", p["entry_price"])
            for p in positions.values()
        )
        daily_values.append(pv)
        daily_dates_list.append(date)

        # Rebalance check
        if day_i % rebalance_days != 0:
            continue

        # Detect BTC regime (drives all crypto)
        regime = detect_regime(btc, date, vol_window=20, trend_window=50)
        regime_log.append(regime)

        # Drawdown-based risk management (Talyxion-style)
        if len(daily_values) > 20:
            peak_pv = max(daily_values[-20:])
            current_dd = (peak_pv - pv) / peak_pv
            if current_dd > 0.08:  # >8% drawdown from recent peak (optimized)
                # Emergency: sell everything
                for tic in list(positions.keys()):
                    pos = positions[tic]
                    px = pos.get("current_price", pos["entry_price"])
                    pnl = (px - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["qty"] * px * (1 - tx_cost)
                    trade_log.append({
                        "date": str(date)[:10], "ticker": tic, "side": "SELL",
                        "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                        "reason": "dd_risk_mgmt (>8%)",
                    })
                    del positions[tic]
                continue  # skip new entries this cycle

        for tic in tickers:
            if tic in positions:
                continue
            if tic not in close.columns:
                continue

            hist = close[tic].loc[:date].dropna()
            if len(hist) < 60:
                continue

            px = float(hist.iloc[-1])
            rsi_series = calc_rsi(hist)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
            mom_14d = float(hist.iloc[-1] / hist.iloc[-14] - 1) if len(hist) > 14 else 0

            _, bb_upper, bb_lower = calc_bollinger(hist, period=20, num_std=2.0)
            bb_low = float(bb_lower.iloc[-1]) if not bb_lower.isna().iloc[-1] else 0
            bb_up = float(bb_upper.iloc[-1]) if not bb_upper.isna().iloc[-1] else float('inf')

            buy_signal = False
            buy_reason = ""

            if regime == "trending":
                # RSI momentum: buy when RSI > 50 AND positive 14d momentum
                # This is what works in crypto (not mean reversion!)
                if rsi > 50 and mom_14d > 0.02:
                    buy_signal = True
                    buy_reason = f"TREND: rsi={rsi:.0f} mom={mom_14d*100:+.1f}%"

            elif regime == "ranging":
                # Bollinger mean reversion: buy near lower band
                if px < bb_low * 1.02 and rsi < 40:
                    buy_signal = True
                    buy_reason = f"RANGE: px=${px:,.0f} < bb_low=${bb_low:,.0f} rsi={rsi:.0f}"

            else:  # unknown - use conservative momentum
                if rsi > 55 and mom_14d > 0.05:
                    buy_signal = True
                    buy_reason = f"UNK: rsi={rsi:.0f} mom={mom_14d*100:+.1f}%"

            if not buy_signal:
                continue

            # Position sizing (capped)
            current_pv = cash + sum(
                p["qty"] * p.get("current_price", p["entry_price"])
                for p in positions.values()
            )
            current_exposure = 1 - (cash / current_pv) if current_pv > 0 else 0
            if current_exposure >= max_exposure:
                continue

            alloc = current_pv * position_pct
            alloc = min(alloc, cash * 0.90)
            if alloc < 50:
                continue

            qty = alloc / px
            cost = alloc * (1 + tx_cost)
            if cost > cash:
                continue

            cash -= cost
            positions[tic] = {
                "qty": qty, "entry_price": px, "entry_day": day_i,
                "high": px, "current_price": px,
            }
            trade_log.append({
                "date": str(date)[:10], "ticker": tic, "side": "BUY",
                "qty": qty, "price": px,
                "reason": buy_reason,
            })

        # Sell positions where signal reversed
        for tic in list(positions.keys()):
            if tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if len(hist) < 14:
                continue

            rsi_series = calc_rsi(hist)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
            mom_14d = float(hist.iloc[-1] / hist.iloc[-14] - 1)

            _, bb_upper, _ = calc_bollinger(hist, period=20, num_std=2.0)
            bb_up = float(bb_upper.iloc[-1]) if not bb_upper.isna().iloc[-1] else float('inf')
            px = float(hist.iloc[-1])

            should_sell = False
            sell_reason = ""

            if regime == "trending" and mom_14d < -0.05:
                should_sell = True
                sell_reason = f"TREND_EXIT: mom={mom_14d*100:+.1f}%"
            elif regime == "ranging" and px > bb_up * 0.98:
                should_sell = True
                sell_reason = f"RANGE_EXIT: px>${px:,.0f} > bb_up=${bb_up:,.0f}"

            if should_sell:
                pos = positions[tic]
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                cash += pos["qty"] * px * (1 - tx_cost)
                trade_log.append({
                    "date": str(date)[:10], "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                    "reason": sell_reason,
                })
                del positions[tic]

    # Regime distribution
    if regime_log:
        n_trend = regime_log.count("trending")
        n_range = regime_log.count("ranging")
        n_unk = regime_log.count("unknown")
        total = len(regime_log)
        print(f"  Regime distribution: trending={n_trend/total*100:.0f}% "
              f"ranging={n_range/total*100:.0f}% unknown={n_unk/total*100:.0f}%")

    return _calc_metrics_crypto(
        "Crypto Regime Blend",
        daily_values, daily_dates_list, initial_cash, trade_log,
        btc, bt_dates
    )


# ===================================================================
# Strategy B: Equity Regime Switch (QQQ/TLT)
# ===================================================================

def run_equity_regime_switch(
    start_date: str = "2022-01-01",
    end_date: str = "2026-02-01",
    initial_cash: float = 100_000.0,
) -> Dict[str, Any]:
    """QQQ/TLT regime switch with RSI entry timing.

    Based on:
    - Dual Momentum (Antonacci 2014): absolute + relative momentum
    - Regime switching (Price Action Lab 2024): Sharpe 1.15-1.21
    - RSI for entry timing (not reversal signal, but timing aid)

    Rules:
    1. Monthly check: QQQ 12-month momentum
    2. If QQQ mom > 0 AND RSI > 40 -> hold QQQ
    3. If QQQ mom <= 0 -> hold TLT (bonds)
    4. RSI < 30 in bull -> add to position (buy the dip)
    """

    tickers = ["QQQ", "TLT", "SPY"]

    print(f"\n{'='*70}")
    print(f"  STRATEGY B: Equity Regime Switch (QQQ/TLT)")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Rule: QQQ 12m mom > 0 -> hold QQQ, else TLT")
    print(f"  Enhancement: RSI < 30 dip buying in bull regime")
    print(f"  Cash: ${initial_cash:,.0f}")
    print(f"{'='*70}\n")

    dl_start = (pd.Timestamp(start_date) - timedelta(days=400)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    qqq = close["QQQ"]
    tlt = close["TLT"]
    spy = close["SPY"]

    all_dates = close.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    cash = initial_cash
    holding: Optional[str] = None  # QQQ or TLT
    qty = 0.0
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []
    tx_cost = 0.001
    lookback_12m = 252

    last_rebal_month = None
    extra_qty = 0.0  # from dip buying

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        # Daily PV
        if holding and holding in close.columns:
            px = float(close[holding].iloc[date_idx])
            pv = cash + (qty + extra_qty) * px
        else:
            pv = cash
        daily_values.append(pv)
        daily_dates_list.append(date)

        # Check RSI for dip buying (weekly check, only when holding QQQ)
        if holding == "QQQ" and day_i % 5 == 0:
            qqq_hist = qqq.loc[:date].dropna()
            if len(qqq_hist) > 20:
                rsi = float(calc_rsi(qqq_hist).iloc[-1])
                if rsi < 30 and cash > pv * 0.05:
                    # Buy the dip with remaining cash
                    px = float(qqq.iloc[date_idx])
                    dip_alloc = min(cash * 0.5, pv * 0.10)  # up to 10% of PV
                    if dip_alloc > 100:
                        dip_qty = dip_alloc / px
                        cash -= dip_alloc * (1 + tx_cost)
                        if cash < 0:
                            cash = 0
                        extra_qty += dip_qty
                        trade_log.append({
                            "date": str(date)[:10], "side": "BUY", "ticker": "QQQ",
                            "price": px, "reason": f"dip_buy RSI={rsi:.0f}",
                        })

        # Monthly rebalance
        month_key = (date.year, date.month)
        if month_key == last_rebal_month:
            continue
        last_rebal_month = month_key

        qqq_hist = qqq.loc[:date].dropna()
        if len(qqq_hist) < lookback_12m:
            continue

        qqq_12m = float(qqq_hist.iloc[-1] / qqq_hist.iloc[-lookback_12m] - 1)

        # Decision: QQQ momentum > 0 = bullish
        if qqq_12m > 0:
            target = "QQQ"
        else:
            target = "TLT"

        if target != holding:
            # Sell current
            if holding and (qty + extra_qty) > 0:
                px = float(close[holding].iloc[date_idx])
                proceeds = (qty + extra_qty) * px * (1 - tx_cost)
                cash += proceeds
                trade_log.append({
                    "date": str(date)[:10], "side": "SELL", "ticker": holding,
                    "price": px, "reason": f"switch to {target} (qqq_12m={qqq_12m*100:+.1f}%)",
                })
                qty = 0
                extra_qty = 0

            # Buy new
            if target in close.columns:
                px = float(close[target].iloc[date_idx])
                if not pd.isna(px) and px > 0:
                    invest = cash * 0.99
                    qty = invest / px
                    cash -= invest * (1 + tx_cost)
                    if cash < 0:
                        cash = 0
                    holding = target
                    trade_log.append({
                        "date": str(date)[:10], "side": "BUY", "ticker": target,
                        "price": px, "reason": f"qqq_12m={qqq_12m*100:+.1f}%",
                    })

    return _calc_metrics_equity(
        "Equity Regime Switch (QQQ/TLT)",
        daily_values, daily_dates_list, initial_cash, trade_log,
        spy, bt_dates
    )


# ===================================================================
# Strategy C: Combined Signal Crypto (all indicators)
# ===================================================================

def run_crypto_combined(
    start_date: str = "2023-01-01",
    end_date: str = "2026-02-01",
    initial_cash: float = 5_000.0,
) -> Dict[str, Any]:
    """Combined signal strategy using ALL evidence-based indicators.

    Signals (each 0-1, weighted):
    1. RSI Momentum (weight 0.25): RSI > 50 = bullish momentum
    2. MACD (weight 0.20): MACD line > signal = bullish
    3. Bollinger Band (weight 0.20): Price near lower band = buy
    4. Trend (weight 0.20): Above 50d SMA = bullish
    5. Volume (weight 0.15): Volume > 20d avg = confirmation

    Buy when combined score > 0.6
    Sell when combined score < 0.3 OR trailing stop hit
    """

    tickers = ["BTC-USD", "ETH-USD"]

    print(f"\n{'='*70}")
    print(f"  STRATEGY C: Combined Signal Crypto")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Universe: {tickers}")
    print(f"  Signals: RSI(0.25) + MACD(0.20) + BB(0.20) + Trend(0.20) + Vol(0.15)")
    print(f"  Buy: score > 0.6 | Sell: score < 0.3 or trail -12%")
    print(f"  Cash: ${initial_cash:,.0f}")
    print(f"{'='*70}\n")

    dl_start = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close_df = data["Close"]
        volume_df = data["Volume"]
    else:
        close_df = data[["Close"]].rename(columns={"Close": tickers[0]})
        volume_df = data[["Volume"]].rename(columns={"Volume": tickers[0]})

    btc = close_df["BTC-USD"] if "BTC-USD" in close_df.columns else None
    if btc is None:
        return {"error": "No BTC data"}

    all_dates = close_df.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    cash = initial_cash
    positions: Dict[str, Dict] = {}
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []

    tx_cost = 0.001
    trail_pct = 0.12
    max_exposure = 0.80
    position_pct = 0.40
    check_interval = 3  # every 3 days

    def calc_combined_score(tic: str, date) -> Tuple[float, str]:
        """Calculate combined score from all indicators."""
        hist = close_df[tic].loc[:date].dropna() if tic in close_df.columns else pd.Series()
        if len(hist) < 60:
            return 0.0, "insufficient_data"

        px = float(hist.iloc[-1])
        scores = {}
        details = []

        # 1. RSI Momentum (0.25)
        rsi_series = calc_rsi(hist, 14)
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
        rsi_score = min(max((rsi - 30) / 40, 0), 1)  # 30=0, 70=1
        scores["rsi"] = rsi_score * 0.25
        details.append(f"rsi={rsi:.0f}")

        # 2. MACD (0.20)
        ema12 = hist.ewm(span=12).mean()
        ema26 = hist.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = float((macd_line - signal_line).iloc[-1])
        macd_score = 1.0 if macd_hist > 0 else 0.0
        scores["macd"] = macd_score * 0.20
        details.append(f"macd={'+'if macd_hist>0 else '-'}")

        # 3. Bollinger Band Position (0.20)
        sma20, bb_up, bb_low = calc_bollinger(hist, 20, 2.0)
        bb_u = float(bb_up.iloc[-1])
        bb_l = float(bb_low.iloc[-1])
        if bb_u > bb_l:
            bb_pos = (px - bb_l) / (bb_u - bb_l)  # 0=lower, 1=upper
            # Near lower band = higher score (mean reversion buy)
            bb_score = max(0, 1 - bb_pos)
        else:
            bb_score = 0.5
        scores["bb"] = bb_score * 0.20
        details.append(f"bb={bb_pos:.2f}" if bb_u > bb_l else "bb=?")

        # 4. Trend (0.20)
        sma50 = float(hist.iloc[-50:].mean())
        trend_score = 1.0 if px > sma50 else 0.0
        scores["trend"] = trend_score * 0.20
        details.append(f"trend={'UP' if px > sma50 else 'DN'}")

        # 5. Volume (0.15)
        if tic in volume_df.columns:
            vol_hist = volume_df[tic].loc[:date].dropna()
            if len(vol_hist) > 20:
                vol_avg = float(vol_hist.iloc[-20:].mean())
                vol_current = float(vol_hist.iloc[-1])
                vol_score = min(vol_current / max(vol_avg, 1), 2.0) / 2.0
            else:
                vol_score = 0.5
        else:
            vol_score = 0.5
        scores["vol"] = vol_score * 0.15
        details.append(f"vol={vol_score:.2f}")

        total = sum(scores.values())
        return total, " ".join(details)

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        # Update + trailing stops
        pv = cash
        exits_today = []
        for tic, pos in list(positions.items()):
            if tic not in close_df.columns:
                continue
            px = float(close_df[tic].iloc[date_idx])
            if pd.isna(px) or px <= 0:
                continue

            pos["current_price"] = px
            if px > pos.get("high", px):
                pos["high"] = px
            pv += pos["qty"] * px

            trail_stop = pos["high"] * (1 - trail_pct)
            if pos["high"] > pos["entry_price"] * 1.03 and px <= trail_stop:
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                exits_today.append((tic, "trail_stop", pnl))

        for tic, reason, pnl in exits_today:
            pos = positions[tic]
            cash += pos["qty"] * pos["current_price"] * (1 - tx_cost)
            trade_log.append({
                "date": str(date)[:10], "ticker": tic, "side": "SELL",
                "qty": pos["qty"], "price": pos["current_price"],
                "pnl_pct": pnl, "reason": reason,
            })
            del positions[tic]

        pv = cash + sum(
            p["qty"] * p.get("current_price", p["entry_price"])
            for p in positions.values()
        )
        daily_values.append(pv)
        daily_dates_list.append(date)

        if day_i % check_interval != 0:
            continue

        # Score all assets
        for tic in tickers:
            score, detail = calc_combined_score(tic, date)

            # Sell if score dropped
            if tic in positions and score < 0.3:
                pos = positions[tic]
                px = pos.get("current_price", pos["entry_price"])
                pnl = (px - pos["entry_price"]) / pos["entry_price"]
                cash += pos["qty"] * px * (1 - tx_cost)
                trade_log.append({
                    "date": str(date)[:10], "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                    "reason": f"score={score:.2f} ({detail})",
                })
                del positions[tic]

            # Buy if score high enough
            if tic not in positions and score > 0.6:
                current_pv = cash + sum(
                    p["qty"] * p.get("current_price", p["entry_price"])
                    for p in positions.values()
                )
                current_exposure = 1 - (cash / current_pv) if current_pv > 0 else 0
                if current_exposure >= max_exposure:
                    continue

                hist = close_df[tic].loc[:date].dropna()
                if hist.empty:
                    continue
                px = float(hist.iloc[-1])

                alloc = current_pv * position_pct
                alloc = min(alloc, cash * 0.90)
                if alloc < 50:
                    continue

                qty = alloc / px
                cost = alloc * (1 + tx_cost)
                if cost > cash:
                    continue

                cash -= cost
                positions[tic] = {
                    "qty": qty, "entry_price": px, "entry_day": day_i,
                    "high": px, "current_price": px,
                }
                trade_log.append({
                    "date": str(date)[:10], "ticker": tic, "side": "BUY",
                    "qty": qty, "price": px,
                    "reason": f"score={score:.2f} ({detail})",
                })

    return _calc_metrics_crypto(
        "Combined Signal Crypto",
        daily_values, daily_dates_list, initial_cash, trade_log,
        btc, bt_dates
    )


# ===================================================================
# Metrics
# ===================================================================

def _calc_metrics_equity(
    name: str,
    daily_values: List[float],
    daily_dates: List,
    initial_cash: float,
    trade_log: List[Dict],
    spy: pd.Series,
    bt_dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": "Not enough data"}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 252**0.5) / max(np.std(returns), 1e-8)
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / np.maximum(peak, 1e-8)))

    spy_start = float(spy.loc[bt_dates[0]])
    spy_end = float(spy.loc[bt_dates[-1]])
    spy_return = spy_end / spy_start - 1
    alpha = total_return - spy_return

    spy_vals = spy.loc[bt_dates].values.astype(float)
    spy_peak = np.maximum.accumulate(spy_vals)
    spy_mdd = float(np.max((spy_peak - spy_vals) / np.maximum(spy_peak, 1e-8)))

    print(f"\n  {'='*60}")
    print(f"  RESULTS: {name}")
    print(f"  {'='*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%")
    print(f"  SPY B&H:         {spy_return*100:+.2f}%")
    print(f"  Alpha:           {alpha*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}% (vs SPY {spy_mdd*100:.1f}%)")
    print(f"  Trades:          {len(trade_log)}")
    print(f"  Final:           ${values[-1]:,.0f}")

    for t in trade_log:
        pnl = f" pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"    {t['date']} {t['side']:<5} {t['ticker']:<5} @ ${t['price']:>9.1f}  {t['reason']}{pnl}")

    print(f"{'='*70}\n")

    return {
        "name": name, "asset_class": "equity",
        "total_return": total_return, "bench_return": spy_return,
        "alpha": alpha, "sharpe": sharpe,
        "max_dd": max_dd, "bench_mdd": spy_mdd,
        "n_trades": len(trade_log), "final_value": values[-1],
    }


def _calc_metrics_crypto(
    name: str,
    daily_values: List[float],
    daily_dates: List,
    initial_cash: float,
    trade_log: List[Dict],
    btc: pd.Series,
    bt_dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": "Not enough data"}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 365**0.5) / max(np.std(returns), 1e-8)
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / np.maximum(peak, 1e-8)))

    btc_start = float(btc.loc[bt_dates[0]])
    btc_end = float(btc.loc[bt_dates[-1]])
    btc_return = btc_end / btc_start - 1
    alpha = total_return - btc_return

    btc_vals = btc.loc[bt_dates].values.astype(float)
    btc_peak = np.maximum.accumulate(btc_vals)
    btc_mdd = float(np.max((btc_peak - btc_vals) / np.maximum(btc_peak, 1e-8)))

    sells = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    n_wins = sum(1 for t in sells if t["pnl_pct"] > 0)
    win_rate = n_wins / len(sells) if sells else 0
    avg_win = float(np.mean([t["pnl_pct"] for t in sells if t["pnl_pct"] > 0])) if n_wins else 0
    avg_loss = float(np.mean([t["pnl_pct"] for t in sells if t["pnl_pct"] <= 0])) if (len(sells) - n_wins) else 0

    print(f"\n  {'='*60}")
    print(f"  RESULTS: {name}")
    print(f"  {'='*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%")
    print(f"  BTC B&H:         {btc_return*100:+.2f}%")
    print(f"  Alpha vs BTC:    {alpha*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}% (vs BTC {btc_mdd*100:.1f}%)")
    if sells:
        print(f"  Win Rate:        {win_rate*100:.0f}% ({n_wins}/{len(sells)})")
        print(f"  Avg Win:         {avg_win*100:+.1f}%")
        print(f"  Avg Loss:        {avg_loss*100:+.1f}%")
        if avg_loss != 0:
            print(f"  Payoff:          {abs(avg_win/avg_loss):.1f}x")
    print(f"  Trades:          {len(trade_log)}")
    print(f"  Final:           ${values[-1]:,.0f}")

    # Show last 20 trades
    for t in trade_log[-20:]:
        pnl = f" pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"    {t['date']} {t['side']:<5} {t.get('ticker','?'):<10} @ ${t['price']:>9.1f}  {t['reason']}{pnl}")

    print(f"{'='*70}\n")

    return {
        "name": name, "asset_class": "crypto",
        "total_return": total_return, "btc_return": btc_return,
        "alpha": alpha, "sharpe": sharpe,
        "max_dd": max_dd, "btc_mdd": btc_mdd,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "n_trades": len(trade_log), "final_value": values[-1],
    }


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-02-01")
    args = parser.parse_args()

    results = []

    print("\n" + "=" * 70)
    print("  BACKTEST v2: Latest Research (2024-2025)")
    print("  Testing regime-aware combined strategies")
    print("=" * 70)

    # Strategy A: Crypto Regime Blend
    r1 = run_crypto_regime_blend("2023-01-01", args.end)
    results.append(r1)

    # Strategy B: Equity Regime Switch
    r2 = run_equity_regime_switch(args.start, args.end)
    results.append(r2)

    # Strategy C: Combined Signal Crypto
    r3 = run_crypto_combined("2023-01-01", args.end)
    results.append(r3)

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Return':>10} {'Bench':>10} {'Alpha':>10} {'Sharpe':>8} {'MDD':>8} {'Trades':>7}")
    print(f"  {'-'*85}")

    for r in results:
        if "error" in r:
            print(f"  {r.get('name','?'):<30} ERROR: {r['error']}")
            continue
        bench_key = "bench_return" if "bench_return" in r else "btc_return"
        bench_val = r.get(bench_key, 0)
        print(f"  {r['name']:<30} "
              f"{r['total_return']*100:>+9.1f}% "
              f"{bench_val*100:>+9.1f}% "
              f"{r['alpha']*100:>+9.1f}% "
              f"{r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>7.1f}% "
              f"{r['n_trades']:>7}")

    # Benchmarks
    print(f"\n  {'--- BENCHMARKS ---':<30}")
    for r in results:
        if "bench_return" in r and "error" not in r:
            print(f"  {'SPY Buy&Hold':<30} {r['bench_return']*100:>+9.1f}% {'---':>10} {'  0.0%':>10} {'---':>8} {r.get('bench_mdd',0)*100:>7.1f}% {'0':>7}")
            break
    for r in results:
        if "btc_return" in r and "error" not in r:
            print(f"  {'BTC Buy&Hold':<30} {r['btc_return']*100:>+9.1f}% {'---':>10} {'  0.0%':>10} {'---':>8} {r.get('btc_mdd',0)*100:>7.1f}% {'0':>7}")
            break

    # Verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT (based on 2024-2025 research)")
    print(f"  {'='*60}")

    for r in results:
        if "error" in r:
            continue
        name = r["name"]
        alpha = r["alpha"]
        sharpe = r["sharpe"]
        mdd = r["max_dd"]

        bench_mdd = r.get("bench_mdd", r.get("btc_mdd", 1.0))

        if alpha > -0.05 and sharpe > 0.8:
            verdict = "GO"
            reason = f"alpha {alpha*100:+.1f}%, Sharpe {sharpe:.2f}"
        elif mdd < bench_mdd * 0.7 and sharpe > 0.5:
            verdict = "CONDITIONAL GO (risk mgmt)"
            reason = f"MDD {mdd*100:.0f}% vs bench {bench_mdd*100:.0f}%"
        elif alpha > -0.15 and sharpe > 0.4:
            verdict = "MARGINAL"
            reason = f"alpha {alpha*100:+.1f}%, needs improvement"
        else:
            verdict = "NO-GO"
            reason = f"alpha {alpha*100:+.1f}%, Sharpe {sharpe:.2f}"

        print(f"  {name}: {verdict}")
        print(f"    {reason}")

    print(f"\n{'='*70}")
