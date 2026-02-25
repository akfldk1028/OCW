"""Full v3 pipeline backtest — scan → rank → synthesize → execute.

Simulates the complete multi-agent pipeline over historical data to validate
the full system (not just individual components).

Usage::

    python backtest_pipeline.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SWING_EXIT_CONFIG
from agents.quant_agent import (
    ALL_STOCKS,
    ETF_TICKERS,
    FEATURE_NAMES,
    FORWARD_HORIZON,
    PURGE_DAYS,
    SECTOR_MAP,
    TICKER_SECTOR,
    compute_zscore_features,
    _rank_sectors,
    _compute_labels,
)

# ── Signal weights (same as synthesizer.py) ──

AGENT_WEIGHTS = {
    "quant": 0.35,       # XGBoost P(top-quartile) — primary
    "sector": 0.25,      # sector momentum (MarketAgent proxy)
    "momentum": 0.20,    # raw price momentum
    "regime": 0.10,      # regime bias
    "mean_rev": 0.10,    # RSI mean-reversion (replaces RL/sentiment in backtest)
}

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.10
MAX_POSITION_PCT = 0.12
MAX_POSITIONS = 8
MAX_EXPOSURE = 0.80
TX_COST = 0.0015  # 0.15% per trade (conservative default)


def _detect_regime(spy_close: pd.Series, eval_date) -> str:
    """Simple regime detection: realized vol vs 20d median."""
    hist = spy_close.loc[:eval_date].dropna()
    if len(hist) < 60:
        return "low_volatility"
    returns = hist.pct_change().dropna()
    vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
    vol_60d_median = returns.rolling(20).std().iloc[-60:].median() * np.sqrt(252)
    return "high_volatility" if vol_20d > vol_60d_median * 1.3 else "low_volatility"


def _regime_signal(regime: str, ticker: str) -> float:
    """Regime-based bias per ticker's sector."""
    sector = TICKER_SECTOR.get(ticker, "")
    defensive = {"Healthcare", "Staples", "Utilities", "RealEstate"}
    growth = {"Technology", "Semis", "Communication", "ConsDisc", "Financials"}

    if regime == "high_volatility":
        if sector in defensive:
            return 0.3
        if sector in growth:
            return -0.2
    else:
        if sector in growth:
            return 0.1
        if sector in defensive:
            return -0.1
    return 0.0


def _score_candidates(
    zscored: Dict[str, Dict],
    sector_scores: Dict[str, float],
    regime: str,
    model=None,
) -> List[Dict[str, Any]]:
    """Score all candidates using weighted signals (no RL/sentiment in backtest)."""
    scored = []
    for tic, feats in zscored.items():
        price = feats.get("price", 0)
        if price <= 0:
            continue

        # 1. Quant signal: XGBoost P(top-quartile) if model available
        if model is not None:
            X = np.array([[feats[f] for f in FEATURE_NAMES]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                proba = model.predict_proba(X)[0]
                classes = list(model.classes_)
                p_dict = {cls: float(p) for cls, p in zip(classes, proba)}
                p_win = p_dict.get(1, 0.25)
            except Exception:
                p_win = 0.5
            quant_sig = (p_win - 0.5) * 2
        else:
            # Fallback: momentum z-score as proxy
            quant_sig = max(-1.0, min(1.0, feats.get("momentum_21d", 0.0) / 3.0))

        # 2. Sector signal
        sec_name = TICKER_SECTOR.get(tic, "")
        sec_score = sector_scores.get(sec_name, 0.0)
        sector_sig = max(-1.0, min(1.0, sec_score * 5))

        # 3. Momentum signal
        mom_z = feats.get("momentum_21d", 0.0)
        mom_sig = max(-1.0, min(1.0, mom_z / 3.0))

        # 4. Regime signal
        regime_sig = _regime_signal(regime, tic)

        # 5. Mean-reversion (RSI-based)
        rsi_z = feats.get("rsi_14", 0.0)
        # Overbought (high z) → sell, oversold (low z) → buy
        mean_rev_sig = max(-1.0, min(1.0, -rsi_z / 3.0))

        final_score = (
            AGENT_WEIGHTS["quant"] * quant_sig
            + AGENT_WEIGHTS["sector"] * sector_sig
            + AGENT_WEIGHTS["momentum"] * mom_sig
            + AGENT_WEIGHTS["regime"] * regime_sig
            + AGENT_WEIGHTS["mean_rev"] * mean_rev_sig
        )
        final_score = max(-1.0, min(1.0, final_score))

        scored.append({
            "ticker": tic,
            "score": final_score,
            "price": price,
            "p_win": p_win if model else 0.5,
            "signals": {
                "quant": quant_sig,
                "sector": sector_sig,
                "momentum": mom_sig,
                "regime": regime_sig,
                "mean_rev": mean_rev_sig,
            },
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def run_pipeline_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2026-02-01",
    rebalance_days: int = 21,
    initial_cash: float = 100_000.0,
    train_months: int = 24,
    use_xgboost: bool = True,
) -> Dict[str, Any]:
    """Run full v3 pipeline backtest.

    For each rebalance period:
        1. Detect regime (HMM proxy)
        2. Rank sectors
        3. Compute z-scored features → XGBoost ranking
        4. Weighted signal combination
        5. Position management (buy/sell/exit)
        6. Track daily portfolio value
    """
    print(f"\n{'='*70}")
    print(f"  FULL PIPELINE BACKTEST (v3)")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Rebalance: every {rebalance_days} trading days")
    print(f"  XGBoost: {'ON' if use_xgboost else 'OFF (momentum proxy)'}")
    print(f"  Initial: ${initial_cash:,.0f}")
    print(f"{'='*70}\n")

    # ── Download data ──
    # Extra lookback for training + indicators
    dl_start = (pd.Timestamp(start_date) - timedelta(days=(train_months + 6) * 30)).strftime("%Y-%m-%d")
    all_tickers = list(set(ALL_STOCKS + ETF_TICKERS + ["SPY"]))

    print("  Downloading data...")
    t0 = time.time()
    data = yf.download(all_tickers, start=dl_start, end=end_date, progress=False, auto_adjust=True)
    print(f"  Downloaded in {time.time() - t0:.0f}s")

    def extract(field):
        if isinstance(data.columns, pd.MultiIndex):
            return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
        return data

    stock_close = extract("Close")
    stock_high = extract("High")
    stock_low = extract("Low")
    stock_volume = extract("Volume")
    stock_open = extract("Open")

    if stock_close.empty:
        print("  ERROR: No data")
        return {"error": "No data"}

    # ── Identify backtest trading days ──
    all_dates = stock_close.index
    bt_start_idx = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start_idx:]

    spy_close = stock_close["SPY"] if "SPY" in stock_close.columns else None
    if spy_close is None:
        print("  ERROR: No SPY data")
        return {"error": "No SPY data"}

    # ── XGBoost model (trained on pre-backtest data) ──
    model = None
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            print("  Training XGBoost on pre-backtest data...")
            model = _train_xgboost(
                ALL_STOCKS, all_dates, bt_start_idx, train_months,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
            )
            if model is not None:
                print("  XGBoost model trained.")
            else:
                print("  XGBoost training failed — using momentum proxy.")
        except ImportError:
            print("  xgboost not installed — using momentum proxy.")

    # ── Simulation state ──
    cash = initial_cash
    positions: Dict[str, Dict[str, Any]] = {}  # ticker -> {qty, entry_price, entry_idx}
    daily_values = []
    daily_dates_list = []
    trade_log = []
    rebalance_count = 0
    scan_miss_counts: Dict[str, int] = {}

    # ── Main loop: daily tracking + periodic rebalance ──
    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        # Portfolio value
        pv = cash
        for tic, pos in positions.items():
            if tic in stock_close.columns:
                p = stock_close[tic].iloc[date_idx]
                if not pd.isna(p) and p > 0:
                    pv += pos["qty"] * p
        daily_values.append(pv)
        daily_dates_list.append(date)

        # ── Rebalance day? ──
        if day_i % rebalance_days != 0:
            continue

        rebalance_count += 1

        # Step 1: Regime
        regime = _detect_regime(spy_close, date)
        exposure_scale = 0.7 if regime == "high_volatility" else 1.0

        # Step 2: Sector scores
        etf_cols = [c for c in ETF_TICKERS if c in stock_close.columns]
        etf_close = stock_close[etf_cols] if etf_cols else pd.DataFrame()
        sector_scores = _rank_sectors(etf_close, date) if not etf_close.empty else {}

        # Pick top 3 sectors
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [s[0] for s in sorted_sectors[:3]]

        # Candidate tickers from top sectors
        candidates = []
        for sec_name in top_sectors:
            if sec_name in SECTOR_MAP:
                candidates.extend(SECTOR_MAP[sec_name]["stocks"])
        # Expand if too few
        if len(candidates) < 15:
            for sec_name, _ in sorted_sectors[3:]:
                if sec_name in SECTOR_MAP:
                    candidates.extend(SECTOR_MAP[sec_name]["stocks"])
                if len(candidates) >= 30:
                    break
        candidates = list(set(candidates))

        # Step 3: Z-scored features
        zscored = compute_zscore_features(
            candidates, date,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            sector_scores,
        )

        # Retrain XGBoost periodically (every 3 rebalances ≈ quarterly)
        if use_xgboost and model is not None and rebalance_count % 3 == 0:
            new_model = _train_xgboost(
                ALL_STOCKS, all_dates, date_idx, train_months,
                stock_close, stock_high, stock_low, stock_volume, stock_open,
            )
            if new_model is not None:
                model = new_model

        # Step 4: Score candidates
        scored = _score_candidates(zscored, sector_scores, regime, model)
        candidate_set = set(s["ticker"] for s in scored)

        # Step 5: EXIT management — evaluate held positions
        # Update scan miss counts
        for tic in list(positions.keys()):
            if tic in candidate_set:
                scan_miss_counts.pop(tic, None)
            else:
                scan_miss_counts[tic] = scan_miss_counts.get(tic, 0) + 1

        sells = []
        for tic in list(positions.keys()):
            pos = positions[tic]
            if tic not in stock_close.columns:
                continue
            current_price = stock_close[tic].iloc[date_idx]
            if pd.isna(current_price) or current_price <= 0:
                continue
            entry_price = pos["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price
            hold_days = date_idx - pos["entry_idx"]
            miss_count = scan_miss_counts.get(tic, 0)

            should_sell = False
            reason = ""

            if pnl_pct <= SWING_EXIT_CONFIG["stop_loss_pct"]:
                should_sell = True
                reason = f"stop_loss ({pnl_pct:+.1%})"
            elif pnl_pct >= SWING_EXIT_CONFIG["take_profit_pct"]:
                should_sell = True
                reason = f"take_profit ({pnl_pct:+.1%})"
            elif hold_days > SWING_EXIT_CONFIG["max_hold_days"] * 5 and pnl_pct < SWING_EXIT_CONFIG["min_profit_pct"]:
                # max_hold_days is in calendar-ish days; multiply by ~5 for trading days equiv
                should_sell = True
                reason = f"timeout ({hold_days}d, {pnl_pct:+.1%})"
            elif miss_count >= SWING_EXIT_CONFIG["consecutive_miss_limit"]:
                should_sell = True
                reason = f"scan_miss x{miss_count}"

            # Also sell if score is strongly negative
            tic_scored = [s for s in scored if s["ticker"] == tic]
            if tic_scored and tic_scored[0]["score"] < SELL_THRESHOLD:
                should_sell = True
                reason = f"negative_signal ({tic_scored[0]['score']:+.2f})"

            if should_sell:
                proceeds = pos["qty"] * current_price * (1 - TX_COST)
                cash += proceeds
                trade_log.append({
                    "date": str(date)[:10], "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": current_price, "pnl_pct": pnl_pct,
                    "reason": reason,
                })
                sells.append(tic)

        for tic in sells:
            del positions[tic]
            scan_miss_counts.pop(tic, None)

        # Step 6: BUY new positions
        buys = [s for s in scored if s["score"] > BUY_THRESHOLD and s["ticker"] not in positions]
        n_slots = MAX_POSITIONS - len(positions)
        buys = buys[:n_slots]

        for entry in buys:
            tic = entry["ticker"]
            price = entry["price"]
            if price <= 0:
                continue

            # Position sizing
            size_pct = min(abs(entry["score"]) * 0.15, MAX_POSITION_PCT) * exposure_scale
            pv_now = cash + sum(
                positions[t]["qty"] * stock_close[t].iloc[date_idx]
                for t in positions if t in stock_close.columns
                and not pd.isna(stock_close[t].iloc[date_idx])
            )
            alloc = size_pct * pv_now
            alloc = min(alloc, cash * 0.4)  # max 40% of cash per buy
            if alloc < 200 or alloc > cash:
                continue

            qty = alloc / price
            cost = alloc * (1 + TX_COST)
            if cost > cash:
                continue

            cash -= cost
            positions[tic] = {"qty": qty, "entry_price": price, "entry_idx": date_idx}
            trade_log.append({
                "date": str(date)[:10], "ticker": tic, "side": "BUY",
                "qty": qty, "price": price, "score": entry["score"],
                "reason": f"signal={entry['score']:+.2f}",
            })

    # ── Metrics ──
    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": "Not enough data"}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret * 252**0.5) / max(std_ret, 1e-8)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(drawdown))

    # SPY buy & hold
    spy_start = spy_close.loc[bt_dates[0]]
    spy_end = spy_close.loc[bt_dates[-1]]
    spy_return = (spy_end / spy_start - 1) if spy_start > 0 else 0
    alpha = total_return - spy_return

    # Win rate
    sells_log = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    n_wins = sum(1 for t in sells_log if t["pnl_pct"] > 0)
    win_rate = n_wins / len(sells_log) if sells_log else 0

    # Turnover
    total_buy_value = sum(t["qty"] * t["price"] for t in trade_log if t["side"] == "BUY")
    avg_pv = np.mean(values)
    turnover = total_buy_value / avg_pv if avg_pv > 0 else 0

    # ── Print results ──
    print(f"\n  {'─'*60}")
    print(f"  RESULTS")
    print(f"  {'─'*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%  (${values[-1] - initial_cash:+,.0f})")
    print(f"  SPY B&H:         {spy_return*100:+.2f}%")
    print(f"  Alpha:           {alpha*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}%")
    print(f"  Win Rate:        {win_rate*100:.0f}%  ({n_wins}/{len(sells_log)} trades)")
    print(f"  Trades:          {len(trade_log)} total  ({len(sells_log)} round-trips)")
    print(f"  Rebalances:      {rebalance_count}")
    print(f"  Turnover:        {turnover:.1f}x")
    print(f"  Final Portfolio:  ${values[-1]:,.0f}")
    print(f"  Final Positions: {len(positions)}")

    # Rebalance-by-rebalance P&L
    print(f"\n  {'─'*60}")
    print(f"  REBALANCE PERIODS")
    print(f"  {'─'*60}")
    print(f"  {'#':<4} {'Date':<12} {'Value':>10} {'Period':>8} {'Cumul':>8} {'Regime':<8}")

    rb_indices = list(range(0, len(daily_values), rebalance_days))
    for i, idx in enumerate(rb_indices):
        val = daily_values[idx]
        cum_ret = (val - initial_cash) / initial_cash
        if i > 0:
            prev_val = daily_values[rb_indices[i-1]]
            period_ret = (val - prev_val) / prev_val
        else:
            period_ret = 0.0
        date_str = str(daily_dates_list[idx])[:10] if idx < len(daily_dates_list) else "?"
        regime = _detect_regime(spy_close, daily_dates_list[idx]) if idx < len(daily_dates_list) else "?"
        r_label = "hi_vol" if regime == "high_volatility" else "lo_vol"
        print(f"  {i+1:<4} {date_str:<12} ${val:>9,.0f} {period_ret*100:>+7.2f}% {cum_ret*100:>+7.2f}% {r_label:<8}")

    # Recent trades
    print(f"\n  {'─'*60}")
    print(f"  LAST 20 TRADES")
    print(f"  {'─'*60}")
    for t in trade_log[-20:]:
        pnl_str = f"  pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"  {t['date']}  {t['side']:<5} {t['ticker']:<6} {t['qty']:>7.1f} @ ${t['price']:>7.1f}  {t['reason']}{pnl_str}")

    print(f"\n{'='*70}\n")

    return {
        "total_return": total_return,
        "spy_return": float(spy_return),
        "alpha": alpha,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "num_trades": len(trade_log),
        "num_round_trips": len(sells_log),
        "rebalance_count": rebalance_count,
        "turnover": turnover,
        "final_value": values[-1],
        "trade_log": trade_log,
        "daily_values": daily_values,
    }


def _train_xgboost(
    all_tickers, all_dates, eval_idx, train_months,
    stock_close, stock_high, stock_low, stock_volume, stock_open,
):
    """Train XGBoost on historical data up to eval_idx."""
    from xgboost import XGBClassifier

    etf_cols = [c for c in ETF_TICKERS if c in stock_close.columns]
    etf_close = stock_close[etf_cols] if etf_cols else pd.DataFrame()

    X_train, y_train = [], []

    for month_offset in range(train_months, 0, -1):
        feature_date_idx = eval_idx - (month_offset + 1) * 21
        if feature_date_idx < 63:
            continue
        target_end_idx = feature_date_idx + FORWARD_HORIZON
        if target_end_idx >= eval_idx - PURGE_DAYS:
            continue

        feature_date = all_dates[feature_date_idx]
        sec_scores = _rank_sectors(etf_close, feature_date) if not etf_close.empty else {}

        zscored = compute_zscore_features(
            all_tickers, feature_date,
            stock_close, stock_high, stock_low, stock_volume, stock_open,
            sec_scores,
        )

        labels = _compute_labels(all_tickers, feature_date_idx, FORWARD_HORIZON, stock_close)

        for tic, feats in zscored.items():
            if tic not in labels:
                continue
            X_train.append([feats[f] for f in FEATURE_NAMES])
            y_train.append(labels[tic])

    if len(X_train) < 80:
        return None

    X = np.nan_to_num(np.array(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_train)

    unique = np.unique(y)
    if len(unique) < 2:
        return None

    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.03,
        reg_alpha=0.1,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        gamma=0.1,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # Default: 2-year backtest with 21-day rebalance
    result = run_pipeline_backtest(
        start_date="2024-06-01",
        end_date="2026-02-01",
        rebalance_days=21,
        initial_cash=100_000,
        use_xgboost=True,
    )

    if "error" not in result:
        print(f"\n  SUMMARY")
        print(f"  Strategy: {result['total_return']*100:+.2f}%")
        print(f"  SPY B&H:  {result['spy_return']*100:+.2f}%")
        print(f"  Alpha:    {result['alpha']*100:+.2f}%")
        print(f"  Sharpe:   {result['sharpe']:.2f}")
        print(f"  MDD:      {result['max_dd']*100:.1f}%")
        print(f"  Win Rate: {result['win_rate']*100:.0f}%")
