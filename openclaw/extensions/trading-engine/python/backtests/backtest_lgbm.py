"""LightGBM short-timeframe backtest for crypto trading.

Tests LightGBM signal on 15m candles with rolling retrain.
Compares: LightGBM-only, Regime Blend-only, Ensemble (vote), Buy & Hold.

Based on: arXiv 2511.00665, 2503.18096, 2309.00626

Usage:
    python backtests/backtest_lgbm.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config import LGBM_SIGNAL_CONFIG, REGIME_BLEND_CONFIG
from analysis.lgbm_signal import LGBMSignalModel, compute_features

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
START = "2025-03-01"
END = "2026-02-01"
INITIAL_CASH = 5_000.0
TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TICKER_MAP = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT"}
INTERVAL = "15m"
TRAIN_WINDOW_BARS = 90 * 96  # 90 days * 96 bars/day
RETRAIN_BARS = 7 * 96        # weekly
HORIZON = 16                 # 4 hours = 16 bars
TX_COST = 0.001
REPORT_PATH = _ROOT / "backtests" / "backtest_lgbm_report.txt"

CFG = dict(LGBM_SIGNAL_CONFIG)


def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical klines from Binance REST API."""
    import time as _time
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    all_data = []
    current = start_ms
    req_count = 0

    while current < end_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    API error (req #{req_count}): {e}, retrying in 5s...")
            _time.sleep(5)
            continue
        if not data:
            break

        all_data.extend(data)
        current = data[-1][0] + 1  # next ms after last candle
        req_count += 1
        if req_count % 10 == 0:
            print(f"    ... {len(all_data)} bars fetched ({req_count} requests)", flush=True)
        if len(data) < 1000:
            break
        _time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def regime_signal(close_daily: pd.Series, cfg: dict) -> str:
    """Simplified regime detection from daily close (same as RegimeBlendDetectNode)."""
    if len(close_daily) < 60:
        return "unknown"

    vol_w = cfg.get("vol_window", 20)
    trend_w = cfg.get("trend_window", 50)

    total_range = close_daily.iloc[-vol_w:].max() - close_daily.iloc[-vol_w:].min()
    net_move = abs(float(close_daily.iloc[-1]) - float(close_daily.iloc[-vol_w]))
    efficiency = net_move / total_range if total_range > 0 else 0

    sma = close_daily.iloc[-trend_w:].mean()
    trend_strength = abs(float(close_daily.iloc[-1]) - sma) / sma if sma > 0 else 0

    if efficiency > 0.5 and trend_strength > 0.03:
        return "trending"
    return "ranging"


def regime_blend_signal(close_daily: pd.Series, regime: str, cfg: dict) -> str:
    """Simplified regime blend signal (same logic as RegimeBlendSignalNode)."""
    if len(close_daily) < 60:
        return "HOLD"

    px = float(close_daily.iloc[-1])

    # RSI
    delta = close_daily.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])

    # Momentum
    mom_14d = float(close_daily.iloc[-1] / close_daily.iloc[-14] - 1) if len(close_daily) > 14 else 0

    # Bollinger
    sma20 = close_daily.rolling(20).mean()
    std20 = close_daily.rolling(20).std()
    bb_lower = float((sma20 - 2 * std20).iloc[-1])
    bb_upper = float((sma20 + 2 * std20).iloc[-1])

    if regime == "trending":
        if rsi > cfg["trending_rsi_threshold"] and mom_14d > cfg["trending_momentum_threshold"]:
            return "BUY"
        elif mom_14d < cfg["trending_exit_momentum"]:
            return "SELL"
    elif regime == "ranging":
        if px < bb_lower * cfg["ranging_bb_threshold"] and rsi < cfg["ranging_rsi_threshold"]:
            return "BUY"
        elif px > bb_upper * cfg["ranging_exit_bb_factor"]:
            return "SELL"

    return "HOLD"


def run():
    lines = []

    def log(msg: str):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 70)
    log("  LGBM SHORT-TIMEFRAME BACKTEST")
    log(f"  Period: {START} ~ {END}")
    log(f"  Interval: {INTERVAL}, Horizon: {HORIZON} bars (4h)")
    log(f"  Initial: ${INITIAL_CASH:,.0f}")
    log(f"  Tickers: {TICKERS}")
    log("=" * 70)

    # ----------------------------------------------------------------
    # Download data
    # ----------------------------------------------------------------
    start_dt = datetime.strptime(START, "%Y-%m-%d") - timedelta(days=100)  # extra for training
    end_dt = datetime.strptime(END, "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    ticker_data = {}
    for tic in TICKERS:
        symbol = TICKER_MAP[tic]
        log(f"\nDownloading {symbol} {INTERVAL} data...")
        df = fetch_binance_klines(symbol, INTERVAL, start_ms, end_ms)
        if df.empty:
            log(f"  WARN: No data for {symbol}")
            continue
        ticker_data[tic] = df
        log(f"  Got {len(df)} bars ({df.index[0]} ~ {df.index[-1]})")

    if not ticker_data:
        log("ERROR: No data downloaded")
        return

    # ----------------------------------------------------------------
    # Build daily close for regime detection
    # ----------------------------------------------------------------
    daily_closes = {}
    for tic, df in ticker_data.items():
        daily_closes[tic] = df["close"].resample("1D").last().dropna()

    # ----------------------------------------------------------------
    # Backtest
    # ----------------------------------------------------------------
    bt_start = pd.Timestamp(START)
    model = LGBMSignalModel(CFG)

    # State per strategy: lgbm_only, blend_only, ensemble, bh
    strategies = {
        "lgbm_only": {"cash": INITIAL_CASH, "positions": {}, "highs": {}, "trades": []},
        "blend_only": {"cash": INITIAL_CASH, "positions": {}, "highs": {}, "trades": []},
        "ensemble": {"cash": INITIAL_CASH, "positions": {}, "highs": {}, "trades": []},
    }
    bh_start_prices = {}

    daily_pvs = {s: [] for s in strategies}
    daily_pvs["bh"] = []
    daily_dates = []

    # Determine bar indices for backtest period
    ref_tic = TICKERS[0]
    ref_df = ticker_data[ref_tic]
    bt_mask = ref_df.index >= bt_start
    bt_indices = ref_df.index[bt_mask]

    if len(bt_indices) == 0:
        log("ERROR: No bars in backtest period")
        return

    log(f"\nBacktest bars: {len(bt_indices)} ({bt_indices[0]} ~ {bt_indices[-1]})")

    # Record B&H start prices
    for tic in TICKERS:
        if tic in ticker_data:
            mask = ticker_data[tic].index >= bt_start
            if mask.any():
                bh_start_prices[tic] = float(ticker_data[tic].loc[mask, "close"].iloc[0])

    last_decision_bar = {s: -999 for s in strategies}
    decision_interval = 16  # decide every 4 hours (16 bars)
    last_daily_check = None

    for bar_i, bar_time in enumerate(bt_indices):
        # ---- Daily PV snapshot (once per day) ----
        bar_date = bar_time.date()
        if bar_date != last_daily_check:
            last_daily_check = bar_date
            for sname, state in strategies.items():
                pv = state["cash"]
                for tic, pos in state["positions"].items():
                    if tic in ticker_data:
                        idx = ticker_data[tic].index.get_indexer([bar_time], method="pad")[0]
                        if idx >= 0:
                            pv += pos["qty"] * float(ticker_data[tic]["close"].iloc[idx])
                daily_pvs[sname].append(pv)

            # B&H PV
            bh_pv = 0
            alloc = INITIAL_CASH / len(bh_start_prices) if bh_start_prices else INITIAL_CASH
            for tic, start_px in bh_start_prices.items():
                if tic in ticker_data:
                    idx = ticker_data[tic].index.get_indexer([bar_time], method="pad")[0]
                    if idx >= 0:
                        cur_px = float(ticker_data[tic]["close"].iloc[idx])
                        bh_pv += alloc * cur_px / start_px
            daily_pvs["bh"].append(bh_pv or INITIAL_CASH)
            daily_dates.append(bar_date)

        # ---- Risk check every bar ----
        for sname, state in strategies.items():
            risk_exits = []
            for tic in list(state["positions"].keys()):
                if tic not in ticker_data:
                    continue
                idx = ticker_data[tic].index.get_indexer([bar_time], method="pad")[0]
                if idx < 0:
                    continue
                px = float(ticker_data[tic]["close"].iloc[idx])
                pos = state["positions"][tic]
                pnl = (px - pos["entry_price"]) / pos["entry_price"]

                # Update trailing high
                prev_high = state["highs"].get(tic, pos["entry_price"])
                if px > prev_high:
                    state["highs"][tic] = px
                    prev_high = px

                reason = None
                if prev_high > pos["entry_price"] * (1 + CFG["trail_activation_pct"]):
                    trail_stop = prev_high * (1 - CFG["trail_pct"])
                    if px <= trail_stop:
                        reason = f"trail_stop (high={prev_high:.0f})"

                if reason is None and pnl < -CFG["dd_trigger"]:
                    reason = "dd_exit"

                if reason:
                    risk_exits.append((tic, px, pnl, reason))

            for tic, px, pnl, reason in risk_exits:
                pos = state["positions"][tic]
                state["cash"] += pos["qty"] * px * (1 - TX_COST)
                state["trades"].append({
                    "time": str(bar_time), "ticker": tic, "side": "SELL",
                    "price": px, "pnl_pct": pnl, "reason": f"RISK: {reason}",
                })
                del state["positions"][tic]
                state["highs"].pop(tic, None)

        # ---- Decision every 4 hours ----
        if bar_i % decision_interval != 0:
            continue

        for tic in TICKERS:
            if tic not in ticker_data:
                continue

            df = ticker_data[tic]
            idx = df.index.get_indexer([bar_time], method="pad")[0]
            if idx < 0:
                continue

            # Current OHLCV slice for LightGBM
            train_start = max(0, idx - TRAIN_WINDOW_BARS)
            ohlcv_slice = df.iloc[train_start:idx + 1]
            px = float(df["close"].iloc[idx])

            # Train/retrain if needed
            if model.needs_retrain(tic, idx):
                result = model.train(ohlcv_slice, tic)
                if "error" not in result:
                    if bar_i % (96 * 30) == 0:  # log monthly
                        log(f"  [{bar_time.date()}] {tic} retrain: AUC={result['auc']:.3f} "
                            f"acc={result['val_accuracy']:.3f} n={result['n_samples']}")

            # LightGBM signal
            lgbm_pred = model.predict(ohlcv_slice, tic)
            lgbm_signal = lgbm_pred["signal"]

            # Regime Blend signal (from daily close)
            daily_close = daily_closes.get(tic, pd.Series())
            daily_up_to = daily_close[daily_close.index <= bar_time]
            regime = regime_signal(daily_up_to, REGIME_BLEND_CONFIG)
            blend_signal = regime_blend_signal(daily_up_to, regime, REGIME_BLEND_CONFIG)

            # ---- Execute per strategy ----
            for sname, state in strategies.items():
                if sname == "lgbm_only":
                    sig = lgbm_signal
                elif sname == "blend_only":
                    sig = blend_signal
                elif sname == "ensemble":
                    # Both must agree
                    if lgbm_signal == blend_signal and lgbm_signal != "HOLD":
                        sig = lgbm_signal
                    else:
                        sig = "HOLD"
                else:
                    sig = "HOLD"

                # Execute
                if sig == "BUY" and tic not in state["positions"]:
                    alloc = state["cash"] * CFG["position_pct"]
                    if alloc < 10:
                        continue
                    qty = alloc / (px * (1 + TX_COST))
                    if qty <= 0:
                        continue
                    cost = qty * px * (1 + TX_COST)
                    state["cash"] -= cost
                    state["positions"][tic] = {"qty": qty, "entry_price": px, "entry_time": str(bar_time)}
                    state["highs"][tic] = px
                    state["trades"].append({
                        "time": str(bar_time), "ticker": tic, "side": "BUY",
                        "price": px, "pnl_pct": 0, "reason": f"{sname}:{lgbm_pred.get('reason','')}|{regime}:{blend_signal}",
                    })

                elif sig == "SELL" and tic in state["positions"]:
                    pos = state["positions"][tic]
                    pnl = (px - pos["entry_price"]) / pos["entry_price"]
                    state["cash"] += pos["qty"] * px * (1 - TX_COST)
                    state["trades"].append({
                        "time": str(bar_time), "ticker": tic, "side": "SELL",
                        "price": px, "pnl_pct": pnl, "reason": f"{sname}:{lgbm_pred.get('reason','')}|{regime}:{blend_signal}",
                    })
                    del state["positions"][tic]
                    state["highs"].pop(tic, None)

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  PERFORMANCE COMPARISON")
    log("=" * 70)

    for sname in list(strategies.keys()) + ["bh"]:
        if sname == "bh":
            pv_list = daily_pvs["bh"]
            trades = []
            label = "Buy & Hold"
        else:
            pv_list = daily_pvs[sname]
            trades = strategies[sname]["trades"]
            label = sname

        if not pv_list:
            log(f"\n  {label}: NO DATA")
            continue

        pv_series = pd.Series(pv_list, dtype=float)
        final_pv = pv_list[-1]
        total_ret = final_pv / INITIAL_CASH - 1
        daily_ret = pv_series.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * (365 ** 0.5) if daily_ret.std() > 0 else 0
        running_max = pv_series.cummax()
        dd = ((pv_series - running_max) / running_max)
        max_dd = dd.min()

        sells = [t for t in trades if t["side"] == "SELL"]
        wins = [t for t in sells if t["pnl_pct"] > 0]
        losses = [t for t in sells if t["pnl_pct"] <= 0]
        win_rate = len(wins) / len(sells) if sells else 0
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

        log(f"\n  --- {label} ---")
        log(f"  Final Value:   ${final_pv:,.0f}")
        log(f"  Total Return:  {total_ret:+.1%}")
        log(f"  Sharpe:        {sharpe:.2f}")
        log(f"  Max Drawdown:  {max_dd:.1%}")
        if sells:
            log(f"  Trades:        {len(trades)} ({len(sells)} round-trips)")
            log(f"  Win Rate:      {win_rate:.0%} ({len(wins)}W / {len(losses)}L)")
            log(f"  Avg Win:       {avg_win:+.1%}")
            log(f"  Avg Loss:      {avg_loss:+.1%}")
            risk_exits = [t for t in trades if "RISK" in t.get("reason", "")]
            log(f"  Risk Exits:    {len(risk_exits)}")

    # Feature importance
    log("\n  --- Feature Importance (BTC/USDT) ---")
    for name, score in model.feature_importance("BTC/USDT", top_n=10):
        log(f"  {name:<25} {score}")

    # Alpha comparison
    bh_ret = daily_pvs["bh"][-1] / INITIAL_CASH - 1 if daily_pvs["bh"] else 0
    log(f"\n  --- Alpha vs Buy & Hold ({bh_ret:+.1%}) ---")
    for sname in strategies:
        if daily_pvs[sname]:
            strat_ret = daily_pvs[sname][-1] / INITIAL_CASH - 1
            alpha = strat_ret - bh_ret
            log(f"  {sname:<15} Alpha: {alpha:+.1%}")

    log("\n" + "=" * 70)

    # Save report
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run()
