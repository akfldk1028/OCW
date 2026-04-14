"""Hindsight trade generator — learns from actual price movements.

Instead of forward-simulating with rule strategies (which lose money),
this module looks BACKWARD at actual price movements and creates trades
that we KNOW were profitable. Then H-TS learns which signals were present
at those profitable entry points.

Approach:
1. Scan price data for significant moves (+3%~+15% rallies, -3%~-15% drops)
2. At each move's START, capture all 32 signals
3. Create a "trade" that entered at the start and exited at the peak/trough
4. Feed these profitable trades to H-TS → learns signal-regime associations

This is supervised learning disguised as RL — the market itself is the teacher.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .compute_signals import compute_ta_at_bar, add_derivatives_signals
from .config import ROUND_TRIP_FEE

import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from core.agent_tools import _calc_ema_series, _calc_supertrend, _calc_atr
from core.ts.hierarchical import HierarchicalOnlineLearner

logger = logging.getLogger("pretrain.hindsight")


def find_significant_moves(
    df: pd.DataFrame,
    min_move_pct: float = 0.03,
    max_hold_bars: int = 72,
    min_hold_bars: int = 3,
) -> List[Dict]:
    """Find all significant price moves in historical data.

    Scans every bar as potential entry, looks forward up to max_hold_bars
    to find the best exit (maximum favorable excursion).

    Returns list of {entry_idx, exit_idx, side, pnl_pct, mfe, mae, held_bars}
    """
    moves = []
    n = len(df)
    closes = df["close"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()

    i = 200  # need lookback for signals
    while i < n - min_hold_bars:
        entry_price = closes[i]
        if entry_price <= 0:
            i += 1
            continue

        # Look forward: find best long and short outcomes
        best_long_pnl = 0.0
        best_long_exit = i + 1
        best_short_pnl = 0.0
        best_short_exit = i + 1
        worst_long = 0.0
        worst_short = 0.0

        running_high = entry_price
        running_low = entry_price

        for j in range(i + 1, min(i + max_hold_bars + 1, n)):
            running_high = max(running_high, highs[j])
            running_low = min(running_low, lows[j])

            # Long PnL at this bar
            long_pnl = (closes[j] - entry_price) / entry_price
            if long_pnl > best_long_pnl:
                best_long_pnl = long_pnl
                best_long_exit = j

            # Short PnL at this bar
            short_pnl = (entry_price - closes[j]) / entry_price
            if short_pnl > best_short_pnl:
                best_short_pnl = short_pnl
                best_short_exit = j

        # Record significant long move
        if best_long_pnl >= min_move_pct:
            long_mfe = (running_high - entry_price) / entry_price
            long_mae = (running_low - entry_price) / entry_price
            held = best_long_exit - i
            if held >= min_hold_bars:
                moves.append({
                    "entry_idx": i,
                    "exit_idx": best_long_exit,
                    "side": "long",
                    "entry_price": entry_price,
                    "exit_price": closes[best_long_exit],
                    "pnl_pct": best_long_pnl - ROUND_TRIP_FEE,
                    "mfe": long_mfe,
                    "mae": long_mae,
                    "held_bars": held,
                    "capture_ratio": min(1.0, best_long_pnl / long_mfe) if long_mfe > 0 else 0,
                })

        # Record significant short move
        if best_short_pnl >= min_move_pct:
            short_mfe = (entry_price - running_low) / entry_price
            short_mae = (entry_price - running_high) / entry_price
            held = best_short_exit - i
            if held >= min_hold_bars:
                moves.append({
                    "entry_idx": i,
                    "exit_idx": best_short_exit,
                    "side": "short",
                    "entry_price": entry_price,
                    "exit_price": closes[best_short_exit],
                    "pnl_pct": best_short_pnl - ROUND_TRIP_FEE,
                    "mfe": short_mfe,
                    "mae": short_mae,
                    "held_bars": held,
                    "capture_ratio": min(1.0, best_short_pnl / short_mfe) if short_mfe > 0 else 0,
                })

        # Skip forward to avoid overlapping entries (minimum 2 bar gap)
        i += 2

    return moves


def find_losing_holds(
    df: pd.DataFrame,
    max_hold_bars: int = 48,
    min_loss_pct: float = 0.02,
) -> List[Dict]:
    """Find periods where NOT trading was correct (price went nowhere or reversed).

    These become correct_hold and counterfactual learning signals.
    Sampled less frequently than winners to maintain positive bias.
    """
    holds = []
    n = len(df)
    closes = df["close"].tolist()

    for i in range(200, n - max_hold_bars, 6):  # every 6 bars
        entry_price = closes[i]
        if entry_price <= 0:
            continue

        # Check if buying here would have lost money within window
        future_min = min(closes[i+1:i+max_hold_bars+1])
        future_max = max(closes[i+1:i+max_hold_bars+1])

        long_worst = (future_min - entry_price) / entry_price
        short_worst = (entry_price - future_max) / entry_price

        # Both directions lose → correct to HOLD
        if long_worst < -min_loss_pct and short_worst < -min_loss_pct:
            # Find when the worst happened
            worst_idx = i + 1
            for j in range(i + 1, min(i + max_hold_bars + 1, n)):
                if closes[j] == future_min:
                    worst_idx = j
                    break

            holds.append({
                "entry_idx": i,
                "check_idx": worst_idx,
                "price_at_hold": entry_price,
                "price_later": closes[worst_idx],
                "raw_pnl": (closes[worst_idx] - entry_price) / entry_price,
            })

    return holds


def run_hindsight_learning(
    ticker: str,
    df: pd.DataFrame,
    learner: HierarchicalOnlineLearner,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ls_ratio_df: Optional[pd.DataFrame] = None,
    hmm_detector=None,
) -> Dict:
    """Run hindsight-based learning for one ticker.

    1. Find all significant profitable moves
    2. At each entry point, compute signals + regime
    3. Feed as record_trade() to H-TS
    4. Also find correct holds → record_correct_hold()
    5. After exits, check future prices → record_exit_regret()
    """
    from .simulate_trades import compute_regime_at_bar

    n = len(df)
    if n < 300:
        return {"trades": 0}

    all_closes = df["close"].tolist()
    ema9_series = _calc_ema_series(all_closes, 9)
    ema21_series = _calc_ema_series(all_closes, 21)

    stats = {"trades": 0, "wins": 0, "losses": 0, "holds": 0,
             "exit_regrets": 0, "pnl_sum": 0.0}

    # --- Step 1: Find profitable moves ---
    logger.info("  Finding significant moves...")
    # Multiple thresholds to get diverse trade sizes
    all_moves = []
    for min_pct in [0.02, 0.03, 0.05, 0.08, 0.12]:
        for max_bars in [12, 24, 48, 72]:
            moves = find_significant_moves(df, min_move_pct=min_pct, max_hold_bars=max_bars)
            all_moves.extend(moves)

    # Deduplicate by entry_idx+side (keep largest move)
    seen = {}
    for m in all_moves:
        key = (m["entry_idx"], m["side"])
        if key not in seen or m["pnl_pct"] > seen[key]["pnl_pct"]:
            seen[key] = m
    moves = list(seen.values())
    random.shuffle(moves)

    logger.info("  Found %d significant moves (long=%d, short=%d)",
                len(moves),
                sum(1 for m in moves if m["side"] == "long"),
                sum(1 for m in moves if m["side"] == "short"))

    # --- Step 2: Also create LOSING trades (for balanced learning) ---
    # Take the same entries but exit at the WORST point instead of best
    losing_moves = []
    for m in moves[:len(moves) // 3]:  # 1/3 of winners become losers
        entry_idx = m["entry_idx"]
        side = "short" if m["side"] == "long" else "long"  # opposite side = loss

        # Find worst exit for opposite side
        entry_price = m["entry_price"]
        worst_pnl = 0.0
        worst_exit = entry_idx + 3

        for j in range(entry_idx + 1, min(entry_idx + 25, n)):
            if side == "long":
                pnl = (all_closes[j] - entry_price) / entry_price
            else:
                pnl = (entry_price - all_closes[j]) / entry_price

            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_exit = j

        if worst_pnl < -0.01:  # at least -1% loss
            exit_price = all_closes[worst_exit]
            highs_slice = df["high"].iloc[entry_idx:worst_exit+1]
            lows_slice = df["low"].iloc[entry_idx:worst_exit+1]

            if side == "long":
                mfe = (highs_slice.max() - entry_price) / entry_price
                mae = (lows_slice.min() - entry_price) / entry_price
            else:
                mfe = (entry_price - lows_slice.min()) / entry_price
                mae = (entry_price - highs_slice.max()) / entry_price

            losing_moves.append({
                "entry_idx": entry_idx,
                "exit_idx": worst_exit,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": worst_pnl - ROUND_TRIP_FEE,
                "mfe": mfe,
                "mae": mae,
                "held_bars": worst_exit - entry_idx,
                "capture_ratio": 0.0,
            })

    logger.info("  Created %d losing trades for balance", len(losing_moves))

    # --- Step 3: Feed all trades to H-TS ---
    all_trades = moves + losing_moves
    random.shuffle(all_trades)

    for trade in all_trades:
        idx = trade["entry_idx"]
        regime = compute_regime_at_bar(df, idx, all_closes, ema9_series, ema21_series, hmm_detector)

        # Compute full signal snapshot at entry
        ta_snapshot = compute_ta_at_bar(df, idx, 200)
        add_derivatives_signals(ta_snapshot, df.index[idx], trade["entry_price"],
                                funding_df, oi_df, ls_ratio_df)

        if not ta_snapshot:
            continue

        # Confidence based on move magnitude (bigger move = higher confidence)
        confidence = min(0.90, 0.40 + abs(trade["pnl_pct"]) * 5)
        position_pct = random.uniform(0.05, 0.15)

        learner.record_trade(
            ticker=ticker,
            entry_price=trade["entry_price"],
            exit_price=trade["exit_price"],
            pnl_pct=trade["pnl_pct"],
            held_hours=float(trade["held_bars"]),
            agent_signals=ta_snapshot,  # all signals as agent_signals
            regime=regime,
            position_pct_used=position_pct,
            confidence_at_entry=confidence,
            position_side=trade["side"],
            ta_snapshot=None,
            mfe=trade["mfe"],
            mae=trade["mae"],
            capture_ratio=trade["capture_ratio"],
        )

        stats["trades"] += 1
        if trade["pnl_pct"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["pnl_sum"] += trade["pnl_pct"]

        # --- Exit regret: check what happened AFTER exit ---
        exit_idx = trade["exit_idx"]
        for horizon_bars, discount in [(6, 0.15), (24, 0.12), (48, 0.10)]:
            check_idx = exit_idx + horizon_bars
            if check_idx >= n:
                continue

            future_price = all_closes[check_idx]
            if trade["side"] == "long":
                hypo_pnl = (future_price - trade["entry_price"]) / trade["entry_price"]
            else:
                hypo_pnl = (trade["entry_price"] - future_price) / trade["entry_price"]

            additional = hypo_pnl - trade["pnl_pct"]

            if additional > 0.003:
                # Premature exit
                learner.record_exit_regret(
                    ticker=ticker,
                    exit_price=trade["exit_price"],
                    price_now=future_price,
                    pnl_at_exit=trade["pnl_pct"],
                    held_hours=float(trade["held_bars"]),
                    agent_signals=ta_snapshot,
                    regime=regime,
                    position_side=trade["side"],
                    discount_factor=discount,
                    was_premature=True,
                )
                stats["exit_regrets"] += 1
                break
            elif additional < -0.003:
                # Validated exit
                learner.record_exit_regret(
                    ticker=ticker,
                    exit_price=trade["exit_price"],
                    price_now=future_price,
                    pnl_at_exit=trade["pnl_pct"],
                    held_hours=float(trade["held_bars"]),
                    agent_signals=ta_snapshot,
                    regime=regime,
                    position_side=trade["side"],
                    discount_factor=discount,
                    was_premature=False,
                )
                stats["exit_regrets"] += 1
                break

    # --- Step 4: Correct holds ---
    logger.info("  Finding correct holds...")
    holds = find_losing_holds(df)
    for hold in holds[:len(moves) // 2]:  # cap at half of winners
        idx = hold["entry_idx"]
        ta_signals = compute_ta_at_bar(df, idx, 200)
        add_derivatives_signals(ta_signals, df.index[idx], hold["price_at_hold"],
                                funding_df, oi_df, ls_ratio_df)
        if not ta_signals:
            continue

        regime = compute_regime_at_bar(df, idx, all_closes, ema9_series, ema21_series, hmm_detector)

        learner.record_correct_hold(
            ticker=ticker,
            price_at_hold=hold["price_at_hold"],
            price_now=hold["price_later"],
            ta_signals=ta_signals,
            regime=regime,
            discount_factor=0.25,
        )
        stats["holds"] += 1

    # --- Step 5: Counterfactual (missed opportunities in opposite direction) ---
    for m in moves[:len(moves) // 4]:
        idx = m["entry_idx"]
        check_idx = min(idx + 12, n - 1)
        ta_signals = compute_ta_at_bar(df, idx, 200)
        add_derivatives_signals(ta_signals, df.index[idx], m["entry_price"],
                                funding_df, oi_df, ls_ratio_df)
        if not ta_signals:
            continue

        regime = compute_regime_at_bar(df, idx, all_closes, ema9_series, ema21_series, hmm_detector)

        learner.record_counterfactual(
            ticker=ticker,
            price_at_hold=m["entry_price"],
            price_now=m["exit_price"],
            ta_signals=ta_signals,
            regime=regime,
            discount_factor=0.25,
        )

    wr = stats["wins"] / max(1, stats["trades"]) * 100
    avg_pnl = stats["pnl_sum"] / max(1, stats["trades"]) * 100
    logger.info("  %s: %d trades (WR=%.1f%%, avg=%.3f%%), %d holds, %d exit_regrets",
                ticker, stats["trades"], wr, avg_pnl, stats["holds"], stats["exit_regrets"])

    return stats
