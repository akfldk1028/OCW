"""Main pre-training pipeline for H-TS.

Uses HINDSIGHT learning: actual price movements are the teacher.
Finds profitable moves in historical data, captures signals at entry,
feeds to H-TS so it learns signal-regime-profit associations.

Usage:
    cd openclaw/extensions/trading-engine/python
    .venv/bin/python3 -m pretrain.pretrain_hts --start 2021-01-01 --end 2026-03-01
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from .config import DEFAULT_START, DEFAULT_END, TICKERS, OUTPUT_DIR
from .fetch_historical import fetch_and_cache_all
from .hindsight_trades import run_hindsight_learning
from .simulate_trades import HistoricalCryptoRegimeDetector

from core.ts.hierarchical import HierarchicalOnlineLearner
from core.ts.constants import ALL_SIGNALS, SIGNAL_GROUPS, META_PARAMS, _GLOBAL_REGIME

logger = logging.getLogger("pretrain")


def run_pretrain(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    tickers: list[str] | None = None,
    output_path: str | None = None,
) -> HierarchicalOnlineLearner:
    """Execute the full pre-training pipeline with hindsight learning."""
    tickers = tickers or TICKERS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) if output_path else OUTPUT_DIR / "pretrained_learner.json"

    t0 = time.time()

    # Step 1: Fetch data
    logger.info("=" * 60)
    logger.info("Step 1: Fetching historical data (%s to %s)...", start, end)
    logger.info("=" * 60)
    all_data = fetch_and_cache_all(tickers, start, end)

    # Step 2: Build HMM regime detector (same as live CryptoRegimeDetector)
    logger.info("=" * 60)
    logger.info("Step 2: Fitting HMM CryptoRegimeDetector...")
    logger.info("=" * 60)
    btc_df = all_data.get("BTC/USDT", {}).get("klines")
    hmm_detector = None
    if btc_df is not None and not btc_df.empty:
        hmm_detector = HistoricalCryptoRegimeDetector(btc_df)

    # Step 3: Hindsight learning
    logger.info("=" * 60)
    logger.info("Step 3: Hindsight learning (market = teacher)...")
    logger.info("=" * 60)

    learner = HierarchicalOnlineLearner(
        save_path=str(output_file),
        min_trades_to_adapt=5,
        max_window=5000,
    )
    learner._reprocessing = True

    total = {"trades": 0, "wins": 0, "losses": 0, "holds": 0,
             "exit_regrets": 0, "pnl_sum": 0.0}

    for ticker, data in all_data.items():
        df = data.get("klines")
        if df is None or df.empty:
            continue

        logger.info("Processing %s (%d bars)...", ticker, len(df))
        stats = run_hindsight_learning(
            ticker=ticker,
            df=df,
            learner=learner,
            funding_df=data.get("funding"),
            oi_df=data.get("oi"),
            ls_ratio_df=data.get("ls_ratio"),
            hmm_detector=hmm_detector,
        )

        for k in ["trades", "wins", "losses", "holds", "exit_regrets", "pnl_sum"]:
            total[k] += stats.get(k, 0)

    learner._reprocessing = False
    learner.save()

    elapsed = time.time() - t0

    # Summary
    logger.info("=" * 60)
    logger.info("PRE-TRAINING COMPLETE (hindsight)")
    logger.info("=" * 60)
    n_trades = total["trades"]
    logger.info("  Trades: %d (wins=%d, losses=%d)", n_trades, total["wins"], total["losses"])
    logger.info("  Win rate: %.1f%%", total["wins"] / max(1, n_trades) * 100)
    logger.info("  Avg PnL: %+.3f%%", total["pnl_sum"] / max(1, n_trades) * 100)
    logger.info("  Correct holds: %d", total["holds"])
    logger.info("  Exit regrets: %d", total["exit_regrets"])
    logger.info("  Elapsed: %.1fs", elapsed)
    logger.info("  Output: %s", output_file)

    logger.info("\n  H-TS Meta-params (global):")
    meta = learner.get_meta_param_means()
    for p, v in meta.items():
        logger.info("    %s: %.4f", p, v)

    logger.info("\n  Regime trade counts:")
    for regime, count in sorted(learner._regime_trade_counts.items()):
        if count >= 10:
            logger.info("    %s: %d trades", regime, count)

    logger.info("\n  Group weights:")
    for group in SIGNAL_GROUPS:
        gb = learner._group_betas.get(_GLOBAL_REGIME, {}).get(group)
        if gb:
            logger.info("    %s: mean=%.4f (ESS=%.1f, trades=%d)", group, gb.mean, gb.alpha + gb.beta, gb.total_trades)

    logger.info("\n  Top/Bottom signals:")
    signal_means = []
    for group, sigs in SIGNAL_GROUPS.items():
        for sig in sigs:
            sb = learner._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig)
            if sb and sb.total_trades > 0:
                signal_means.append((sig, sb.mean, sb.total_trades))
    signal_means.sort(key=lambda x: x[1], reverse=True)
    for sig, mean, trades in signal_means[:5]:
        logger.info("    TOP %s: %.4f (%d trades)", sig, mean, trades)
    for sig, mean, trades in signal_means[-5:]:
        logger.info("    BOT %s: %.4f (%d trades)", sig, mean, trades)

    return learner


def main():
    parser = argparse.ArgumentParser(description="H-TS Historical Pre-training (Hindsight)")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date (default: {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END, help=f"End date (default: {DEFAULT_END})")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("trading-engine.online_learner").setLevel(logging.WARNING)

    run_pretrain(
        start=args.start,
        end=args.end,
        tickers=args.tickers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
