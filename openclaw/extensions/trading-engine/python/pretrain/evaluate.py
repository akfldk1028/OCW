"""Evaluate pre-trained H-TS model quality.

Compares pre-trained vs cold-start on holdout data:
- Signal alignment accuracy
- Regime coverage
- Effective sample size
- Meta-param convergence
- Holdout prediction accuracy
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from core.ts.hierarchical import HierarchicalOnlineLearner
from core.ts.constants import ALL_SIGNALS, SIGNAL_GROUPS, META_PARAMS, _GLOBAL_REGIME

from .config import OUTPUT_DIR

logger = logging.getLogger("pretrain.evaluate")


def evaluate_pretrained(
    learner: HierarchicalOnlineLearner,
    holdout_trades: list | None = None,
) -> Dict:
    """Run all evaluation checks on a pre-trained learner."""
    results = {}

    # 1. Signal alignment check
    logger.info("=" * 50)
    logger.info("EVALUATION: Pre-trained H-TS Model")
    logger.info("=" * 50)

    # Effective sample size (alpha + beta sum)
    logger.info("\n1. Effective Sample Size:")
    signal_ess = {}
    for group, sigs in SIGNAL_GROUPS.items():
        for sig in sigs:
            sb = learner._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig)
            if sb:
                ess = sb.alpha + sb.beta
                signal_ess[sig] = ess

    if signal_ess:
        avg_ess = sum(signal_ess.values()) / len(signal_ess)
        active_signals = sum(1 for v in signal_ess.values() if v > 3.0)  # beyond prior
        logger.info("  Average ESS: %.1f", avg_ess)
        logger.info("  Active signals (ESS > 3): %d / %d", active_signals, len(signal_ess))
        results["avg_ess"] = avg_ess
        results["active_signals"] = active_signals

    # 2. Regime coverage
    logger.info("\n2. Regime Coverage:")
    regime_counts = learner._regime_trade_counts
    regimes_with_10plus = sum(1 for c in regime_counts.values() if c >= 10)
    logger.info("  Total regimes: %d", len(regime_counts))
    logger.info("  Regimes with 10+ trades: %d", regimes_with_10plus)
    for regime, count in sorted(regime_counts.items()):
        logger.info("    %s: %d trades", regime, count)
    results["regime_count"] = len(regime_counts)
    results["regimes_10plus"] = regimes_with_10plus

    # 3. Meta-param convergence
    logger.info("\n3. Meta-parameter Convergence:")
    meta = learner.get_meta_param_means()
    meta_ok = True
    for param, value in meta.items():
        in_range = 0.3 <= value <= 0.7
        status = "OK" if in_range else "WARN (extreme)"
        logger.info("  %s: %.4f [%s]", param, value, status)
        if not in_range:
            meta_ok = False
    results["meta_convergence_ok"] = meta_ok
    results["meta_params"] = meta

    # 4. Signal mean distribution check (bias check)
    logger.info("\n4. Signal Bias Check:")
    extreme_signals = []
    for sig, ess in signal_ess.items():
        group = None
        for g, sigs in SIGNAL_GROUPS.items():
            if sig in sigs:
                group = g
                break
        if group:
            sb = learner._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig)
            if sb and sb.total_trades > 0:
                mean = sb.mean
                if mean < 0.15 or mean > 0.85:
                    extreme_signals.append((sig, mean))

    if extreme_signals:
        logger.info("  Extreme signals (outside 0.15-0.85):")
        for sig, mean in extreme_signals:
            logger.info("    %s: %.4f", sig, mean)
    else:
        logger.info("  All signal means within 0.15-0.85 range")
    results["extreme_signals"] = len(extreme_signals)

    # 5. Group weight balance check
    logger.info("\n5. Group Weight Balance:")
    group_means = {}
    for group in SIGNAL_GROUPS:
        gb = learner._group_betas.get(_GLOBAL_REGIME, {}).get(group)
        if gb:
            group_means[group] = gb.mean
            logger.info("  %s: %.4f (ESS=%.1f)", group, gb.mean, gb.alpha + gb.beta)

    if group_means:
        max_w = max(group_means.values())
        min_w = min(group_means.values())
        ratio = max_w / min_w if min_w > 0 else float('inf')
        balanced = ratio < 3.0
        logger.info("  Max/min ratio: %.2f (%s)", ratio, "OK" if balanced else "WARN: imbalanced")
        results["group_balance_ratio"] = ratio
        results["group_balanced"] = balanced

    # 6. Holdout prediction accuracy
    if holdout_trades:
        logger.info("\n6. Holdout Prediction Accuracy:")
        correct = 0
        total_holdout = 0

        for trade in holdout_trades:
            if not trade.agent_signals:
                continue

            # Simple prediction: weighted sum of signals > 0 → predict long profitable
            total_holdout += 1
            weighted_sum = 0.0
            for sig_name, sig_val in trade.agent_signals.items():
                if abs(sig_val) < 0.05:
                    continue
                group = None
                for g, sigs in SIGNAL_GROUPS.items():
                    if sig_name in sigs:
                        group = g
                        break
                if group:
                    sb = learner._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig_name)
                    gb = learner._group_betas.get(_GLOBAL_REGIME, {}).get(group)
                    if sb and gb:
                        weight = gb.mean * sb.mean
                        weighted_sum += sig_val * weight

            if trade.position_side == "short":
                predict_profitable = weighted_sum < 0
            else:
                predict_profitable = weighted_sum > 0
            actual_profitable = trade.pnl_pct > 0

            if predict_profitable == actual_profitable:
                correct += 1

        if total_holdout > 0:
            accuracy = correct / total_holdout * 100
            logger.info("  Holdout trades: %d", total_holdout)
            logger.info("  Prediction accuracy: %.1f%%", accuracy)
            results["holdout_accuracy"] = accuracy
            results["holdout_count"] = total_holdout
    else:
        logger.info("\n6. No holdout trades provided, skipping prediction check")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)

    checks = [
        ("Regime coverage (>=3 with 10+ trades)", regimes_with_10plus >= 3),
        ("Meta-params in range (0.3-0.7)", meta_ok),
        ("No extreme signal bias", len(extreme_signals) == 0),
        ("Group balance (<3x ratio)", results.get("group_balanced", False)),
        ("Active signals (>50%)", results.get("active_signals", 0) > len(ALL_SIGNALS) // 2),
    ]

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        logger.info("  [%s] %s", "PASS" if ok else "FAIL", name)
    logger.info("  Result: %d/%d checks passed", passed, len(checks))
    results["checks_passed"] = passed
    results["checks_total"] = len(checks)

    return results


def compare_with_cold_start(
    pretrained: HierarchicalOnlineLearner,
    holdout_trades: list,
) -> Dict:
    """Compare pre-trained vs cold-start on holdout data."""
    logger.info("\n" + "=" * 50)
    logger.info("COMPARISON: Pre-trained vs Cold-start")
    logger.info("=" * 50)

    # Create cold-start learner
    cold = HierarchicalOnlineLearner(min_trades_to_adapt=5)
    cold._reprocessing = True

    # Feed holdout to both
    pretrained_copy = HierarchicalOnlineLearner(min_trades_to_adapt=5)
    pretrained_copy._reprocessing = True

    # Copy pre-trained state
    pretrained_copy._meta_betas = {
        r: {p: type(mb)(name=mb.name, alpha=mb.alpha, beta=mb.beta, total_trades=mb.total_trades)
            for p, mb in params.items()}
        for r, params in pretrained._meta_betas.items()
    }
    pretrained_copy._group_betas = {
        r: {g: type(gb)(name=gb.name, alpha=gb.alpha, beta=gb.beta, total_trades=gb.total_trades)
            for g, gb in groups.items()}
        for r, groups in pretrained._group_betas.items()
    }
    pretrained_copy._signal_betas = {
        r: {g: {s: type(sb)(name=sb.name, alpha=sb.alpha, beta=sb.beta, total_trades=sb.total_trades)
                for s, sb in sigs.items()}
            for g, sigs in groups.items()}
        for r, groups in pretrained._signal_betas.items()
    }
    pretrained_copy._regime_trade_counts = dict(pretrained._regime_trade_counts)

    pretrained_wins = 0
    cold_wins = 0

    for trade in holdout_trades:
        # Pre-trained prediction
        pre_weighted = 0.0
        cold_weighted = 0.0

        for sig_name, sig_val in trade.agent_signals.items():
            if abs(sig_val) < 0.05:
                continue
            group = None
            for g, sigs in SIGNAL_GROUPS.items():
                if sig_name in sigs:
                    group = g
                    break
            if group:
                # Pre-trained weights
                sb_pre = pretrained_copy._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig_name)
                gb_pre = pretrained_copy._group_betas.get(_GLOBAL_REGIME, {}).get(group)
                if sb_pre and gb_pre:
                    pre_weighted += sig_val * gb_pre.mean * sb_pre.mean

                # Cold-start weights
                sb_cold = cold._signal_betas.get(_GLOBAL_REGIME, {}).get(group, {}).get(sig_name)
                gb_cold = cold._group_betas.get(_GLOBAL_REGIME, {}).get(group)
                if sb_cold and gb_cold:
                    cold_weighted += sig_val * gb_cold.mean * sb_cold.mean

        actual = trade.pnl_pct > 0
        if trade.position_side == "short":
            pre_correct = (pre_weighted < 0) == actual
            cold_correct = (cold_weighted < 0) == actual
        else:
            pre_correct = (pre_weighted > 0) == actual
            cold_correct = (cold_weighted > 0) == actual

        if pre_correct:
            pretrained_wins += 1
        if cold_correct:
            cold_wins += 1

        # Feed trade to both for online learning
        for l in [pretrained_copy, cold]:
            l.record_trade(
                ticker=trade.ticker,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl_pct=trade.pnl_pct,
                held_hours=trade.held_hours,
                agent_signals=trade.agent_signals,
                regime=trade.regime,
                position_side=trade.position_side,
                mfe=trade.mfe,
                mae=trade.mae,
                capture_ratio=trade.capture_ratio,
            )

    n = len(holdout_trades) or 1
    pre_acc = pretrained_wins / n * 100
    cold_acc = cold_wins / n * 100

    logger.info("  Pre-trained accuracy: %.1f%% (%d/%d)", pre_acc, pretrained_wins, n)
    logger.info("  Cold-start accuracy:  %.1f%% (%d/%d)", cold_acc, cold_wins, n)
    logger.info("  Advantage: %+.1f pp", pre_acc - cold_acc)

    return {
        "pretrained_accuracy": pre_acc,
        "cold_accuracy": cold_acc,
        "advantage_pp": pre_acc - cold_acc,
    }
