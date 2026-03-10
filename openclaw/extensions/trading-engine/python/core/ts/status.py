"""Status and diary bridge for Hierarchical Thompson Sampling."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

from .constants import (
    SIGNAL_GROUPS, GROUP_NAMES, ALL_SIGNALS, SIGNAL_TO_GROUP,
    _GLOBAL_REGIME, _MIN_REGIME_TRADES,
)

logger = logging.getLogger("trading-engine.online_learner")


class StatusMixin:
    """Introspection (get_status) and diary lesson injection."""

    def get_status(self) -> Dict[str, Any]:
        global_groups = self._group_betas.get(_GLOBAL_REGIME, {})
        global_signals = self._signal_betas.get(_GLOBAL_REGIME, {})

        group_summary = {}
        for g, gb in global_groups.items():
            signals_in_group = {}
            for s, sb in global_signals.get(g, {}).items():
                signals_in_group[s] = sb.to_dict()
            group_summary[g] = {
                "group_beta": gb.to_dict(),
                "signals": signals_in_group,
            }

        global_meta = self._meta_betas.get(_GLOBAL_REGIME, {})
        meta_params_global = {p: mb.to_dict() for p, mb in global_meta.items()}

        regime_info = {}
        for regime in self._group_betas:
            if regime == _GLOBAL_REGIME:
                continue
            count = self._regime_trade_counts.get(regime, 0)
            regime_meta = self._meta_betas.get(regime, {})
            # Signal reliability per regime
            regime_signals = self._signal_betas.get(regime, {})
            signal_reliability = {}
            for g, sigs in SIGNAL_GROUPS.items():
                for s in sigs:
                    sb = regime_signals.get(g, {}).get(s)
                    signal_reliability[s] = {
                        "mean": round(sb.mean, 4) if sb else 0.5,
                        "total_trades": sb.total_trades if sb else 0,
                    }
            regime_info[regime] = {
                "trade_count": count,
                "using_own_weights": count >= _MIN_REGIME_TRADES,
                "group_means": {
                    g: round(gb.mean, 4)
                    for g, gb in self._group_betas[regime].items()
                },
                "meta_param_means": {
                    p: round(mb.mean, 4)
                    for p, mb in regime_meta.items()
                },
                "signal_reliability": signal_reliability,
            }

        return {
            "version": 4,
            "total_trades": self.total_trades,
            "has_enough_data": self.has_enough_data,
            "min_trades_to_adapt": self._min_trades,
            "cumulative_pnl_pct": round(self._total_pnl * 100, 2),
            "num_groups": len(GROUP_NAMES),
            "num_signals": len(ALL_SIGNALS),
            "group_summary": group_summary,
            "meta_params_global": meta_params_global,
            "meta_param_means_global": self.get_meta_param_means(),
            "regime_info": regime_info,
            "regime_trade_counts": dict(self._regime_trade_counts),
            "adaptive_discount": {
                "last_regime": self._last_regime,
                "regime_streak": self._regime_streak,
                "current_discount": self._get_adaptive_discount(
                    self._last_regime or "unknown"
                ),
            },
            "group_weights_global": self.get_group_weights(),
            "mean_weights_global": self.get_mean_weights(),
            "recent_trades": [
                {
                    "ticker": t.ticker,
                    "pnl_pct": t.pnl_pct,
                    "held_hours": t.held_hours,
                    "agent_signals": t.agent_signals,
                    "regime": t.regime,
                    "market": t.market_type,
                }
                for t in self._trades[-10:]
            ],
        }

    # ------------------------------------------------------------------
    # Diary L3 -> H-TS Prior Bridge
    # ------------------------------------------------------------------
    def apply_diary_lessons(self, lessons: list[dict]) -> int:
        """Inject L3 semantic lessons into H-TS as Bayesian prior adjustments.

        Each lesson has:
            text, regime_scope, signals (list), strength, occurrences, decay_factor

        Returns number of adjustments made.
        """
        BOOST_FACTOR = 0.15
        MIN_EFFECTIVE = 0.3
        MAX_BOOST = 0.5

        adjustments = 0

        for lesson in lessons:
            effective = lesson.get("strength", 0.5) * lesson.get("decay_factor", 1.0)
            if effective < MIN_EFFECTIVE:
                continue

            signals = lesson.get("signals", [])
            if not signals:
                continue

            occurrences = min(lesson.get("occurrences", 1), 5)
            boost = min(effective * occurrences * BOOST_FACTOR, MAX_BOOST)

            regime_scope = lesson.get("regime_scope", "*")

            target_regimes = []
            if regime_scope == "*":
                target_regimes = list(self._group_betas.keys())
            else:
                # Strip wildcard "*" and match remaining parts
                # e.g. "high_*" → {"high"}, matches "high_volatility_ranging_reflation"
                scope_parts = set(
                    p for p in regime_scope.replace("_", " ").split()
                    if p != "*"
                )
                if not scope_parts:
                    target_regimes = list(self._group_betas.keys())
                else:
                    for regime in self._group_betas:
                        regime_parts = set(regime.replace("_", " ").split())
                        if scope_parts.issubset(regime_parts):
                            target_regimes.append(regime)
                if not target_regimes:
                    target_regimes = [_GLOBAL_REGIME]

            for regime in target_regimes:
                for sig_name in signals:
                    group = SIGNAL_TO_GROUP.get(sig_name)
                    if not group:
                        continue

                    if regime not in self._signal_betas:
                        if regime == _GLOBAL_REGIME:
                            continue
                        self._ensure_regime(regime)
                        logger.debug("[h-ts] Created regime '%s' for diary lesson", regime)

                    sig_betas = self._signal_betas[regime].get(group, {})
                    sb = sig_betas.get(sig_name)
                    outcome = lesson.get("outcome", "unknown")
                    if sb:
                        if not (math.isfinite(sb.alpha) and math.isfinite(sb.beta)):
                            sb.alpha, sb.beta = 2.0, 2.0
                        if outcome == "loss":
                            sb.beta = min(sb.beta + boost, 100.0)
                        else:
                            sb.alpha = min(sb.alpha + boost, 100.0)
                        adjustments += 1

                    gb = self._group_betas.get(regime, {}).get(group)
                    if gb:
                        if outcome == "loss":
                            gb.beta += boost * 0.5
                        else:
                            gb.alpha += boost * 0.5

        if adjustments > 0:
            self.save()
            logger.info(
                "[H-TS] Diary prior injection: %d signal adjustments from %d lessons",
                adjustments, len(lessons),
            )

        return adjustments
