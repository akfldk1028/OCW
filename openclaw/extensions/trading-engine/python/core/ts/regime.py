"""Regime management mixin for Hierarchical Thompson Sampling."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .types import AgentBeta
from .constants import (
    SIGNAL_GROUPS, GROUP_NAMES, META_PARAMS, _GLOBAL_REGIME,
    _MIN_REGIME_TRADES,
)

logger = logging.getLogger("trading-engine.online_learner")


class RegimeMixin:
    """Regime lifecycle: creation, warm-start, adaptive discount, fallback."""

    # --- Adaptive Discount (ADTS — arXiv:2410.04217) ---

    def _get_adaptive_discount(self, regime: str) -> float:
        """Return regime-dependent adaptive discount rate.

        - Regime just changed (streak <= 1): fast discount (0.92)
        - Stable regime (streak >= 5):       slow discount (0.995)
        - Transitional (2-4 trades):         default discount
        """
        if regime in (None, "unknown", _GLOBAL_REGIME):
            return self._group_discount

        if self._last_regime is None:
            return self._group_discount

        if regime != self._last_regime:
            return 0.92

        if self._regime_streak >= 5:
            return 0.995

        return self._group_discount

    def _update_regime_streak(self, regime: str) -> None:
        """Track consecutive trades in the same regime for adaptive discount."""
        if regime in (None, "unknown", _GLOBAL_REGIME):
            return

        if regime == self._last_regime:
            self._regime_streak += 1
        else:
            self._regime_streak = 1
            logger.info(
                "[h-ts] Regime transition: %s -> %s (adaptive discount -> 0.92)",
                self._last_regime, regime,
            )
        self._last_regime = regime

    # --- Dynamic Prior (arXiv:2602.00943) ---

    @staticmethod
    def _dynamic_prior(src: AgentBeta, epsilon: float = 0.15,
                       n_total: float = 4.0) -> tuple:
        """Compute Dynamic Prior for cold-start.

        Sets prior mean so P(new_arm > incumbent) = epsilon at introduction.
        Falls back to Beta(2,2) if no global data.
        """
        if src.total_trades == 0:
            return 2.0, 2.0

        prior_mean = src.mean * (1.0 - epsilon)
        prior_mean = max(0.1, min(0.9, prior_mean))
        a = n_total * prior_mean
        b = n_total * (1.0 - prior_mean)
        return a, b

    # --- Regime Lifecycle ---

    def _ensure_regime(self, regime: str) -> None:
        """Create Beta structures for a new regime with Dynamic Prior warm-start."""
        if regime not in self._group_betas:
            # Level 1: groups
            g_global = self._group_betas.get(_GLOBAL_REGIME, {})
            self._group_betas[regime] = {}
            for g in GROUP_NAMES:
                src = g_global.get(g)
                if src and src.total_trades > 0:
                    a, b = self._dynamic_prior(src)
                    ab = AgentBeta(name=g, alpha=a, beta=b)
                    ab.total_trades = 0
                else:
                    ab = AgentBeta(name=g)
                self._group_betas[regime][g] = ab

            # Level 2: signals
            s_global = self._signal_betas.get(_GLOBAL_REGIME, {})
            self._signal_betas[regime] = {}
            for g, sigs in SIGNAL_GROUPS.items():
                self._signal_betas[regime][g] = {}
                sg = s_global.get(g, {})
                for s in sigs:
                    src = sg.get(s)
                    if src and src.total_trades > 0:
                        a, b = self._dynamic_prior(src)
                        sb = AgentBeta(name=s, alpha=a, beta=b)
                        sb.total_trades = 0
                    else:
                        sb = AgentBeta(name=s)
                    self._signal_betas[regime][g][s] = sb

            # Level 0: meta
            m_global = self._meta_betas.get(_GLOBAL_REGIME, {})
            self._meta_betas[regime] = {}
            for p in META_PARAMS:
                src = m_global.get(p)
                if src and src.total_trades > 0:
                    a, b = self._dynamic_prior(src)
                    mb = AgentBeta(name=p, alpha=a, beta=b)
                    mb.total_trades = 0
                else:
                    mb = AgentBeta(name=p)
                self._meta_betas[regime][p] = mb

            self._regime_trade_counts.setdefault(regime, 0)
            logger.info("[h-ts] New regime '%s' Dynamic Prior from global (epsilon=0.15)",
                        regime)

    def _effective_regime(self, regime: str) -> str:
        """Return regime key to use, falling back to global if insufficient data."""
        count = self._regime_trade_counts.get(regime, 0)
        if count >= _MIN_REGIME_TRADES and regime in self._group_betas:
            return regime
        return _GLOBAL_REGIME
