"""HierarchicalOnlineLearner v3/v4 — 3-Level Hierarchical Thompson Sampling.

Orchestrator class that composes functionality from focused mixins:
- RegimeMixin:      regime lifecycle, adaptive discount, dynamic prior
- RecordingMixin:   record_trade, counterfactual, correct_hold, exit_regret
- MetaUpdateMixin:  Level 0 meta-parameter learning and access
- SamplingMixin:    2-level weight sampling and accessors
- PersistenceMixin: save/load/migration/reprocess
- StatusMixin:      get_status, diary lesson injection

References:
    Carlsson et al. (IJCAI 2021): Thompson Sampling for Bandits with Clustered Arms
    Zhao et al. (arXiv:2602.15972): Hierarchical Unimodal Thompson Sampling
    Meta-Thompson Sampling (Kveton 2021, ICML): inter-task prior transfer
    MARS (2025, arXiv:2508.01173): heterogeneous agent ensemble
    Regime-Aware RL (Nixon 2025): HMM regime + risk aversion
"""

from __future__ import annotations

import time as _time_module
from pathlib import Path
from typing import Dict, List, Optional

from .types import AgentBeta, TradeRecord
from .constants import (
    SIGNAL_GROUPS, GROUP_NAMES, ALL_SIGNALS, SIGNAL_TO_GROUP,
    META_PARAMS, _GLOBAL_REGIME,
)
from .regime import RegimeMixin
from .recording import RecordingMixin
from .meta_update import MetaUpdateMixin
from .sampling import SamplingMixin
from .persistence import PersistenceMixin
from .status import StatusMixin


class HierarchicalOnlineLearner(
    RegimeMixin,
    RecordingMixin,
    MetaUpdateMixin,
    SamplingMixin,
    PersistenceMixin,
    StatusMixin,
):
    """3-Level Hierarchical Thompson Sampling for signal weight optimization.

    Level 0: 8 META PARAMS — how to trade in this regime?
    Level 1: 6 signal GROUPS — which category of analysis is most reliable?
    Level 2: ~5-7 signals per group — which specific indicator within a group?

    Final weight for signal s in group g:
        w(s) = sample(group_beta[g]) x sample(signal_beta[g][s]) / Z
    """

    # Expose constants as class attributes for backward compatibility
    SIGNAL_GROUPS = SIGNAL_GROUPS
    ALL_SIGNALS = ALL_SIGNALS
    SIGNAL_TO_GROUP = SIGNAL_TO_GROUP
    GROUP_NAMES = GROUP_NAMES
    META_PARAMS = META_PARAMS

    def __init__(
        self,
        save_path: Optional[str] = None,
        min_trades_to_adapt: int = 5,
        max_window: int = 100,
        group_discount: float = 0.99,
        signal_discount: float = 0.985,
    ) -> None:
        self._save_path = Path(save_path) if save_path else None
        self._min_trades = min_trades_to_adapt
        self._max_window = max_window
        self._group_discount = group_discount
        self._signal_discount = signal_discount

        # Level 0: meta-parameter Betas per regime
        self._meta_betas: Dict[str, Dict[str, AgentBeta]] = {
            _GLOBAL_REGIME: {p: AgentBeta(name=p) for p in META_PARAMS},
        }

        # Level 1: group-level Betas per regime
        self._group_betas: Dict[str, Dict[str, AgentBeta]] = {
            _GLOBAL_REGIME: {g: AgentBeta(name=g) for g in GROUP_NAMES},
        }

        # Level 2: signal-level Betas per regime per group
        self._signal_betas: Dict[str, Dict[str, Dict[str, AgentBeta]]] = {
            _GLOBAL_REGIME: {
                g: {s: AgentBeta(name=s) for s in sigs}
                for g, sigs in SIGNAL_GROUPS.items()
            },
        }

        self._trades: List[TradeRecord] = []
        self._total_pnl: float = 0.0
        self._regime_trade_counts: Dict[str, int] = {_GLOBAL_REGIME: 0}

        # Adaptive discount state (ADTS — arXiv:2410.04217)
        self._last_regime: Optional[str] = None
        self._regime_streak: int = 0

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def total_trades_today(self) -> int:
        """Count trades recorded today (for CF cap ratio)."""
        today = _time_module.strftime("%Y-%m-%d")
        return sum(1 for t in self._trades if hasattr(t, 'exit_time')
                   and _time_module.strftime("%Y-%m-%d", _time_module.localtime(t.exit_time)) == today)

    @property
    def has_enough_data(self) -> bool:
        return self.total_trades >= self._min_trades
