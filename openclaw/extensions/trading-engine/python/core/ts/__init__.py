"""Hierarchical Thompson Sampling for agent weight adaptation.

Two architectures available:
1. OnlineLearner (v2): Flat 8-arm regime-aware TS (legacy)
2. HierarchicalOnlineLearner (v3/v4): 3-level TS with 6 groups x ~5-7 signals = ~28 arms

Modules:
- types:       AgentBeta (Beta distribution), TradeRecord
- constants:   SIGNAL_GROUPS, META_PARAMS, regime constants
- regime:      RegimeMixin — lifecycle, adaptive discount, dynamic prior
- recording:   RecordingMixin — trade/CF/hold/regret recording
- meta_update: MetaUpdateMixin — Level 0 meta-parameter learning
- sampling:    SamplingMixin — 2-level weight sampling
- persistence: PersistenceMixin — save/load/migration
- status:      StatusMixin — introspection, diary bridge

References:
- Thompson (1933): original paper
- Agrawal & Goyal (2012): near-optimal regret bounds
- Carlsson et al. (IJCAI 2021): Thompson Sampling for Bandits with Clustered Arms
- Zhao et al. (arXiv:2602.15972): Hierarchical Unimodal Thompson Sampling
- CADTS (arXiv:2410.04217): Adaptive Discounted TS for portfolios
- 167-paper meta-analysis (arXiv:2512.10913): implementation > algorithm
"""

from .types import AgentBeta, TradeRecord
from .constants import (
    SIGNAL_GROUPS, GROUP_NAMES, ALL_SIGNALS, SIGNAL_TO_GROUP, META_PARAMS,
    _GLOBAL_REGIME, _MIN_REGIME_TRADES,
)
from .legacy import OnlineLearner
from .hierarchical import HierarchicalOnlineLearner

__all__ = [
    "AgentBeta", "TradeRecord",
    "OnlineLearner", "HierarchicalOnlineLearner",
    "SIGNAL_GROUPS", "GROUP_NAMES", "ALL_SIGNALS", "SIGNAL_TO_GROUP", "META_PARAMS",
    "_GLOBAL_REGIME", "_MIN_REGIME_TRADES",
]
