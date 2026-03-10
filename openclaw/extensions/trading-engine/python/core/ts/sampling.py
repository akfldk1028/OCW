"""Weight sampling for Hierarchical Thompson Sampling."""

from __future__ import annotations

from typing import Dict

from .constants import SIGNAL_GROUPS, GROUP_NAMES, ALL_SIGNALS, _GLOBAL_REGIME


class SamplingMixin:
    """2-level Thompson Sampling weight generation and accessors."""

    def sample_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Sample weights via 2-level hierarchical Thompson Sampling.

        1. Sample from Level 1 (group Betas)
        2. Sample from Level 2 (signal Betas within each group)
        3. Final weight = group_weight x within_group_weight (normalized)
        """
        if not self.has_enough_data:
            return self._default_weights()

        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        signal_betas = self._signal_betas[eff_regime]

        group_samples = {g: gb.sample(use_ps=True) for g, gb in group_betas.items()}
        total_group = sum(group_samples.values())
        if total_group <= 0:
            return self._default_weights()

        final_weights: Dict[str, float] = {}
        for group_name, sigs in SIGNAL_GROUPS.items():
            group_w = group_samples[group_name] / total_group

            sig_samples = {}
            for sig_name in sigs:
                sb = signal_betas[group_name].get(sig_name)
                if sb:
                    sig_samples[sig_name] = sb.sample(use_ps=True)
                else:
                    sig_samples[sig_name] = 0.5

            total_sig = sum(sig_samples.values())
            if total_sig <= 0:
                total_sig = 1.0

            for sig_name, sv in sig_samples.items():
                final_weights[sig_name] = group_w * (sv / total_sig)

        return final_weights

    def get_mean_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Deterministic mean weights (no sampling noise). For display."""
        if not self.has_enough_data:
            return self._default_weights()

        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        signal_betas = self._signal_betas[eff_regime]

        group_means = {g: gb.mean for g, gb in group_betas.items()}
        total_group = sum(group_means.values())
        if total_group <= 0:
            return self._default_weights()

        final: Dict[str, float] = {}
        for group_name, sigs in SIGNAL_GROUPS.items():
            group_w = group_means[group_name] / total_group

            sig_means = {}
            for sig_name in sigs:
                sb = signal_betas[group_name].get(sig_name)
                sig_means[sig_name] = sb.mean if sb else 0.5

            total_sig = sum(sig_means.values())
            if total_sig <= 0:
                total_sig = 1.0

            for sig_name, sv in sig_means.items():
                final[sig_name] = group_w * (sv / total_sig)

        return final

    def get_signal_reliability(self, regime: str = "unknown") -> Dict[str, float]:
        """Raw Beta means for each signal (0.0-1.0). NOT normalized."""
        eff_regime = self._effective_regime(regime)
        signal_betas = self._signal_betas[eff_regime]

        result: Dict[str, float] = {}
        for group_name, sigs in SIGNAL_GROUPS.items():
            for sig_name in sigs:
                sb = signal_betas[group_name].get(sig_name)
                result[sig_name] = sb.mean if sb else 0.5
        return result

    def get_group_weights(self, regime: str = "unknown") -> Dict[str, float]:
        """Get group-level mean weights only (for dashboard summary)."""
        eff_regime = self._effective_regime(regime)
        group_betas = self._group_betas[eff_regime]
        raw = {g: gb.mean for g, gb in group_betas.items()}
        total = sum(raw.values())
        if total <= 0:
            return {g: 1.0 / len(GROUP_NAMES) for g in GROUP_NAMES}
        return {g: v / total for g, v in raw.items()}

    @staticmethod
    def _default_weights() -> Dict[str, float]:
        """Uniform weights when no data yet."""
        n = len(ALL_SIGNALS)
        return {s: 1.0 / n for s in ALL_SIGNALS}
