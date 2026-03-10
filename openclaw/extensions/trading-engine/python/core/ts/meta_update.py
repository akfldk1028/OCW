"""Meta-parameter update logic for Hierarchical Thompson Sampling (Level 0)."""

from __future__ import annotations

from typing import Dict, List, Optional

from .constants import SIGNAL_TO_GROUP, _GLOBAL_REGIME, META_PARAMS


class MetaUpdateMixin:
    """Level 0 meta-parameter learning and access."""

    def _update_meta_params(
        self,
        pnl_pct: float,
        held_hours: float,
        position_pct_used: float,
        confidence_at_entry: float,
        agent_signals: Dict[str, float],
        target_regimes: List[str],
        discount: Optional[float] = None,
        mfe: float = 0.0,
        mae: float = 0.0,
        capture_ratio: float = 0.0,
    ) -> Dict[str, float]:
        """Update Level 0 meta-parameter Betas based on trade outcome.

        Each meta-param has custom reward shaping capturing its semantics.
        Returns dict of {param: effective_pnl} for logging.
        """
        if not target_regimes:
            return {}

        meta_updates: Dict[str, float] = {}

        # 1. position_scale
        pos_scale = min(position_pct_used / 0.10, 3.0) if position_pct_used > 0 else 1.0
        eff_position = pnl_pct * pos_scale
        meta_updates["position_scale"] = eff_position

        # 2. entry_selectivity
        conf_scale = confidence_at_entry if confidence_at_entry > 0 else 0.5
        eff_selectivity = pnl_pct * conf_scale
        meta_updates["entry_selectivity"] = eff_selectivity

        # 3. hold_patience: ASYMMETRIC — break the early-exit vicious cycle
        _MIN_PATIENCE_HOLD = 0.25  # hours (15 min)
        if pnl_pct >= 0:
            hold_scale = max(min(held_hours / 0.5, 2.0), 0.3)
            eff_patience = pnl_pct * hold_scale
        elif held_hours >= _MIN_PATIENCE_HOLD:
            hold_scale = min(held_hours / 0.5, 2.0)
            eff_patience = pnl_pct * hold_scale
        else:
            eff_patience = 0.0
        meta_updates["hold_patience"] = eff_patience

        # 4. trade_frequency: pure outcome
        meta_updates["trade_frequency"] = pnl_pct

        # 5. trend_vs_reversion
        trend_groups = {"technical_trend"}
        reversion_groups = {"technical_reversion"}
        trend_weight = 0.0
        reversion_weight = 0.0
        for sig_name, sig_val in agent_signals.items():
            group = SIGNAL_TO_GROUP.get(sig_name)
            if group in trend_groups and abs(sig_val) >= 0.05:
                trend_weight += abs(sig_val)
            elif group in reversion_groups and abs(sig_val) >= 0.05:
                reversion_weight += abs(sig_val)
        total_tr_weight = trend_weight + reversion_weight
        if total_tr_weight > 0:
            direction = (trend_weight - reversion_weight) / total_tr_weight
            eff_trend = pnl_pct * direction
        else:
            eff_trend = 0.0
        meta_updates["trend_vs_reversion"] = eff_trend

        # 6. risk_aversion: sign-flipped
        eff_risk = -pnl_pct * pos_scale
        meta_updates["risk_aversion"] = eff_risk

        # 7. profit_target_width
        if pnl_pct > 0 and mfe > 0:
            if capture_ratio >= 0.30:
                eff_tp_width = pnl_pct
            else:
                eff_tp_width = -pnl_pct * 0.5
        elif pnl_pct <= 0 and mfe >= 0.002:
            eff_tp_width = pnl_pct
        else:
            eff_tp_width = 0.0
        meta_updates["profit_target_width"] = eff_tp_width

        # 8. loss_tolerance
        _DEEP_MAE_THRESHOLD = -0.003
        if pnl_pct > 0 and mae < _DEEP_MAE_THRESHOLD:
            eff_loss_tol = abs(pnl_pct) * 1.5
        elif pnl_pct <= 0 and mae < _DEEP_MAE_THRESHOLD:
            eff_loss_tol = pnl_pct
        else:
            eff_loss_tol = 0.0
        meta_updates["loss_tolerance"] = eff_loss_tol

        # Apply updates (clamped to +/-0.03)
        disc = discount if discount is not None else self._group_discount
        for param, eff_pnl in list(meta_updates.items()):
            eff_pnl = max(-0.03, min(0.03, eff_pnl))
            meta_updates[param] = eff_pnl
            if abs(eff_pnl) < 1e-8:
                continue
            for r in target_regimes:
                mb = self._meta_betas.get(r, {}).get(param)
                if mb:
                    mb.update(pnl_pct=eff_pnl, discount=disc)

        # Soft regularization anchors (arXiv:2602.06014)
        _META_ANCHORS = {"hold_patience": (0.50, 0.02), "trade_frequency": (0.40, 0.02)}
        for param, (anchor_mean, pull_strength) in _META_ANCHORS.items():
            for r in target_regimes:
                mb = self._meta_betas.get(r, {}).get(param)
                if mb:
                    mb.alpha += pull_strength * anchor_mean
                    mb.beta += pull_strength * (1.0 - anchor_mean)

        return meta_updates

    # --- Meta-parameter accessors ---

    def get_meta_param_means(self, regime: str = "unknown") -> Dict[str, float]:
        """Get deterministic meta-parameter means (no sampling noise)."""
        eff_regime = self._effective_regime(regime)
        meta = self._meta_betas.get(eff_regime, self._meta_betas[_GLOBAL_REGIME])
        return {p: round(mb.mean, 4) for p, mb in meta.items()}

    def get_meta_param_betas(self, regime: str = "unknown") -> Dict[str, Dict]:
        """Get full alpha/beta for meta-parameters."""
        eff_regime = self._effective_regime(regime)
        meta = self._meta_betas.get(eff_regime, self._meta_betas[_GLOBAL_REGIME])
        return {p: mb.to_dict() for p, mb in meta.items()}

    def get_meta_param_betas_global(self) -> Dict[str, Dict]:
        """Get full alpha/beta for global meta-parameters."""
        meta = self._meta_betas.get(_GLOBAL_REGIME, {})
        return {p: mb.to_dict() for p, mb in meta.items()}

    def sample_meta_params(self, regime: str = "unknown") -> Dict[str, float]:
        """Thompson Sample meta-parameters with Predictive Sampling."""
        eff_regime = self._effective_regime(regime)
        meta = self._meta_betas.get(eff_regime, self._meta_betas[_GLOBAL_REGIME])
        return {p: round(mb.sample(use_ps=True), 4) for p, mb in meta.items()}
