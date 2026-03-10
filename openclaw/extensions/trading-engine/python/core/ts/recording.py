"""Trade recording mixins for Hierarchical Thompson Sampling."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .types import AgentBeta, TradeRecord
from .constants import (
    SIGNAL_GROUPS, ALL_SIGNALS, SIGNAL_TO_GROUP,
    _GLOBAL_REGIME,
)

logger = logging.getLogger("trading-engine.online_learner")


class RecordingMixin:
    """Record trade outcomes and update posteriors (Levels 1 & 2).

    Handles: real trades, counterfactuals, correct holds, exit regrets.
    """

    def record_trade(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        held_hours: float,
        agent_signals: Dict[str, float],
        market_type: str = "crypto",
        regime: str = "unknown",
        position_pct_used: float = 0.0,
        confidence_at_entry: float = 0.0,
        position_side: str = "long",
        ta_snapshot: Optional[Dict[str, float]] = None,
        mfe: float = 0.0,
        mae: float = 0.0,
        capture_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """Record a completed trade and update group + signal + meta posteriors.

        Trades held < 5 minutes are recorded but DO NOT update posteriors.
        """
        now = time.time()
        record = TradeRecord(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            held_hours=held_hours,
            agent_signals=agent_signals,
            entry_time=now - held_hours * 3600,
            exit_time=now,
            market_type=market_type,
            regime=regime,
            position_side=position_side,
        )
        self._trades.append(record)
        self._total_pnl += pnl_pct

        # Skip posterior update for trades held < 5 minutes (artifacts)
        if held_hours < 5.0 / 60.0:
            logger.info("[h-ts] Trade #%d: %s skipped posterior update (held %.1f min < 5 min threshold)",
                        len(self._trades), ticker, held_hours * 60)
            if self._save_path:
                self.save()
            return {"trade_number": len(self._trades), "ticker": ticker, "pnl_pct": pnl_pct,
                    "skipped": True, "reason": "held < 5 minutes"}

        if len(self._trades) > self._max_window:
            self._trades = self._trades[-self._max_window:]

        # Determine target regimes to update
        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)
            self._regime_trade_counts[regime] = self._regime_trade_counts.get(regime, 0) + 1
        self._regime_trade_counts[_GLOBAL_REGIME] = self._regime_trade_counts.get(_GLOBAL_REGIME, 0) + 1

        # Adaptive discount
        adaptive_disc = self._get_adaptive_discount(regime)
        self._update_regime_streak(regime)

        profitable = pnl_pct > 0
        is_short = position_side == "short"
        updates = {}
        active_groups: Dict[str, List[float]] = {}

        # Level 2: update individual signal Betas
        for sig_name, signal_value in agent_signals.items():
            if abs(signal_value) < 0.05:
                continue

            group = SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                continue

            signal_bullish = signal_value > 0
            if is_short:
                aligned = (not signal_bullish and profitable) or (signal_bullish and not profitable)
            else:
                aligned = (signal_bullish and profitable) or (not signal_bullish and not profitable)
            effective_pnl = abs(pnl_pct) if aligned else -abs(pnl_pct)

            for r in target_regimes:
                sig_beta = self._signal_betas[r][group].get(sig_name)
                if sig_beta is None:
                    sig_beta = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sig_beta
                sig_beta.update(pnl_pct=effective_pnl, discount=self._signal_discount)

            if group not in active_groups:
                active_groups[group] = [effective_pnl]
            else:
                active_groups[group].append(effective_pnl)

            updates[sig_name] = {
                "group": group,
                "signal": round(signal_value, 3),
                "aligned": aligned,
                "new_mean": round(
                    self._signal_betas[_GLOBAL_REGIME][group][sig_name].mean, 4
                ),
            }

        # TA-based independent learning (arXiv:2402.10289)
        _VIRTUAL_WEIGHT = 0.5
        _VIRTUAL_MIN_STRENGTH = 0.15
        virtual_count = 0
        if ta_snapshot:
            mentioned = set(agent_signals.keys())
            for sig_name, ta_value in ta_snapshot.items():
                if sig_name in mentioned:
                    continue
                if abs(ta_value) < _VIRTUAL_MIN_STRENGTH:
                    continue
                group = SIGNAL_TO_GROUP.get(sig_name)
                if group is None:
                    continue
                sig_bullish = ta_value > 0
                if is_short:
                    aligned = (not sig_bullish and profitable) or (sig_bullish and not profitable)
                else:
                    aligned = (sig_bullish and profitable) or (not sig_bullish and not profitable)
                virtual_pnl = (abs(pnl_pct) if aligned else -abs(pnl_pct)) * _VIRTUAL_WEIGHT
                for r in target_regimes:
                    sb = self._signal_betas[r][group].get(sig_name)
                    if sb is None:
                        sb = AgentBeta(name=sig_name)
                        self._signal_betas[r][group][sig_name] = sb
                    sb.update(pnl_pct=virtual_pnl, discount=0.999, count_trade=False)
                virtual_count += 1

        # Level 1: update group Betas (adaptive discount)
        for group_name, pnl_list in active_groups.items():
            avg_pnl = sum(pnl_list) / len(pnl_list)
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=avg_pnl, discount=adaptive_disc
                )

        # Level 0: update meta-parameter Betas
        meta_updates = self._update_meta_params(
            pnl_pct=pnl_pct,
            held_hours=held_hours,
            position_pct_used=position_pct_used,
            confidence_at_entry=confidence_at_entry,
            agent_signals=agent_signals,
            target_regimes=target_regimes,
            discount=adaptive_disc,
            mfe=mfe,
            mae=mae,
            capture_ratio=capture_ratio,
        )

        logger.info(
            "[h-ts] Trade #%d: %s pnl=%+.2f%% regime=%s discount=%.3f(streak=%d) groups=%s signals=%s virtual=%d meta=%s",
            self.total_trades, ticker, pnl_pct * 100, regime,
            adaptive_disc, self._regime_streak,
            list(active_groups.keys()),
            {k: v["aligned"] for k, v in updates.items()},
            virtual_count,
            {k: f"{v:+.3f}" for k, v in meta_updates.items()} if meta_updates else "N/A",
        )

        if self._save_path and not getattr(self, '_reprocessing', False):
            self.save()

        return {
            "trade_number": self.total_trades,
            "ticker": ticker,
            "pnl_pct": pnl_pct,
            "profitable": profitable,
            "regime": regime,
            "active_groups": list(active_groups.keys()),
            "signal_updates": updates,
            "cumulative_pnl": self._total_pnl,
        }

    def record_counterfactual(
        self,
        ticker: str,
        price_at_hold: float,
        price_now: float,
        ta_signals: Dict[str, float],
        regime: str = "unknown",
        discount_factor: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """Record a phantom trade from a missed opportunity."""
        if price_at_hold <= 0:
            return None

        raw_pnl = (price_now - price_at_hold) / price_at_hold
        if abs(raw_pnl) < 0.003:
            return None

        effective_pnl = raw_pnl * discount_factor

        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)

        adaptive_disc = self._get_adaptive_discount(regime)

        updates = {}
        active_groups: Dict[str, List[float]] = {}

        for sig_name, signal_value in ta_signals.items():
            if abs(signal_value) < 0.05:
                continue
            group = SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                continue

            signal_bullish = signal_value > 0
            price_went_up = raw_pnl > 0
            aligned = (signal_bullish and price_went_up) or (not signal_bullish and not price_went_up)
            sig_effective = abs(effective_pnl) if aligned else -abs(effective_pnl)

            for r in target_regimes:
                sb = self._signal_betas[r][group].get(sig_name)
                if sb is None:
                    sb = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sb
                sb.update(pnl_pct=sig_effective, discount=self._signal_discount,
                          count_trade=False)

            if group not in active_groups:
                active_groups[group] = [sig_effective]
            else:
                active_groups[group].append(sig_effective)

            updates[sig_name] = {
                "group": group, "signal": round(signal_value, 3),
                "aligned": aligned,
            }

        for group_name, pnl_list in active_groups.items():
            avg_pnl = sum(pnl_list) / len(pnl_list)
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=avg_pnl, discount=adaptive_disc
                )

        if updates:
            logger.info(
                "[h-ts] Counterfactual %s: raw_pnl=%+.2f%% effective=%+.2f%% regime=%s signals=%s",
                ticker, raw_pnl * 100, effective_pnl * 100, regime,
                {k: v["aligned"] for k, v in updates.items()},
            )
            if self._save_path:
                self.save()

        return {
            "ticker": ticker,
            "raw_pnl": raw_pnl,
            "effective_pnl": effective_pnl,
            "regime": regime,
            "signal_updates": updates,
        } if updates else None

    def record_correct_hold(
        self,
        ticker: str,
        price_at_hold: float,
        price_now: float,
        ta_signals: Dict[str, float],
        regime: str = "unknown",
        discount_factor: float = 0.2,
    ) -> Optional[Dict[str, Any]]:
        """Record a validated HOLD -- price dropped after we decided not to buy."""
        if price_at_hold <= 0:
            return None

        raw_pnl = (price_now - price_at_hold) / price_at_hold
        if raw_pnl >= -0.003:
            return None

        avoided_loss = abs(raw_pnl)
        effective_reward = avoided_loss * discount_factor

        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)

        adaptive_disc = self._get_adaptive_discount(regime)
        updates = {}
        active_groups: Dict[str, float] = {}

        for sig_name, signal_value in ta_signals.items():
            if abs(signal_value) < 0.05:
                continue
            group = SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                continue

            signal_bearish = signal_value < 0
            price_dropped = True
            aligned = signal_bearish and price_dropped
            sig_effective = effective_reward if aligned else -effective_reward

            for r in target_regimes:
                sb = self._signal_betas[r][group].get(sig_name)
                if sb is None:
                    sb = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sb
                sb.update(pnl_pct=sig_effective, discount=self._signal_discount,
                          count_trade=False)

            if group not in active_groups:
                active_groups[group] = sig_effective

            updates[sig_name] = {
                "group": group, "signal": round(signal_value, 3),
                "aligned": aligned,
            }

        for group_name, group_pnl in active_groups.items():
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=group_pnl, discount=adaptive_disc
                )

        # Meta-param: reward risk_aversion (correctly cautious)
        meta_reward = min(0.03, effective_reward)
        for r in target_regimes:
            ra = self._meta_betas.get(r, {}).get("risk_aversion")
            if ra:
                ra.update(pnl_pct=meta_reward, discount=adaptive_disc,
                          count_trade=False)

        if updates:
            logger.info(
                "[h-ts] Correct HOLD %s: avoided=%+.2f%% reward=%+.2f%% regime=%s signals=%s",
                ticker, raw_pnl * 100, effective_reward * 100, regime,
                {k: v["aligned"] for k, v in updates.items()},
            )
            if self._save_path:
                self.save()

        return {
            "ticker": ticker,
            "raw_pnl": raw_pnl,
            "avoided_loss": avoided_loss,
            "effective_reward": effective_reward,
            "regime": regime,
            "signal_updates": updates,
        } if updates else None

    def record_exit_regret(
        self,
        ticker: str,
        exit_price: float,
        price_now: float,
        pnl_at_exit: float,
        held_hours: float,
        agent_signals: Dict[str, float],
        regime: str = "unknown",
        position_side: str = "long",
        discount_factor: float = 0.15,
        was_premature: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Post-exit counterfactual: learn from price movement after closing."""
        if exit_price <= 0:
            return None

        if position_side == "short":
            additional_pnl = (exit_price - price_now) / exit_price
        else:
            additional_pnl = (price_now - exit_price) / exit_price

        if abs(additional_pnl) < 0.003:
            return None

        effective_pnl = abs(additional_pnl) * discount_factor

        target_regimes = [_GLOBAL_REGIME]
        if regime and regime != "unknown":
            self._ensure_regime(regime)
            target_regimes.append(regime)

        adaptive_disc = self._get_adaptive_discount(regime)
        updates = {}
        active_groups: Dict[str, List[float]] = {}

        # Level 2: Signal updates
        for sig_name, signal_value in agent_signals.items():
            if abs(signal_value) < 0.05:
                continue
            group = SIGNAL_TO_GROUP.get(sig_name)
            if group is None:
                continue

            signal_bullish = signal_value > 0
            is_long = position_side != "short"

            if was_premature:
                signal_supported_hold = (is_long and signal_bullish) or (not is_long and not signal_bullish)
                sig_effective = effective_pnl if signal_supported_hold else -effective_pnl
            else:
                signal_warned_exit = (is_long and not signal_bullish) or (not is_long and signal_bullish)
                sig_effective = effective_pnl if signal_warned_exit else -effective_pnl

            for r in target_regimes:
                sb = self._signal_betas[r][group].get(sig_name)
                if sb is None:
                    sb = AgentBeta(name=sig_name)
                    self._signal_betas[r][group][sig_name] = sb
                sb.update(pnl_pct=sig_effective, discount=self._signal_discount,
                          count_trade=False)

            if group not in active_groups:
                active_groups[group] = [sig_effective]
            else:
                active_groups[group].append(sig_effective)

            updates[sig_name] = {
                "group": group, "signal": round(signal_value, 3),
                "supported_hold" if was_premature else "warned_exit":
                    signal_supported_hold if was_premature else signal_warned_exit,
            }

        # Level 1: Group updates
        for group_name, pnl_list in active_groups.items():
            avg_pnl = sum(pnl_list) / len(pnl_list)
            for r in target_regimes:
                self._group_betas[r][group_name].update(
                    pnl_pct=avg_pnl, discount=adaptive_disc
                )

        # Level 0: Meta-parameter updates
        meta_reward = min(0.03, effective_pnl)
        for r in target_regimes:
            if was_premature:
                patience_scale = min(2.0, max(0.5, 1.0 / max(held_hours, 0.1)))
                patience_eff = min(0.03, meta_reward * patience_scale)
                hp = self._meta_betas.get(r, {}).get("hold_patience")
                if hp:
                    hp.update(pnl_pct=patience_eff, discount=adaptive_disc,
                              count_trade=False)
                ra = self._meta_betas.get(r, {}).get("risk_aversion")
                if ra:
                    ra.update(pnl_pct=-meta_reward, discount=adaptive_disc,
                              count_trade=False)
            else:
                ra = self._meta_betas.get(r, {}).get("risk_aversion")
                if ra:
                    ra.update(pnl_pct=meta_reward, discount=adaptive_disc,
                              count_trade=False)

        label = "PREMATURE" if was_premature else "VALIDATED"
        logger.info(
            "[h-ts] Exit regret %s %s: additional=%+.2f%% effective=%+.2f%% regime=%s signals=%d",
            label, ticker, additional_pnl * 100, effective_pnl * 100, regime, len(updates),
        )
        if self._save_path:
            self.save()

        return {
            "ticker": ticker,
            "type": label.lower(),
            "additional_pnl": additional_pnl,
            "effective_pnl": effective_pnl,
            "regime": regime,
            "signal_updates": updates,
        }
