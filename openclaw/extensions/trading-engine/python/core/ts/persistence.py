"""Save/load/migration for Hierarchical Thompson Sampling state."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List

from .types import AgentBeta, TradeRecord
from .constants import (
    SIGNAL_GROUPS, GROUP_NAMES, ALL_SIGNALS, SIGNAL_TO_GROUP,
    META_PARAMS, _GLOBAL_REGIME,
)

logger = logging.getLogger("trading-engine.online_learner")

# Mapping from old flat 8-arm signal names to new hierarchy
_OLD_TO_NEW_MAPPING: Dict[str, List[str]] = {
    "market": ["volume_spike", "vwap_deviation"],
    "quant": ["rsi_signal", "bb_deviation", "support_resistance"],
    "sentiment": ["news_sentiment", "fear_greed"],
    "momentum": ["ema_cross_fast", "trend_strength"],
    "regime": ["market_regime", "volatility_regime"],
    "funding_rate": ["funding_rate"],
    "oi_signal": ["oi_change", "long_short_ratio", "cvd_signal"],
    "macro": ["btc_dominance", "dxy_direction"],
}


def _validate_alpha_beta(a: float, b: float) -> tuple:
    """Ensure alpha/beta are finite and positive."""
    if not (math.isfinite(a) and a > 0):
        a = 2.0
    if not (math.isfinite(b) and b > 0):
        b = 2.0
    return a, b


class PersistenceMixin:
    """Save, load, migrate, and reprocess H-TS state."""

    def save(self) -> None:
        if self._save_path is None:
            return

        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 4,
            "meta_betas": {
                regime: {p: mb.to_dict() for p, mb in params.items()}
                for regime, params in self._meta_betas.items()
            },
            "group_betas": {
                regime: {g: gb.to_dict() for g, gb in groups.items()}
                for regime, groups in self._group_betas.items()
            },
            "signal_betas": {
                regime: {
                    g: {s: sb.to_dict() for s, sb in sigs.items()}
                    for g, sigs in groups.items()
                }
                for regime, groups in self._signal_betas.items()
            },
            "regime_trade_counts": dict(self._regime_trade_counts),
            "adaptive_discount": {
                "last_regime": self._last_regime,
                "regime_streak": self._regime_streak,
            },
            "total_trades": self.total_trades,
            "total_pnl": self._total_pnl,
            "last_trades": [
                {
                    "ticker": t.ticker,
                    "pnl_pct": t.pnl_pct,
                    "held_hours": t.held_hours,
                    "agent_signals": t.agent_signals,
                    "market_type": t.market_type,
                    "exit_time": t.exit_time,
                    "regime": t.regime,
                    "position_side": getattr(t, "position_side", "long"),
                }
                for t in self._trades[-100:]
            ],
        }
        tmp = self._save_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._save_path)
        logger.debug("[h-ts] Saved state (%d trades, %d regimes, %d signals)",
                     self.total_trades, len(self._group_betas) - 1,
                     len(ALL_SIGNALS))

    def load(self) -> bool:
        if self._save_path is None or not self._save_path.exists():
            return False

        try:
            state = json.loads(self._save_path.read_text())
            version = state.get("version", 1)

            if version < 3:
                return self._migrate_from_v1(state)

            # Load v3/v4 hierarchical state
            for regime, params in state.get("meta_betas", {}).items():
                if regime != _GLOBAL_REGIME:
                    self._ensure_regime(regime)
                for p, data in params.items():
                    if p in self._meta_betas.get(regime, {}):
                        a, b = _validate_alpha_beta(
                            data.get("alpha", 2.0), data.get("beta", 2.0))
                        self._meta_betas[regime][p].alpha = a
                        self._meta_betas[regime][p].beta = b
                        self._meta_betas[regime][p].total_trades = data.get("total_trades", 0)
                        self._meta_betas[regime][p]._reward_history = data.get("reward_history", [])

            for regime, groups in state.get("group_betas", {}).items():
                if regime not in self._group_betas and regime != _GLOBAL_REGIME:
                    self._ensure_regime(regime)
                for g, data in groups.items():
                    if g in self._group_betas.get(regime, {}):
                        a, b = _validate_alpha_beta(
                            data.get("alpha", 2.0), data.get("beta", 2.0))
                        self._group_betas[regime][g].alpha = a
                        self._group_betas[regime][g].beta = b
                        self._group_betas[regime][g].total_trades = data.get("total_trades", 0)
                        self._group_betas[regime][g]._reward_history = data.get("reward_history", [])

            for regime, groups in state.get("signal_betas", {}).items():
                if regime not in self._signal_betas and regime != _GLOBAL_REGIME:
                    self._ensure_regime(regime)
                for g, sigs in groups.items():
                    if g not in self._signal_betas.get(regime, {}):
                        continue
                    for s, data in sigs.items():
                        if s not in self._signal_betas[regime][g]:
                            self._signal_betas[regime][g][s] = AgentBeta(name=s)
                        sb = self._signal_betas[regime][g][s]
                        a, b = _validate_alpha_beta(
                            data.get("alpha", 2.0), data.get("beta", 2.0))
                        sb.alpha = a
                        sb.beta = b
                        sb.total_trades = data.get("total_trades", 0)
                        sb._reward_history = data.get("reward_history", [])

            self._regime_trade_counts = state.get("regime_trade_counts", {_GLOBAL_REGIME: 0})
            self._total_pnl = state.get("total_pnl", 0.0)

            ad_state = state.get("adaptive_discount", {})
            self._last_regime = ad_state.get("last_regime")
            self._regime_streak = ad_state.get("regime_streak", 0)

            self._trades = []
            for t in state.get("last_trades", []):
                self._trades.append(TradeRecord(
                    ticker=t["ticker"],
                    entry_price=0.0,
                    exit_price=0.0,
                    pnl_pct=t["pnl_pct"],
                    held_hours=t.get("held_hours", 0.0),
                    agent_signals=t.get("agent_signals", {}),
                    entry_time=0.0,
                    exit_time=t.get("exit_time", 0.0),
                    market_type=t.get("market_type", "crypto"),
                    regime=t.get("regime", "unknown"),
                    position_side=t.get("position_side", "long"),
                ))

            logger.info(
                "[h-ts] Loaded v3 state: %d trades, %d regimes, %d groups, pnl=%+.2f%%",
                self.total_trades, len(self._group_betas) - 1,
                len(GROUP_NAMES), self._total_pnl * 100,
            )

            # v3->v4 migration: bootstrap meta betas from existing trades
            if "meta_betas" not in state and self._trades:
                logger.info(
                    "[h-ts] v3->v4: bootstrapping meta-parameters from %d existing trades...",
                    len(self._trades),
                )
                self.reprocess_all_trades()
            else:
                short_trades = sum(1 for t in self._trades if t.held_hours < 5.0 / 60.0 and abs(t.pnl_pct) > 1e-8)
                if short_trades > 0:
                    logger.warning(
                        "[h-ts] Found %d trades held < 5 min (position-restore artifacts). "
                        "Reprocessing to exclude them from posteriors.",
                        short_trades,
                    )
                    self.reprocess_all_trades()

            return True

        except Exception as exc:
            logger.warning("[h-ts] Failed to load state: %s", exc)
            return False

    def _migrate_from_v1(self, state: Dict) -> bool:
        """Migrate old flat 8-arm OnlineLearner state to hierarchical v3."""
        logger.info("[h-ts] Migrating from v1/v2 flat state to v3 hierarchical...")

        self._total_pnl = state.get("total_pnl", 0.0)
        self._regime_trade_counts = state.get("regime_trade_counts", {_GLOBAL_REGIME: 0})

        self._trades = []
        for t in state.get("last_trades", []):
            old_signals = t.get("agent_signals", {})
            new_signals = self._remap_old_signals(old_signals)

            self._trades.append(TradeRecord(
                ticker=t["ticker"],
                entry_price=0.0,
                exit_price=0.0,
                pnl_pct=t["pnl_pct"],
                held_hours=t.get("held_hours", 0.0),
                agent_signals=new_signals,
                entry_time=0.0,
                exit_time=t.get("exit_time", 0.0),
                market_type=t.get("market_type", "crypto"),
                regime=t.get("regime", "unknown"),
            ))

        old_trades = list(self._trades)
        self._trades = []
        self._total_pnl = 0.0
        self._regime_trade_counts = {_GLOBAL_REGIME: 0}

        for t in old_trades:
            if abs(t.pnl_pct) < 1e-8:
                continue
            self.record_trade(
                ticker=t.ticker,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pct=t.pnl_pct,
                held_hours=t.held_hours,
                agent_signals=t.agent_signals,
                market_type=t.market_type,
                regime=t.regime,
            )

        logger.info(
            "[h-ts] Migration complete: %d trades reprocessed with %d signal types",
            len(self._trades), len(ALL_SIGNALS),
        )
        return True

    @staticmethod
    def _remap_old_signals(old_signals: Dict[str, float]) -> Dict[str, float]:
        """Map old 8-category signal names to new granular names."""
        new_signals: Dict[str, float] = {}
        for old_name, value in old_signals.items():
            new_names = _OLD_TO_NEW_MAPPING.get(old_name)
            if new_names:
                for nn in new_names:
                    new_signals[nn] = value
            elif old_name in SIGNAL_TO_GROUP:
                new_signals[old_name] = value
        return new_signals

    def reprocess_all_trades(self) -> None:
        """Reset all posteriors and re-learn from trade history."""
        old_trades = list(self._trades)
        n = len(old_trades)

        self._meta_betas = {
            _GLOBAL_REGIME: {p: AgentBeta(name=p) for p in META_PARAMS},
        }
        self._group_betas = {
            _GLOBAL_REGIME: {g: AgentBeta(name=g) for g in GROUP_NAMES},
        }
        self._signal_betas = {
            _GLOBAL_REGIME: {
                g: {s: AgentBeta(name=s) for s in sigs}
                for g, sigs in SIGNAL_GROUPS.items()
            },
        }
        self._trades = []
        self._total_pnl = 0.0
        self._regime_trade_counts = {_GLOBAL_REGIME: 0}
        self._last_regime = None
        self._regime_streak = 0

        self._reprocessing = True
        try:
            for t in old_trades:
                if abs(t.pnl_pct) < 1e-8 or t.held_hours < 5.0 / 60.0:
                    continue
                self.record_trade(
                    ticker=t.ticker,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    pnl_pct=t.pnl_pct,
                    held_hours=t.held_hours,
                    agent_signals=t.agent_signals,
                    market_type=t.market_type,
                    regime=t.regime,
                    position_side=t.position_side,
                )
        finally:
            self._reprocessing = False

        if self._save_path:
            self.save()
        logger.info("[h-ts] Reprocessed %d trades -> %d valid.", n, len(self._trades))
