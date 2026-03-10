"""Beta distribution and trade record types for Thompson Sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentBeta:
    """Beta distribution for a single agent's signal quality.

    alpha = successes + prior, beta = failures + prior.
    Mean = alpha / (alpha + beta).

    Supports Predictive Sampling (arXiv:2205.01970): unstable signals
    are pulled toward the uninformative prior (0.5), while stable ones
    are sampled normally via Thompson Sampling.
    """
    name: str
    alpha: float = 2.0    # prior: 2 successes (mildly optimistic)
    beta: float = 2.0     # prior: 2 failures
    total_trades: int = 0
    _reward_history: List[float] = field(default_factory=list, repr=False)

    _PS_WINDOW = 10  # class constant — NOT a dataclass field

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def stability_score(self) -> float:
        """Reward variance-based stability: 0.0 (unstable) to 1.0 (stable).

        Based on arXiv:2205.01970 — Predictive Sampling considers
        information "durability". High reward variance means the signal
        is unreliable across recent trades.

        Normalized: var / 0.12 (max variance of uniform [0,1] is 1/12 ≈ 0.083,
        but real rewards cluster near 0.05-0.95, so 0.12 gives good spread).
        """
        if len(self._reward_history) < 3:
            return 0.5  # insufficient data → neutral (pure TS)
        n = len(self._reward_history)
        mean_r = sum(self._reward_history) / n
        var_r = sum((r - mean_r) ** 2 for r in self._reward_history) / n
        # Invert: high variance → low stability
        return max(0.0, min(1.0, 1.0 - var_r / 0.12))

    def sample(self, use_ps: bool = False) -> float:
        """Sample from Beta(alpha, beta).

        If use_ps=True, applies Predictive Sampling (arXiv:2205.01970):
            result = s * ts_sample + (1 - s) * 0.5
        where s = stability_score. Unstable signals regress toward 0.5.
        """
        a = max(1e-6, self.alpha)
        b = max(1e-6, self.beta)
        ts_sample = random.betavariate(a, b)
        if not use_ps:
            return ts_sample
        s = self.stability_score
        return s * ts_sample + (1.0 - s) * 0.5

    def update(self, pnl_pct: float = 0.0, discount: float = 0.995,
               count_trade: bool = True) -> None:
        """Update posterior with discounted Thompson Sampling.

        - Decay existing beliefs (prevents stale priors dominating)
        - PAR sigmoid reward: minimizes policy gradient variance (Theorem 3.2)
        - count_trade=False for virtual/indirect updates (don't inflate trade count)
        - Tracks reward history for Predictive Sampling stability estimation

        References:
            arXiv:2502.18770 — PAR: bounded sigmoid reward minimizes variance
            arXiv:2305.10718 — optimal discount for non-stationary bandits
            arXiv:2205.01970 — Predictive Sampling for non-stationary bandits
        """
        # Discount existing beliefs (ADTS: proportional decay)
        self.alpha = max(1.0, self.alpha * discount)
        self.beta = max(1.0, self.beta * discount)

        # Asymmetric piecewise-linear reward (loss slope 1.5x win slope)
        # Win:  +1% → 0.90,  Loss: -1% → 0.05 (clamped)
        # Effect: H-TS favors high R:R parameter combos
        # Ref: arXiv:2507.19639 — Asymmetric Reward Shaping
        pnl_scaled = pnl_pct * 100
        if pnl_scaled >= 0:
            reward = 0.50 + pnl_scaled * 0.40   # win: +1% → 0.90
        else:
            reward = 0.50 + pnl_scaled * 0.60   # loss: -1% → -0.10 → clamped 0.05
        reward = max(0.05, min(0.95, reward))
        self.alpha += reward
        self.beta += (1.0 - reward)
        if count_trade:
            self.total_trades += 1

        # Track reward for PS stability estimation
        self._reward_history.append(reward)
        if len(self._reward_history) > self._PS_WINDOW:
            self._reward_history = self._reward_history[-self._PS_WINDOW:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "total_trades": self.total_trades,
            "reward_history": list(self._reward_history),
        }


@dataclass
class TradeRecord:
    """Record of a completed trade for learning."""
    ticker: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    held_hours: float
    agent_signals: Dict[str, float]  # {agent_name: signal_value} at entry
    entry_time: float
    exit_time: float
    market_type: str = "crypto"
    regime: str = "unknown"  # regime at trade entry (for regime-aware learning)
    position_side: str = "long"  # "long" or "short"
