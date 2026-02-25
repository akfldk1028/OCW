"""PPO + A2C + SAC ensemble agent for portfolio trading.

Provides:
- ``TradingEnv`` -- a custom Gymnasium environment that simulates
  portfolio management over a feature array produced by
  ``DataProcessor``.
- ``EnsembleAgent`` -- trains and runs three sub-agents and combines
  their predictions using rolling-Sharpe-ratio-weighted averaging.

SAC (Soft Actor-Critic) replaces DDPG as the default off-policy
algorithm.  Its entropy regularization makes it consistently superior
for volatile crypto markets (arXiv:2511.20678).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise

from config import ENSEMBLE_CONFIG, MODELS_DIR, RISK_CONFIG, TRAIN_CONFIG

logger = logging.getLogger(__name__)


# ======================================================================
# Custom Gymnasium environment
# ======================================================================


class TradingEnv(gym.Env):
    """A multi-stock portfolio trading environment.

    Observation
    -----------
    A flat vector consisting of:
    - Per-ticker features (OHLCV + technical indicators + sentiment)
    - Current portfolio holdings (fraction of portfolio per ticker)
    - Current cash ratio

    Action
    ------
    Continuous in ``[-1, 1]`` per ticker.
    - Negative values => sell (magnitude = fraction of holding to sell)
    - Positive values => buy  (magnitude = fraction of cash to allocate)

    Reward
    ------
    Sharpe-ratio-inspired daily return reward:
    ``r_t = (portfolio_return - risk_free_rate) / max(vol, epsilon)``
    computed over a rolling window to encourage consistent gains.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_array: np.ndarray,
        *,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        sentiment_scores: Optional[np.ndarray] = None,
        reward_scaling: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        feature_array : np.ndarray
            Shape ``(num_days, num_tickers, num_features)`` -- normalised.
        initial_cash : float
            Starting portfolio cash.
        transaction_cost : float
            Proportional transaction cost per trade.
        sentiment_scores : np.ndarray or None
            Shape ``(num_days, num_tickers)``.  Appended to obs if given.
        reward_scaling : float
            Multiplicative scaling applied to the raw reward.
        """
        super().__init__()

        self.data = feature_array.astype(np.float32)
        self.num_days, self.num_tickers, self.num_features = self.data.shape
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        # Sentiment: default to zeros if not provided
        if sentiment_scores is not None:
            self.sentiment = sentiment_scores.astype(np.float32)
        else:
            self.sentiment = np.zeros((self.num_days, self.num_tickers), dtype=np.float32)

        # Total observation size:
        #   features * tickers + sentiment * tickers + holdings * tickers + 1 (cash ratio)
        obs_size = (self.num_features + 1) * self.num_tickers + self.num_tickers + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_tickers,), dtype=np.float32,
        )

        # Internal state -- set in reset()
        self.current_step: int = 0
        self.cash: float = initial_cash
        self.holdings: np.ndarray = np.zeros(self.num_tickers, dtype=np.float64)
        self.portfolio_values: List[float] = []

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_tickers, dtype=np.float64)
        self.portfolio_values = [self.initial_cash]
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)

        # Current close prices are at feature index 0 (first feature is close)
        # Note: data is z-score normalised so we keep a separate "raw close"
        # approximation for portfolio accounting.  For a normalised env we
        # track portfolio value in *normalised* space; absolute dollar values
        # are only meaningful in backtesting with raw data.
        prices = self._current_prices()

        prev_portfolio_value = self._portfolio_value(prices)

        # Execute trades
        self._execute_actions(action, prices)

        # Advance time
        self.current_step += 1
        done = self.current_step >= self.num_days - 1
        truncated = False

        new_prices = self._current_prices()
        new_portfolio_value = self._portfolio_value(new_prices)
        self.portfolio_values.append(new_portfolio_value)

        # Reward: risk-adjusted daily return
        reward = self._compute_reward(prev_portfolio_value, new_portfolio_value)

        obs = self._get_observation()
        info = {
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "step": self.current_step,
        }
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        day_features = self.data[self.current_step].flatten()  # (num_tickers * num_features,)
        sentiment = self.sentiment[self.current_step]          # (num_tickers,)
        portfolio_value = self._portfolio_value(self._current_prices())
        if portfolio_value > 0:
            holding_ratios = (self.holdings * self._current_prices()) / portfolio_value
            cash_ratio = np.array([self.cash / portfolio_value], dtype=np.float32)
        else:
            holding_ratios = np.zeros(self.num_tickers, dtype=np.float32)
            cash_ratio = np.array([1.0], dtype=np.float32)

        obs = np.concatenate([
            day_features,
            sentiment,
            holding_ratios.astype(np.float32),
            cash_ratio,
        ])
        return obs.astype(np.float32)

    def _current_prices(self) -> np.ndarray:
        """Return the (normalised) close price for each ticker at the
        current step.  Close is the 0-th feature column."""
        raw = self.data[self.current_step, :, 0]
        # Ensure prices are positive for accounting (shift normalised prices)
        return np.maximum(raw + 2.0, 0.01)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + np.sum(self.holdings * prices))

    def _execute_actions(self, action: np.ndarray, prices: np.ndarray) -> None:
        """Simulate order execution with transaction costs."""
        # Sell first (negative actions)
        for i in range(self.num_tickers):
            if action[i] < 0 and self.holdings[i] > 0:
                sell_fraction = min(abs(action[i]), 1.0)
                sell_amount = self.holdings[i] * sell_fraction
                proceeds = sell_amount * prices[i] * (1.0 - self.transaction_cost)
                self.holdings[i] -= sell_amount
                self.cash += proceeds

        # Buy (positive actions)
        total_buy_signal = max(np.sum(np.maximum(action, 0)), 1e-8)
        for i in range(self.num_tickers):
            if action[i] > 0:
                # Fraction of available cash proportional to action strength
                buy_weight = action[i] / total_buy_signal
                max_pos_pct = RISK_CONFIG["max_position_pct"]
                portfolio_val = self._portfolio_value(prices)

                # Current position value
                current_pos_value = self.holdings[i] * prices[i]
                max_additional = max(0.0, max_pos_pct * portfolio_val - current_pos_value)

                # Cash to spend
                spend = min(
                    self.cash * buy_weight * action[i],
                    max_additional,
                    self.cash * 0.95,  # never spend all cash
                )
                if spend > 0 and prices[i] > 0:
                    cost = spend * (1.0 + self.transaction_cost)
                    if cost <= self.cash:
                        shares = spend / prices[i]
                        self.holdings[i] += shares
                        self.cash -= cost

    def _compute_reward(self, prev_value: float, curr_value: float) -> float:
        """Sharpe-ratio-inspired reward with cost penalty.

        Returns a reward that balances:
        - Risk-adjusted return (Sharpe-like)
        - Penalty for excessive trading (turnover cost)
        """
        if prev_value <= 0:
            return 0.0

        daily_return = (curr_value - prev_value) / prev_value
        risk_free_daily = 0.04 / 252  # ~4% annual risk-free

        # Rolling volatility of returns (use >= 20 for stability)
        if len(self.portfolio_values) >= 20:
            recent_values = np.array(self.portfolio_values[-20:])
            returns = np.diff(recent_values) / recent_values[:-1]
            vol = float(np.std(returns)) if len(returns) > 1 else 0.01
        elif len(self.portfolio_values) >= 5:
            recent_values = np.array(self.portfolio_values[-5:])
            returns = np.diff(recent_values) / recent_values[:-1]
            vol = float(np.std(returns)) if len(returns) > 1 else 0.01
        else:
            vol = 0.01  # stable default, not extreme 1e-6

        vol = max(vol, 0.001)  # floor to prevent reward explosion
        sharpe_reward = (daily_return - risk_free_daily) / vol

        # Clamp to prevent extreme outliers during training
        sharpe_reward = np.clip(sharpe_reward, -5.0, 5.0)

        return float(sharpe_reward * self.reward_scaling)


# ======================================================================
# Ensemble agent
# ======================================================================


class EnsembleAgent:
    """Trains and combines PPO, A2C and SAC sub-agents.

    At inference time the three models produce independent action
    vectors which are averaged using weights derived from their
    rolling Sharpe ratios.  Supports checkpoint-based continuous
    learning (FreqAI-style).
    """

    MODEL_CLASSES = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "ddpg": DDPG,  # kept for backward compatibility (loading old checkpoints)
    }

    def __init__(self, num_tickers: int = 10, num_features: int = 17) -> None:
        self.num_tickers = num_tickers
        self.num_features = num_features
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {name: 1.0 / 3 for name in ENSEMBLE_CONFIG["models"]}
        self.sharpe_history: Dict[str, List[float]] = {name: [] for name in ENSEMBLE_CONFIG["models"]}
        self.is_trained: bool = False
        self.training_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: np.ndarray,
        total_timesteps: Optional[int] = None,
        *,
        sentiment_scores: Optional[np.ndarray] = None,
        learning_rate: float = 3e-4,
        continue_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Train all sub-agents on the provided training data.

        Parameters
        ----------
        train_data : np.ndarray
            Shape ``(num_days, num_tickers, num_features)``.
        total_timesteps : int or None
            Total environment steps per agent.
        sentiment_scores : np.ndarray or None
            Shape ``(num_days, num_tickers)``.
        learning_rate : float
            Learning rate for all agents.
        continue_from_checkpoint : bool
            If True (default), load existing models and continue training
            instead of starting from scratch (FreqAI-style continuous
            learning).  Falls back to fresh training if no checkpoint
            exists or dimensions mismatch.

        Returns
        -------
        dict  -- training metrics per agent.
        """
        total_timesteps = total_timesteps or TRAIN_CONFIG["total_timesteps"]
        self.num_tickers = train_data.shape[1]
        self.num_features = train_data.shape[2]

        # Attempt to load existing checkpoint for continuous learning
        checkpoint_loaded = False
        if continue_from_checkpoint:
            checkpoint_loaded = self._try_load_checkpoint(train_data)
            if checkpoint_loaded:
                logger.info("Continuing training from checkpoint (continuous learning)")
            else:
                logger.info("No compatible checkpoint found, training from scratch")

        metrics: Dict[str, Any] = {}

        for model_name in ENSEMBLE_CONFIG["models"]:
            logger.info("Training %s for %d timesteps ...", model_name, total_timesteps)

            env = TradingEnv(
                train_data,
                sentiment_scores=sentiment_scores,
            )

            # If checkpoint loaded and model exists, set its env and continue
            if checkpoint_loaded and model_name in self.models:
                model = self.models[model_name]
                model.set_env(env)
                logger.info("Resuming %s from checkpoint", model_name)
            else:
                model = self._create_model(model_name, env, learning_rate, train_data.shape[0], total_timesteps)

            model.learn(total_timesteps=total_timesteps, reset_num_timesteps=not checkpoint_loaded)
            self.models[model_name] = model

            # Evaluate on training env for initial metrics
            train_metrics = self._evaluate_model(model, train_data, sentiment_scores)
            metrics[model_name] = train_metrics
            self.sharpe_history[model_name].append(train_metrics.get("sharpe_ratio", 0.0))

            logger.info(
                "%s training complete -- Sharpe: %.3f, Total return: %.2f%%",
                model_name,
                train_metrics.get("sharpe_ratio", 0.0),
                train_metrics.get("total_return_pct", 0.0),
            )

        self._compute_weights()
        self.is_trained = True
        self.training_metrics = metrics

        # Auto-save
        self.save_models()

        return metrics

    def _try_load_checkpoint(self, train_data: np.ndarray) -> bool:
        """Try to load a compatible checkpoint. Returns True if successful."""
        meta_path = MODELS_DIR / "ensemble_meta.pkl"
        if not meta_path.exists():
            return False

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)  # noqa: S301

            # Check dimension compatibility
            saved_tickers = meta.get("num_tickers", 0)
            saved_features = meta.get("num_features", 0)
            if saved_tickers != train_data.shape[1] or saved_features != train_data.shape[2]:
                logger.warning(
                    "Checkpoint dimension mismatch: saved(%d,%d) vs data(%d,%d). Starting fresh.",
                    saved_tickers, saved_features, train_data.shape[1], train_data.shape[2],
                )
                return False

            # Restore metadata
            self.weights = meta.get("weights", self.weights)
            self.sharpe_history = meta.get("sharpe_history", self.sharpe_history)

            # Load models with a dummy env
            dummy_env = TradingEnv(np.zeros((10, self.num_tickers, self.num_features), dtype=np.float32))
            loaded_any = False
            for name in ENSEMBLE_CONFIG["models"]:
                model_path = MODELS_DIR / f"{name}_model.zip"
                if model_path.exists() and name in self.MODEL_CLASSES:
                    cls = self.MODEL_CLASSES[name]
                    self.models[name] = cls.load(str(MODELS_DIR / f"{name}_model"), env=dummy_env)
                    loaded_any = True

            return loaded_any
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
            return False

    def _create_model(self, model_name: str, env: TradingEnv, learning_rate: float, num_days: int, total_timesteps: int) -> Any:
        """Create a fresh model instance for the given algorithm."""
        if model_name == "ppo":
            return PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=min(2048, num_days - 1),
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=0,
            )
        elif model_name == "a2c":
            return A2C(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=5,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                verbose=0,
            )
        elif model_name == "sac":
            from stable_baselines3 import SAC
            return SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=50_000,
                learning_starts=min(1000, total_timesteps // 10),
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                ent_coef="auto",
                verbose=0,
            )
        elif model_name == "ddpg":
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions),
            )
            return DDPG(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=50_000,
                learning_starts=min(1000, total_timesteps // 10),
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                action_noise=action_noise,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return a weighted ensemble action for a single observation.

        Parameters
        ----------
        observation : np.ndarray
            Flat observation vector from ``TradingEnv``.

        Returns
        -------
        np.ndarray  -- action vector of shape ``(num_tickers,)``.
        """
        if not self.models:
            raise RuntimeError("No trained models available. Call train() first or load_models().")

        combined_action = np.zeros(self.num_tickers, dtype=np.float64)

        for name, model in self.models.items():
            action, _ = model.predict(observation, deterministic=True)
            weight = self.weights.get(name, 1.0 / len(self.models))
            combined_action += weight * action

        return np.clip(combined_action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _compute_weights(self) -> None:
        """Compute dynamic ensemble weights based on rolling Sharpe ratios.

        Each model's weight is proportional to ``max(sharpe, min_weight)``.
        """
        min_w = ENSEMBLE_CONFIG["min_weight"]
        raw_weights: Dict[str, float] = {}

        for name in ENSEMBLE_CONFIG["models"]:
            history = self.sharpe_history.get(name, [])
            if history:
                # Use the most recent window
                window = ENSEMBLE_CONFIG["sharpe_window"]
                recent = history[-window:] if len(history) >= window else history
                avg_sharpe = float(np.mean(recent))
                raw_weights[name] = max(avg_sharpe, min_w)
            else:
                raw_weights[name] = min_w

        total = sum(raw_weights.values())
        if total <= 0:
            # Equal weights fallback
            n = len(ENSEMBLE_CONFIG["models"])
            self.weights = {name: 1.0 / n for name in ENSEMBLE_CONFIG["models"]}
        else:
            self.weights = {name: w / total for name, w in raw_weights.items()}

        logger.info("Ensemble weights updated: %s", self.weights)

    # ------------------------------------------------------------------
    # Evaluation / backtesting
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model: Any,
        data: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Run a single model through the data and collect performance
        metrics."""
        env = TradingEnv(data, sentiment_scores=sentiment_scores)
        obs, _ = env.reset()
        portfolio_values = [env.initial_cash]
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])
            if truncated:
                break

        return self._compute_performance_metrics(portfolio_values)

    def backtest(
        self,
        test_data: np.ndarray,
        *,
        sentiment_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the ensemble agent on test data and return performance.

        Parameters
        ----------
        test_data : np.ndarray
            Shape ``(num_days, num_tickers, num_features)``.

        Returns
        -------
        dict  -- performance metrics including Sharpe, max drawdown,
                 total return, etc.
        """
        if not self.models:
            raise RuntimeError("No trained models. Train or load first.")

        env = TradingEnv(test_data, sentiment_scores=sentiment_scores)
        obs, _ = env.reset()
        portfolio_values = [env.initial_cash]
        actions_taken: List[np.ndarray] = []
        done = False

        while not done:
            action = self.predict(obs)
            obs, _, done, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])
            actions_taken.append(action)
            if truncated:
                break

        metrics = self._compute_performance_metrics(portfolio_values)
        metrics["num_steps"] = len(actions_taken)
        metrics["final_cash"] = float(env.cash)
        metrics["final_holdings"] = env.holdings.tolist()
        return metrics

    @staticmethod
    def _compute_performance_metrics(portfolio_values: List[float]) -> Dict[str, float]:
        """Compute standard performance metrics from a portfolio value
        time series."""
        values = np.array(portfolio_values, dtype=np.float64)
        if len(values) < 2:
            return {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "volatility": 0.0,
                "calmar_ratio": 0.0,
            }

        returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        total_return = (values[-1] - values[0]) / max(values[0], 1e-8)

        # Annualised Sharpe (252 trading days)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret * 252**0.5) / max(std_ret, 1e-8) if std_ret > 1e-8 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / np.maximum(peak, 1e-8)
        max_dd = float(np.max(drawdown))

        # Calmar ratio
        annualised_return = total_return * (252 / max(len(returns), 1))
        calmar = annualised_return / max(max_dd, 1e-8) if max_dd > 1e-8 else 0.0

        return {
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd * 100),
            "volatility": float(std_ret * 252**0.5),
            "calmar_ratio": float(calmar),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self, path: Optional[Path] = None) -> None:
        """Save all trained models and metadata to disk."""
        path = path or MODELS_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = path / f"{name}_model"
            model.save(str(model_path))
            logger.info("Saved %s to %s", name, model_path)

        # Save weights and metadata
        meta = {
            "weights": self.weights,
            "sharpe_history": self.sharpe_history,
            "num_tickers": self.num_tickers,
            "num_features": self.num_features,
            "training_metrics": self.training_metrics,
            "is_trained": self.is_trained,
        }
        with open(path / "ensemble_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Ensemble metadata saved.")

    def load_models(self, path: Optional[Path] = None) -> bool:
        """Load previously saved models.  Returns True if successful.

        Loads each SB3 model without env first to read its actual
        observation/action dimensions from the saved weights, then
        derives ``num_tickers`` and ``num_features`` so the dummy env
        matches exactly — avoiding shape mismatches when metadata is
        stale.
        """
        path = path or MODELS_DIR
        path = Path(path)

        meta_path = path / "ensemble_meta.pkl"
        if not meta_path.exists():
            logger.warning("No ensemble metadata found at %s", meta_path)
            return False

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)  # noqa: S301

        self.weights = meta.get("weights", self.weights)
        self.sharpe_history = meta.get("sharpe_history", self.sharpe_history)
        self.training_metrics = meta.get("training_metrics", {})

        # Tentative dimensions from metadata (may be stale)
        meta_n_tickers = meta.get("num_tickers", self.num_tickers)
        meta_n_features = meta.get("num_features", self.num_features)

        loaded_any = False
        for name in ENSEMBLE_CONFIG["models"]:
            model_path = path / f"{name}_model.zip"
            if not model_path.exists():
                logger.warning("Model file not found: %s", model_path)
                continue

            cls = self.MODEL_CLASSES[name]

            # Load without env to inspect actual dimensions
            try:
                model = cls.load(str(path / f"{name}_model"))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", name, exc)
                continue

            actual_obs = model.observation_space.shape[0]
            actual_act = model.action_space.shape[0]

            # Derive: obs_size = (num_features + 2) * num_tickers + 1
            model_n_tickers = actual_act
            model_n_features = (actual_obs - 1) // model_n_tickers - 2

            logger.info("Model %s: obs=%d, act=%d -> tickers=%d, features=%d",
                        name, actual_obs, actual_act, model_n_tickers, model_n_features)

            if not loaded_any:
                # First model sets the canonical dimensions
                self.num_tickers = model_n_tickers
                self.num_features = model_n_features
                loaded_any = True
            else:
                # Subsequent models must match
                if model_n_tickers != self.num_tickers:
                    logger.warning("Model %s dimension mismatch (tickers=%d vs %d), skipping",
                                   name, model_n_tickers, self.num_tickers)
                    continue

            # Create dummy env with correct dimensions and attach to model
            dummy_data = np.zeros((10, self.num_tickers, self.num_features), dtype=np.float32)
            dummy_env = TradingEnv(dummy_data)
            model.set_env(dummy_env)
            self.models[name] = model
            logger.info("Loaded %s from %s", name, model_path)

        self.is_trained = loaded_any
        return loaded_any
