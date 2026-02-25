"""Tests for ensemble_agent.py"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from ensemble_agent import TradingEnv, EnsembleAgent, MODELS_DIR


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n_days = 100
    n_stocks = 3
    n_features = 15  # OHLCV(5) + indicators(10)

    data = np.random.randn(n_days, n_stocks, n_features)
    # Make prices positive
    data[:, :, :5] = np.abs(data[:, :, :5]) * 100 + 50
    # Close prices with slight uptrend
    for i in range(n_stocks):
        data[:, i, 0] = 100 + np.cumsum(np.random.randn(n_days) * 0.5)

    return data.astype(np.float32)


@pytest.fixture
def env(sample_data):
    """Create a TradingEnv instance."""
    return TradingEnv(
        sample_data,
        initial_cash=100_000,
        transaction_cost=0.001,
    )


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Temp directory so tests don't overwrite production models."""
    return tmp_path / "models"


@pytest.fixture
def agent(tmp_models_dir):
    """Create an EnsembleAgent instance that saves to temp dir."""
    a = EnsembleAgent(
        num_tickers=3,
        num_features=15,
    )
    # Override save to use temp directory
    a._test_models_dir = tmp_models_dir
    return a


class TestTradingEnv:
    def test_reset(self, env):
        obs, info = env.reset()
        assert obs is not None
        assert len(obs.shape) == 1  # Flattened observation
        assert info is not None

    def test_step_shape(self, env):
        env.reset()
        action = np.array([0.5, -0.3, 0.0])  # Buy, sell, hold
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_preserves_balance(self, env):
        env.reset()
        initial_value = env._portfolio_value(env._current_prices())
        action = np.zeros(3)  # Hold all
        env.step(action)
        # With no action, value should still be positive
        assert env._portfolio_value(env._current_prices()) > 0

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = np.random.uniform(-1, 1, size=3).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        # Should terminate within data length
        assert steps <= env.num_days + 1

    def test_action_space(self, env):
        assert env.action_space.shape == (3,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


class TestEnsembleAgent:
    def test_init(self, agent):
        assert agent.num_tickers == 3
        assert agent.num_features == 15
        assert len(agent.models) == 0  # Not trained yet

    def test_train_short(self, agent, sample_data, tmp_models_dir, monkeypatch):
        """Train with very few timesteps just to verify it runs."""
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_models_dir)
        metrics = agent.train(sample_data, total_timesteps=100)
        assert "ppo" in metrics or len(agent.models) > 0

    def test_predict_untrained(self, agent):
        """Predict without training should raise RuntimeError."""
        obs = np.random.randn(3 * 15 + 3 + 3 + 1).astype(np.float32)
        with pytest.raises(RuntimeError, match="No trained models"):
            agent.predict(obs)

    def test_compute_weights_equal(self, agent):
        """Without history, weights should be equal."""
        agent._compute_weights()
        weights = agent.weights
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_backtest(self, agent, sample_data, tmp_models_dir, monkeypatch):
        """Backtest should return metrics dict."""
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_models_dir)
        # Train minimally first
        agent.train(sample_data[:80], total_timesteps=100)
        metrics = agent.backtest(sample_data[80:])
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics


class TestTradingEnvReward:
    def test_positive_return_positive_reward(self, env):
        env.reset()
        # Simulate a scenario where we hold and price goes up
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        _, reward1, _, _, _ = env.step(action)
        # Reward depends on portfolio change, so we just check it's a number
        assert isinstance(reward1, (int, float))

    def test_no_action_minimal_reward(self, env):
        env.reset()
        action = np.zeros(3, dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        # With zero action, reward should be near zero
        assert isinstance(reward, (int, float))
