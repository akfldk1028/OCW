"""Tests for risk_manager.py"""
import pytest
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from risk_manager import RiskManager


@pytest.fixture
def rm():
    """Create a RiskManager with default config."""
    return RiskManager()


@pytest.fixture
def rm_custom():
    """Create a RiskManager with custom config."""
    return RiskManager(
        take_profit_pct=0.02,
        stop_loss_pct=-0.01,
        max_daily_trades=5,
        max_position_pct=0.10,
        max_exposure_pct=0.50,
        kelly_fraction=0.5,
    )


class TestTakeProfit:
    def test_below_target(self, rm):
        assert rm.check_take_profit("AAPL", 100.0, 100.5) is False

    def test_at_target(self, rm):
        # Default TP is 1%
        assert rm.check_take_profit("AAPL", 100.0, 101.0) is True

    def test_above_target(self, rm):
        assert rm.check_take_profit("AAPL", 100.0, 105.0) is True

    def test_negative_move(self, rm):
        assert rm.check_take_profit("AAPL", 100.0, 95.0) is False

    def test_custom_target(self, rm_custom):
        # Custom TP is 2%
        assert rm_custom.check_take_profit("AAPL", 100.0, 101.5) is False
        assert rm_custom.check_take_profit("AAPL", 100.0, 102.0) is True


class TestStopLoss:
    def test_above_stop(self, rm):
        assert rm.check_stop_loss("AAPL", 100.0, 99.8) is False

    def test_at_stop(self, rm):
        # Default SL is -0.5%
        assert rm.check_stop_loss("AAPL", 100.0, 99.5) is True

    def test_below_stop(self, rm):
        assert rm.check_stop_loss("AAPL", 100.0, 98.0) is True

    def test_positive_move(self, rm):
        assert rm.check_stop_loss("AAPL", 100.0, 105.0) is False

    def test_custom_stop(self, rm_custom):
        # Custom SL is -1%
        assert rm_custom.check_stop_loss("AAPL", 100.0, 99.5) is False
        assert rm_custom.check_stop_loss("AAPL", 100.0, 99.0) is True


class TestPositionSizing:
    def test_kelly_basic(self, rm):
        # win_rate=0.6, avg_win=1.0, avg_loss=0.5 → Kelly = (0.6*2 - 0.4)/2 = 0.4
        # Quarter Kelly = 0.4 * 0.25 = 0.1
        size = rm.compute_position_size(
            ticker="AAPL",
            win_rate=0.6,
            avg_win=1.0,
            avg_loss=0.5,
            portfolio_value=100_000,
        )
        assert size > 0
        assert size <= 100_000 * 0.15  # Max position limit

    def test_kelly_negative(self, rm):
        # Negative expectancy should return 0
        size = rm.compute_position_size(
            ticker="AAPL",
            win_rate=0.3,
            avg_win=0.5,
            avg_loss=1.0,
            portfolio_value=100_000,
        )
        assert size == 0

    def test_max_position_cap(self, rm):
        # Even with great stats, should be capped at max_position_pct
        size = rm.compute_position_size(
            ticker="AAPL",
            win_rate=0.9,
            avg_win=5.0,
            avg_loss=0.1,
            portfolio_value=100_000,
        )
        assert size <= 100_000 * 0.15 + 1  # Allow rounding

    def test_zero_avg_loss(self, rm):
        # Edge case: avg_loss is 0
        size = rm.compute_position_size(
            ticker="AAPL",
            win_rate=0.6,
            avg_win=1.0,
            avg_loss=0.0,
            portfolio_value=100_000,
        )
        # Should handle gracefully
        assert size >= 0


class TestDailyLimits:
    def test_under_limit(self, rm):
        assert rm.check_daily_limit() is True

    def test_at_limit(self, rm):
        for i in range(20):
            rm.record_trade(f"TEST{i}", "buy", 1.0, 100.0)
        assert rm.check_daily_limit() is False

    def test_reset(self, rm):
        for i in range(20):
            rm.record_trade(f"TEST{i}", "buy", 1.0, 100.0)
        assert rm.check_daily_limit() is False
        rm.reset_daily()
        assert rm.check_daily_limit() is True

    def test_custom_limit(self, rm_custom):
        for i in range(5):
            rm_custom.record_trade(f"TEST{i}", "buy", 1.0, 100.0)
        assert rm_custom.check_daily_limit() is False


class TestMaxExposure:
    def test_no_positions(self, rm):
        remaining = rm.check_max_exposure({}, 100_000)
        assert remaining == 100_000 * 0.80  # 80% max exposure

    def test_partial_positions(self, rm):
        positions = {"AAPL": 50_000}
        remaining = rm.check_max_exposure(positions, 100_000)
        assert remaining == 100_000 * 0.80 - 50_000

    def test_full_exposure(self, rm):
        positions = {"AAPL": 50_000, "MSFT": 30_000}
        remaining = rm.check_max_exposure(positions, 100_000)
        assert remaining == 0  # 80% - 80% = 0

    def test_over_exposed(self, rm):
        positions = {"AAPL": 50_000, "MSFT": 40_000}
        remaining = rm.check_max_exposure(positions, 100_000)
        assert remaining == 0  # Can't go negative


class TestEvaluateAction:
    def test_buy_allowed(self, rm):
        result = rm.evaluate_action(
            action=0.5,
            ticker="AAPL",
            portfolio_state={
                "positions": {},
                "portfolio_value": 100_000,
                "cash": 100_000,
                "daily_trades": 0,
            },
        )
        assert result["allowed"] is True

    def test_buy_at_daily_limit(self, rm):
        for i in range(20):
            rm.record_trade(f"TEST{i}", "buy", 1.0, 100.0)
        result = rm.evaluate_action(
            action=0.5,
            ticker="AAPL",
            portfolio_state={
                "positions": {},
                "portfolio_value": 100_000,
                "daily_trades": 20,
            },
        )
        assert result["allowed"] is False
        assert "daily" in result["reason"].lower()
