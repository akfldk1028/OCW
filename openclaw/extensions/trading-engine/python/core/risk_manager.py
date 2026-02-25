"""Risk management module for the trading engine.

Provides pre-trade checks including take-profit / stop-loss evaluation,
Kelly-criterion position sizing, daily trade limits and portfolio
exposure constraints.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from config import RISK_CONFIG

logger = logging.getLogger(__name__)


class RiskManager:
    """Stateful risk manager that tracks daily trades and enforces
    position / exposure limits.

    All configuration comes from ``config.RISK_CONFIG`` but can be
    overridden at construction time.
    """

    def __init__(
        self,
        *,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_daily_trades: Optional[int] = None,
        max_position_pct: Optional[float] = None,
        max_exposure_pct: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ) -> None:
        self.take_profit_pct: float = take_profit_pct if take_profit_pct is not None else RISK_CONFIG["take_profit_pct"]
        self.stop_loss_pct: float = stop_loss_pct if stop_loss_pct is not None else RISK_CONFIG["stop_loss_pct"]
        self.max_daily_trades: int = max_daily_trades if max_daily_trades is not None else RISK_CONFIG["max_daily_trades"]
        self.max_position_pct: float = max_position_pct if max_position_pct is not None else RISK_CONFIG["max_position_pct"]
        self.max_exposure_pct: float = max_exposure_pct if max_exposure_pct is not None else RISK_CONFIG["max_exposure_pct"]
        self.kelly_fraction: float = kelly_fraction if kelly_fraction is not None else RISK_CONFIG["kelly_fraction"]

        # ATR dynamic TP/SL
        self.use_atr_dynamic: bool = RISK_CONFIG.get("use_atr_dynamic", False)
        self.atr_tp_multiplier: float = RISK_CONFIG.get("atr_tp_multiplier", 2.5)
        self.atr_sl_multiplier: float = RISK_CONFIG.get("atr_sl_multiplier", 1.5)

        # Trailing stop
        self.trailing_stop_enabled: bool = RISK_CONFIG.get("trailing_stop_enabled", False)
        self.trailing_stop_atr_multiplier: float = RISK_CONFIG.get("trailing_stop_atr_multiplier", 2.0)
        self._trailing_highs: Dict[str, float] = {}  # ticker -> highest price since entry

        # Daily state
        self._trade_count: int = 0
        self._current_date: date = date.today()
        self._trade_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # ATR-based dynamic TP/SL
    # ------------------------------------------------------------------

    def get_dynamic_tp_sl(
        self, ticker: str, entry_price: float, atr: Optional[float] = None
    ) -> tuple:
        """Return (tp_pct, sl_pct) dynamically adjusted by ATR if enabled.

        Parameters
        ----------
        ticker : str
            Ticker symbol (for logging).
        entry_price : float
            Original entry price.
        atr : float or None
            Current ATR value. If None or ATR disabled, uses static defaults.

        Returns
        -------
        tuple of (tp_pct, sl_pct)
        """
        if self.use_atr_dynamic and atr is not None and entry_price > 0 and atr > 0:
            tp_pct = (atr * self.atr_tp_multiplier) / entry_price
            sl_pct = -(atr * self.atr_sl_multiplier) / entry_price
            logger.debug(
                "ATR dynamic TP/SL for %s: ATR=%.4f, TP=%.2f%%, SL=%.2f%%",
                ticker, atr, tp_pct * 100, sl_pct * 100,
            )
            return tp_pct, sl_pct
        return self.take_profit_pct, self.stop_loss_pct

    # ------------------------------------------------------------------
    # Take-profit / Stop-loss
    # ------------------------------------------------------------------

    def check_take_profit(
        self, ticker: str, entry_price: float, current_price: float,
        atr: Optional[float] = None,
    ) -> bool:
        """Return ``True`` if the position should be closed for profit.

        Parameters
        ----------
        ticker : str
            Ticker symbol (for logging).
        entry_price : float
            Original entry price.
        current_price : float
            Current market price.
        atr : float or None
            Current ATR for dynamic adjustment.
        """
        if entry_price <= 0:
            return False
        tp_pct, _ = self.get_dynamic_tp_sl(ticker, entry_price, atr)
        pnl_pct = (current_price - entry_price) / entry_price
        triggered = pnl_pct >= tp_pct
        if triggered:
            logger.info(
                "TAKE-PROFIT triggered for %s: entry=%.4f, current=%.4f, pnl=%.2f%% (threshold=%.2f%%)",
                ticker, entry_price, current_price, pnl_pct * 100, tp_pct * 100,
            )
        return triggered

    def check_stop_loss(
        self, ticker: str, entry_price: float, current_price: float,
        atr: Optional[float] = None,
    ) -> bool:
        """Return ``True`` if the position should be closed for loss
        protection.

        Note that ``stop_loss_pct`` is expected to be negative (e.g.
        ``-0.025`` for a -2.5% stop).
        """
        if entry_price <= 0:
            return False
        _, sl_pct = self.get_dynamic_tp_sl(ticker, entry_price, atr)
        pnl_pct = (current_price - entry_price) / entry_price
        triggered = pnl_pct <= sl_pct
        if triggered:
            logger.info(
                "STOP-LOSS triggered for %s: entry=%.4f, current=%.4f, pnl=%.2f%% (threshold=%.2f%%)",
                ticker, entry_price, current_price, pnl_pct * 100, sl_pct * 100,
            )
        return triggered

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def update_trailing_high(self, ticker: str, current_price: float) -> None:
        """Update the trailing high watermark for a position."""
        prev_high = self._trailing_highs.get(ticker, 0.0)
        if current_price > prev_high:
            self._trailing_highs[ticker] = current_price

    def check_trailing_stop(
        self, ticker: str, current_price: float, atr: Optional[float] = None,
    ) -> bool:
        """Return ``True`` if trailing stop is triggered.

        The trailing stop activates once a position is in profit and
        price drops by ``trailing_stop_atr_multiplier * ATR`` from the
        highest observed price.
        """
        if not self.trailing_stop_enabled:
            return False

        high = self._trailing_highs.get(ticker)
        if high is None or high <= 0:
            return False

        if atr is not None and atr > 0:
            trail_distance = atr * self.trailing_stop_atr_multiplier
        else:
            # Fallback: use 3% of high as trail distance
            trail_distance = high * 0.03

        trail_stop_price = high - trail_distance
        triggered = current_price <= trail_stop_price
        if triggered:
            logger.info(
                "TRAILING-STOP triggered for %s: high=%.4f, trail_stop=%.4f, current=%.4f",
                ticker, high, trail_stop_price, current_price,
            )
        return triggered

    def reset_trailing(self, ticker: str) -> None:
        """Clear the trailing high for a ticker (on position close)."""
        self._trailing_highs.pop(ticker, None)

    # ------------------------------------------------------------------
    # Position sizing (Kelly Criterion)
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        ticker: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float,
    ) -> float:
        """Compute the recommended position size in currency units using
        the Kelly Criterion (fractional).

        Parameters
        ----------
        ticker : str
            Ticker symbol (for logging).
        win_rate : float
            Historical win rate in [0, 1].
        avg_win : float
            Average winning trade return (positive, e.g. 0.02 for 2%).
        avg_loss : float
            Average losing trade return (positive magnitude, e.g. 0.01).
        portfolio_value : float
            Current total portfolio value.

        Returns
        -------
        float
            Recommended position size in currency units, capped by
            ``max_position_pct``.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1 or portfolio_value <= 0:
            logger.debug("Kelly: invalid inputs for %s -- returning 0", ticker)
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        # where b = avg_win / avg_loss, p = win_rate, q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        kelly_full = (p * b - q) / b

        if kelly_full <= 0:
            logger.debug("Kelly negative for %s (%.4f) -- no bet recommended", ticker, kelly_full)
            return 0.0

        # Apply fractional Kelly for safety
        kelly_adj = kelly_full * self.kelly_fraction

        # Cap at max position size
        position_fraction = min(kelly_adj, self.max_position_pct)
        position_size = position_fraction * portfolio_value

        logger.debug(
            "Kelly for %s: full=%.4f, adj=%.4f, size=%.2f (of %.2f)",
            ticker, kelly_full, kelly_adj, position_size, portfolio_value,
        )
        return position_size

    # ------------------------------------------------------------------
    # Daily trade limit
    # ------------------------------------------------------------------

    def check_daily_limit(self) -> bool:
        """Return ``True`` if more trades are allowed today.

        Automatically resets the counter when the date rolls over.
        """
        self._maybe_reset_daily()
        allowed = self._trade_count < self.max_daily_trades
        if not allowed:
            logger.warning("Daily trade limit reached (%d/%d)", self._trade_count, self.max_daily_trades)
        return allowed

    def record_trade(self, ticker: str, side: str, amount: float, price: float) -> None:
        """Record a trade execution and increment the daily counter.

        Parameters
        ----------
        ticker : str
            Symbol traded.
        side : str
            ``"buy"`` or ``"sell"``.
        amount : float
            Number of shares / units.
        price : float
            Execution price.
        """
        self._maybe_reset_daily()
        self._trade_count += 1
        self._trade_log.append({
            "timestamp": datetime.now(tz=None).isoformat(),
            "ticker": ticker,
            "side": side,
            "amount": amount,
            "price": price,
        })

    # ------------------------------------------------------------------
    # Exposure checks
    # ------------------------------------------------------------------

    def check_max_exposure(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float,
    ) -> float:
        """Return the maximum additional allocation (in currency) allowed
        before hitting the total exposure cap.

        Parameters
        ----------
        current_positions : dict[str, float]
            Mapping of ticker to current position value (in currency).
        portfolio_value : float
            Total portfolio value (cash + positions).

        Returns
        -------
        float
            Maximum additional amount that can be invested.
        """
        if portfolio_value <= 0:
            return 0.0

        total_exposure = sum(current_positions.values())
        exposure_ratio = total_exposure / portfolio_value
        max_allowed = self.max_exposure_pct * portfolio_value
        remaining = max(0.0, max_allowed - total_exposure)

        logger.debug(
            "Exposure check: current=%.2f%%, max=%.2f%%, remaining=%.2f",
            exposure_ratio * 100, self.max_exposure_pct * 100, remaining,
        )
        return remaining

    # ------------------------------------------------------------------
    # Comprehensive pre-trade evaluation
    # ------------------------------------------------------------------

    def evaluate_action(
        self,
        action: float,
        ticker: str,
        portfolio_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a comprehensive pre-trade check and return a decision
        dict.

        Parameters
        ----------
        action : float
            Raw action value in [-1, 1].  Positive = buy, negative = sell.
        ticker : str
            Ticker symbol.
        portfolio_state : dict
            Must contain keys:
            - ``portfolio_value`` (float)
            - ``cash`` (float)
            - ``positions`` (dict[str, dict]) with per-ticker
              ``{"value": float, "entry_price": float, "current_price": float,
                "shares": float, "atr": float (optional)}``
            - ``win_rate`` (float, optional)
            - ``avg_win`` (float, optional)
            - ``avg_loss`` (float, optional)

        Returns
        -------
        dict with keys:
            ``allowed`` (bool), ``adjusted_action`` (float),
            ``reason`` (str), ``position_size`` (float),
            ``signals`` (dict of booleans for TP/SL/trailing/limits).
        """
        result: Dict[str, Any] = {
            "allowed": True,
            "adjusted_action": action,
            "reason": "ok",
            "position_size": 0.0,
            "signals": {
                "take_profit": False,
                "stop_loss": False,
                "trailing_stop": False,
                "daily_limit_ok": True,
                "exposure_ok": True,
            },
        }

        pv = portfolio_state.get("portfolio_value", 0.0)
        cash = portfolio_state.get("cash", 0.0)
        positions = portfolio_state.get("positions", {})

        # 1. Daily trade limit
        if not self.check_daily_limit():
            result["allowed"] = False
            result["reason"] = "daily trade limit reached"
            result["signals"]["daily_limit_ok"] = False
            return result

        # 2. Check TP/SL/Trailing for existing position
        pos = positions.get(ticker)
        if pos is not None:
            entry_price = pos.get("entry_price", 0.0)
            current_price = pos.get("current_price", 0.0)
            atr = pos.get("atr")  # ATR value for dynamic adjustment

            # Update trailing high watermark
            if self.trailing_stop_enabled and current_price > 0:
                self.update_trailing_high(ticker, current_price)

            if self.check_take_profit(ticker, entry_price, current_price, atr=atr):
                result["signals"]["take_profit"] = True
                if pos.get("shares", 0) > 0:
                    result["adjusted_action"] = -1.0
                    result["reason"] = "take-profit triggered -- forced sell"
                    result["position_size"] = pos.get("value", 0.0)
                    self.reset_trailing(ticker)
                    return result

            if self.check_stop_loss(ticker, entry_price, current_price, atr=atr):
                result["signals"]["stop_loss"] = True
                if pos.get("shares", 0) > 0:
                    result["adjusted_action"] = -1.0
                    result["reason"] = "stop-loss triggered -- forced sell"
                    result["position_size"] = pos.get("value", 0.0)
                    self.reset_trailing(ticker)
                    return result

            # 2b. Trailing stop check
            if self.check_trailing_stop(ticker, current_price, atr=atr):
                result["signals"]["trailing_stop"] = True
                if pos.get("shares", 0) > 0:
                    result["adjusted_action"] = -1.0
                    result["reason"] = "trailing-stop triggered -- forced sell"
                    result["position_size"] = pos.get("value", 0.0)
                    self.reset_trailing(ticker)
                    return result

        # 3. Exposure check for buys
        if action > 0:
            position_values = {t: p.get("value", 0.0) for t, p in positions.items()}
            remaining_exposure = self.check_max_exposure(position_values, pv)

            if remaining_exposure <= 0:
                result["allowed"] = False
                result["reason"] = "max exposure limit reached"
                result["signals"]["exposure_ok"] = False
                return result

            # 4. Position sizing via Kelly
            win_rate = portfolio_state.get("win_rate", 0.55)
            avg_win = portfolio_state.get("avg_win", 0.02)
            avg_loss = portfolio_state.get("avg_loss", 0.01)

            kelly_size = self.compute_position_size(ticker, win_rate, avg_win, avg_loss, pv)
            max_size = min(kelly_size, remaining_exposure, cash)

            # Check single-position concentration
            current_pos_value = pos.get("value", 0.0) if pos else 0.0
            max_additional = max(0.0, self.max_position_pct * pv - current_pos_value)
            max_size = min(max_size, max_additional)

            if max_size <= 0:
                result["allowed"] = False
                result["reason"] = "position would exceed concentration limit"
                return result

            result["position_size"] = max_size

            # Scale action if the full action would exceed limits
            desired_size = abs(action) * cash
            if desired_size > max_size and desired_size > 0:
                result["adjusted_action"] = action * (max_size / desired_size)

        elif action < 0 and pos is not None:
            # Sell -- position size is the amount being sold
            sell_fraction = min(abs(action), 1.0)
            result["position_size"] = sell_fraction * pos.get("value", 0.0)

        return result

    # ------------------------------------------------------------------
    # Resets and utilities
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Explicitly reset daily counters."""
        self._trade_count = 0
        self._current_date = date.today()
        logger.info("Daily risk counters reset.")

    def _maybe_reset_daily(self) -> None:
        """Auto-reset counters if the date has changed."""
        today = date.today()
        if today != self._current_date:
            self._trade_count = 0
            self._current_date = today

    @property
    def daily_trades_remaining(self) -> int:
        """Number of trades still allowed today."""
        self._maybe_reset_daily()
        return max(0, self.max_daily_trades - self._trade_count)

    @property
    def trade_log(self) -> List[Dict[str, Any]]:
        """Return a copy of the trade log."""
        return list(self._trade_log)

    def get_status(self) -> Dict[str, Any]:
        """Return a summary of the risk manager's current state."""
        return {
            "daily_trades_used": self._trade_count,
            "daily_trades_remaining": self.daily_trades_remaining,
            "max_daily_trades": self.max_daily_trades,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_position_pct": self.max_position_pct,
            "max_exposure_pct": self.max_exposure_pct,
            "kelly_fraction": self.kelly_fraction,
            "total_trades_logged": len(self._trade_log),
            "atr_dynamic_enabled": self.use_atr_dynamic,
            "atr_tp_multiplier": self.atr_tp_multiplier,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_atr_multiplier": self.trailing_stop_atr_multiplier,
            "active_trailing_positions": list(self._trailing_highs.keys()),
        }
