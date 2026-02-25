"""Backtest validator for the sector scanner.

Answers the question: "If we had followed the scanner's recommendations
over the past N months, would we have beaten SPY?"

Methodology:
- Rolling monthly windows over the past 12 months
- At each window start, run scan_sectors() + pick_stocks() using data
  available at that point (no lookahead)
- Measure the recommended tickers' forward 1-month return vs SPY
- Compute hit rate, excess return, and information ratio

Paper backing:
- arXiv 2512.10913: implementation quality > algorithm complexity
- Moskowitz-Grinblatt 1999: sector momentum 0.43% monthly excess return

Usage::

    python scan_backtester.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import SECTOR_MAP, SECTOR_SCAN_CONFIG

logger = logging.getLogger(__name__)


class ScanBacktester:
    """Backtest the sector scanner's recommendations."""

    def __init__(self) -> None:
        self.cfg = SECTOR_SCAN_CONFIG

    def backtest(
        self,
        months: int = 12,
        top_sectors: int = 3,
        stocks_per_sector: int = 5,
        holding_days: int = 21,
    ) -> Dict[str, Any]:
        """Run a rolling backtest of the sector scanner.

        For each month in the past *months*, simulates what the scanner
        would have recommended and measures the forward return.

        Returns a dict with:
        - monthly results (each period's picks and returns)
        - aggregate statistics (hit rate, avg excess return, IR)
        """
        benchmark = self.cfg["benchmark"]
        lookback_weeks = self.cfg["lookback_weeks"]
        weights = self.cfg["momentum_weights"]
        max_lookback_days = max(lookback_weeks) * 7 + 10

        end_date = datetime.now()
        # Need data going back months + max_lookback
        total_days = months * 30 + max_lookback_days + holding_days + 30
        start_date = end_date - timedelta(days=total_days)

        logger.info("Backtest: downloading ETF+benchmark data for %d months", months)

        # Download all ETF data once
        etf_tickers = list(set(
            [info["etf"] for info in SECTOR_MAP.values()] + [benchmark]
        ))
        etf_data = yf.download(
            etf_tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if etf_data.empty:
            return {"error": "No ETF data available"}

        if isinstance(etf_data.columns, pd.MultiIndex):
            etf_close = etf_data["Close"]
        else:
            etf_close = etf_data[["Close"]].rename(columns={"Close": etf_tickers[0]})

        # Download all stock data once
        all_stocks = list(set(
            tic for info in SECTOR_MAP.values() for tic in info["stocks"]
        ))
        stock_data = yf.download(
            all_stocks,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if stock_data.empty:
            return {"error": "No stock data available"}

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data["Close"]
        else:
            stock_close = stock_data[["Close"]].rename(columns={"Close": all_stocks[0]})

        # Generate monthly evaluation dates
        eval_dates = []
        for m in range(months, 0, -1):
            dt = end_date - timedelta(days=m * 30)
            # Find nearest trading day (match index resolution/tz)
            ts = pd.Timestamp(dt)
            if hasattr(etf_close.index, 'tz') and etf_close.index.tz is not None:
                ts = ts.tz_localize(etf_close.index.tz)
            if hasattr(etf_close.index, 'dtype'):
                ts = ts.as_unit(etf_close.index.dtype.str.split('[')[-1].rstrip(']'))
            idx = etf_close.index.searchsorted(ts)
            if idx < len(etf_close.index):
                eval_dates.append(etf_close.index[idx])

        if not eval_dates:
            return {"error": "No valid evaluation dates"}

        # Run rolling backtest
        results: List[Dict[str, Any]] = []

        for eval_date in eval_dates:
            period_result = self._evaluate_period(
                eval_date=eval_date,
                etf_close=etf_close,
                stock_close=stock_close,
                benchmark=benchmark,
                lookback_weeks=lookback_weeks,
                weights=weights,
                top_sectors=top_sectors,
                stocks_per_sector=stocks_per_sector,
                holding_days=holding_days,
            )
            if period_result:
                results.append(period_result)

        if not results:
            return {"error": "No valid backtest periods"}

        # Aggregate statistics
        return self._compute_aggregate(results, months)

    def _evaluate_period(
        self,
        eval_date: pd.Timestamp,
        etf_close: pd.DataFrame,
        stock_close: pd.DataFrame,
        benchmark: str,
        lookback_weeks: List[int],
        weights: List[float],
        top_sectors: int,
        stocks_per_sector: int,
        holding_days: int,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scanner recommendations for a single period."""

        # Get data up to eval_date (no lookahead)
        historical_etf = etf_close.loc[:eval_date]
        historical_stock = stock_close.loc[:eval_date]

        if len(historical_etf) < max(lookback_weeks) * 5 + 10:
            return None

        # Step 1: Rank sectors using data available at eval_date
        sector_rankings = self._rank_sectors(
            historical_etf, benchmark, lookback_weeks, weights
        )

        if len(sector_rankings) < top_sectors:
            return None

        top = sector_rankings[:top_sectors]
        top_sector_names = [s["sector"] for s in top]

        # Step 2: Pick top stocks from each sector (simple momentum)
        picks: List[str] = []
        for sector_name in top_sector_names:
            if sector_name not in SECTOR_MAP:
                continue
            tickers = SECTOR_MAP[sector_name]["stocks"]
            sector_picks = self._rank_stocks(historical_stock, tickers)
            picks.extend(sector_picks[:stocks_per_sector])

        if not picks:
            return None

        # Step 3: Measure forward return (holding_days after eval_date)
        future_data = stock_close.loc[eval_date:]
        spy_future = etf_close[benchmark].loc[eval_date:] if benchmark in etf_close.columns else None

        if len(future_data) < holding_days + 1:
            return None

        # Portfolio return: equal-weight the picks
        pick_returns: List[float] = []
        for tic in picks:
            if tic in future_data.columns:
                series = future_data[tic].dropna()
                if len(series) > holding_days:
                    ret = float(series.iloc[holding_days] / series.iloc[0] - 1)
                    pick_returns.append(ret)

        if not pick_returns:
            return None

        portfolio_return = float(np.mean(pick_returns))

        # Benchmark return
        spy_return = 0.0
        if spy_future is not None and len(spy_future) > holding_days:
            spy_return = float(spy_future.iloc[holding_days] / spy_future.iloc[0] - 1)

        excess_return = portfolio_return - spy_return
        beat_benchmark = portfolio_return > spy_return

        return {
            "eval_date": eval_date.strftime("%Y-%m-%d"),
            "top_sectors": top_sector_names,
            "picks": picks,
            "num_picks": len(pick_returns),
            "portfolio_return": round(portfolio_return, 4),
            "benchmark_return": round(spy_return, 4),
            "excess_return": round(excess_return, 4),
            "beat_benchmark": beat_benchmark,
        }

    def _rank_sectors(
        self,
        etf_close: pd.DataFrame,
        benchmark: str,
        lookback_weeks: List[int],
        weights: List[float],
    ) -> List[Dict[str, Any]]:
        """Rank sectors by composite momentum (same logic as SectorScanner)."""
        # Compute benchmark returns
        spy_returns = {}
        for weeks in lookback_weeks:
            td = weeks * 5
            if benchmark in etf_close.columns and len(etf_close) > td:
                spy_s = etf_close[benchmark].dropna()
                if len(spy_s) > td:
                    spy_returns[weeks] = float(spy_s.iloc[-1] / spy_s.iloc[-td] - 1)
                else:
                    spy_returns[weeks] = 0.0
            else:
                spy_returns[weeks] = 0.0

        results = []
        for sector_name, sector_info in SECTOR_MAP.items():
            etf = sector_info["etf"]
            if etf not in etf_close.columns:
                continue
            etf_s = etf_close[etf].dropna()
            if len(etf_s) < 10:
                continue

            composite = 0.0
            for weeks, weight in zip(lookback_weeks, weights):
                td = weeks * 5
                if len(etf_s) > td:
                    ret = float(etf_s.iloc[-1] / etf_s.iloc[-td] - 1)
                else:
                    ret = 0.0
                rel = ret - spy_returns.get(weeks, 0.0)
                composite += weight * rel

            results.append({"sector": sector_name, "composite_score": composite})

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    def _rank_stocks(
        self,
        stock_close: pd.DataFrame,
        tickers: List[str],
        mom_days: int = 60,
    ) -> List[str]:
        """Rank stocks by momentum within a sector."""
        scores = []
        for tic in tickers:
            if tic not in stock_close.columns:
                continue
            prices = stock_close[tic].dropna()
            if len(prices) < mom_days:
                continue
            momentum = float(prices.iloc[-1] / prices.iloc[-mom_days] - 1)
            scores.append((tic, momentum))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores]

    def _compute_aggregate(
        self,
        results: List[Dict[str, Any]],
        months: int,
    ) -> Dict[str, Any]:
        """Compute aggregate backtest statistics."""
        n = len(results)
        hits = sum(1 for r in results if r["beat_benchmark"])
        hit_rate = hits / n if n > 0 else 0.0

        portfolio_returns = [r["portfolio_return"] for r in results]
        benchmark_returns = [r["benchmark_return"] for r in results]
        excess_returns = [r["excess_return"] for r in results]

        avg_portfolio = float(np.mean(portfolio_returns))
        avg_benchmark = float(np.mean(benchmark_returns))
        avg_excess = float(np.mean(excess_returns))
        std_excess = float(np.std(excess_returns)) if n > 1 else 0.0

        # Information Ratio: avg_excess / std_excess
        info_ratio = avg_excess / std_excess if std_excess > 0 else 0.0

        # Cumulative returns (compounded)
        cum_portfolio = float(np.prod([1 + r for r in portfolio_returns]) - 1)
        cum_benchmark = float(np.prod([1 + r for r in benchmark_returns]) - 1)

        return {
            "backtest_months": months,
            "valid_periods": n,
            "hit_rate": round(hit_rate, 4),
            "avg_monthly_return": round(avg_portfolio, 4),
            "avg_benchmark_return": round(avg_benchmark, 4),
            "avg_excess_return": round(avg_excess, 4),
            "information_ratio": round(info_ratio, 4),
            "cumulative_portfolio": round(cum_portfolio, 4),
            "cumulative_benchmark": round(cum_benchmark, 4),
            "cumulative_excess": round(cum_portfolio - cum_benchmark, 4),
            "monthly_results": results,
            "verdict": _verdict(hit_rate, avg_excess, info_ratio),
        }


def _verdict(hit_rate: float, avg_excess: float, info_ratio: float) -> str:
    """Generate a human-readable verdict."""
    parts = []

    if hit_rate >= 0.6:
        parts.append(f"Hit rate {hit_rate:.0%} — beats benchmark majority of months")
    elif hit_rate >= 0.5:
        parts.append(f"Hit rate {hit_rate:.0%} — coin-flip, no edge")
    else:
        parts.append(f"Hit rate {hit_rate:.0%} — UNDERPERFORMS benchmark")

    if avg_excess > 0.01:
        parts.append(f"Avg excess +{avg_excess:.2%}/month — meaningful alpha")
    elif avg_excess > 0:
        parts.append(f"Avg excess +{avg_excess:.2%}/month — marginal alpha")
    else:
        parts.append(f"Avg excess {avg_excess:.2%}/month — NEGATIVE alpha")

    if info_ratio > 0.5:
        parts.append(f"IR {info_ratio:.2f} — good risk-adjusted excess")
    elif info_ratio > 0:
        parts.append(f"IR {info_ratio:.2f} — weak risk-adjusted excess")
    else:
        parts.append(f"IR {info_ratio:.2f} — no risk-adjusted edge")

    # Moskowitz-Grinblatt benchmark: 0.43% monthly excess
    if avg_excess >= 0.0043:
        parts.append("Meets Moskowitz-Grinblatt (1999) benchmark of 0.43%/month")
    else:
        parts.append(f"Below Moskowitz-Grinblatt benchmark (need 0.43%/month, got {avg_excess:.2%})")

    return " | ".join(parts)


# ------------------------------------------------------------------
# API endpoint integration
# ------------------------------------------------------------------

def run_backtest(months: int = 12) -> Dict[str, Any]:
    """Entry point for server.py integration."""
    bt = ScanBacktester()
    return bt.backtest(months=months)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bt = ScanBacktester()
    result = bt.backtest(months=12)

    if "error" in result:
        print(f"Backtest failed: {result['error']}")
    else:
        print(f"\n{'='*70}")
        print(f"SECTOR SCANNER BACKTEST - {result['backtest_months']} months")
        print(f"{'='*70}")
        print(f"Valid periods:        {result['valid_periods']}")
        print(f"Hit rate:             {result['hit_rate']:.1%}")
        print(f"Avg monthly return:   {result['avg_monthly_return']:+.2%}")
        print(f"Avg benchmark (SPY):  {result['avg_benchmark_return']:+.2%}")
        print(f"Avg excess return:    {result['avg_excess_return']:+.2%}")
        print(f"Information Ratio:    {result['information_ratio']:.2f}")
        print(f"Cumulative portfolio: {result['cumulative_portfolio']:+.2%}")
        print(f"Cumulative benchmark: {result['cumulative_benchmark']:+.2%}")
        print(f"Cumulative excess:    {result['cumulative_excess']:+.2%}")
        print(f"\nVerdict: {result['verdict']}")

        print(f"\n--- Monthly Breakdown ---")
        for r in result["monthly_results"]:
            mark = "+" if r["beat_benchmark"] else "-"
            print(
                f"  [{mark}] {r['eval_date']}  "
                f"port={r['portfolio_return']:+.2%}  "
                f"spy={r['benchmark_return']:+.2%}  "
                f"excess={r['excess_return']:+.2%}  "
                f"sectors={r['top_sectors']}"
            )
