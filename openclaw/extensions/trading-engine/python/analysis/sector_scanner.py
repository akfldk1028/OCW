"""Sector rotation scanner and dynamic stock picker.

Scans 14 sector ETFs for relative momentum vs SPY, then ranks
individual stocks within the top sectors using a composite score
of momentum, volume, RSI, and news sentiment.

Usage::

    python sector_scanner.py          # standalone test
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import SECTOR_MAP, SECTOR_SCAN_CONFIG
from analysis.regime_detector import RegimeDetector
from analysis.sentiment_finbert import FinBERTScorer
from analysis.sentiment_scorer import SentimentScorer
from analysis.stock_ranker import StockRanker

logger = logging.getLogger(__name__)

# Sector classification for regime-based adjustment
SECTOR_TRAITS = {
    "Technology":             "growth",
    "Semiconductors":         "growth",
    "Communication":          "growth",
    "Consumer Discretionary": "growth",
    "Innovation":             "speculative",
    "Financials":             "cyclical",
    "Industrials":            "cyclical",
    "Energy":                 "cyclical",
    "Materials":              "cyclical",
    "Healthcare":             "defensive",
    "Consumer Staples":       "defensive",
    "Utilities":              "defensive",
    "Real Estate":            "defensive",
    "Biotech":                "speculative",
}


class SectorScanner:
    """Scan sectors and pick top stocks based on multi-factor scoring."""

    def __init__(
        self,
        sentiment_scorer: Optional[SentimentScorer] = None,
        finbert_scorer: Optional[FinBERTScorer] = None,
        regime_detector: Optional[RegimeDetector] = None,
        stock_ranker: Optional[StockRanker] = None,
    ) -> None:
        self.sentiment_scorer = sentiment_scorer or SentimentScorer()
        self.finbert_scorer = finbert_scorer  # lazy-loaded on first use
        self.regime_detector = regime_detector or RegimeDetector()
        self.stock_ranker = stock_ranker or StockRanker()
        self.cfg = SECTOR_SCAN_CONFIG

    # ------------------------------------------------------------------
    # Sector-level scanning
    # ------------------------------------------------------------------

    def scan_sectors(self) -> List[Dict[str, Any]]:
        """Rank all sectors by weighted relative momentum vs benchmark.

        Returns a list of dicts sorted by composite score (descending)::

            [{"sector": "Semiconductors", "etf": "SMH",
              "composite_score": 0.12, "returns": {...}}, ...]
        """
        lookback_weeks: List[int] = self.cfg["lookback_weeks"]
        weights: List[float] = self.cfg["momentum_weights"]
        benchmark: str = self.cfg["benchmark"]

        # Collect all ETF tickers + benchmark
        etf_tickers = [info["etf"] for info in SECTOR_MAP.values()]
        all_tickers = list(set(etf_tickers + [benchmark]))

        # Need enough history for the longest lookback
        max_days = max(lookback_weeks) * 7 + 10  # extra buffer for weekends
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max_days)

        logger.info(
            "Downloading sector ETF data for %d tickers (%s to %s)",
            len(all_tickers), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
        )

        data = yf.download(
            all_tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            logger.error("No ETF data returned from yfinance")
            return []

        # Extract close prices — handle both single and multi-ticker formats
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]].rename(columns={"Close": all_tickers[0]})

        # Compute benchmark returns for each window
        spy_returns = {}
        for weeks in lookback_weeks:
            trading_days = weeks * 5
            if benchmark in close.columns and len(close) > trading_days:
                spy_series = close[benchmark].dropna()
                if len(spy_series) > trading_days:
                    spy_returns[weeks] = float(
                        spy_series.iloc[-1] / spy_series.iloc[-trading_days] - 1
                    )
                else:
                    spy_returns[weeks] = 0.0
            else:
                spy_returns[weeks] = 0.0

        # Score each sector
        results: List[Dict[str, Any]] = []
        for sector_name, sector_info in SECTOR_MAP.items():
            etf = sector_info["etf"]
            if etf not in close.columns:
                logger.warning("Skipping sector %s — no data for ETF %s", sector_name, etf)
                continue

            etf_series = close[etf].dropna()
            if len(etf_series) < 10:
                logger.warning("Skipping sector %s — insufficient data for ETF %s", sector_name, etf)
                continue

            sector_returns: Dict[str, float] = {}
            relative_returns: Dict[str, float] = {}
            composite = 0.0

            for weeks, weight in zip(lookback_weeks, weights):
                trading_days = weeks * 5
                if len(etf_series) > trading_days:
                    ret = float(etf_series.iloc[-1] / etf_series.iloc[-trading_days] - 1)
                else:
                    ret = 0.0
                sector_returns[f"{weeks}w"] = ret
                rel = ret - spy_returns.get(weeks, 0.0)
                relative_returns[f"{weeks}w"] = rel
                composite += weight * rel

            results.append({
                "sector": sector_name,
                "etf": etf,
                "composite_score": round(composite, 6),
                "returns": sector_returns,
                "relative_returns": relative_returns,
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        logger.info("Sector scan complete: top sector = %s", results[0]["sector"] if results else "N/A")
        return results

    # ------------------------------------------------------------------
    # Stock-level picking within a sector
    # ------------------------------------------------------------------

    def pick_stocks(
        self,
        sector: str,
        top_n: int = 5,
        include_sentiment: bool = True,
    ) -> List[Dict[str, Any]]:
        """Pick the best stocks within a sector.

        Scoring factors (equal weight):
        - 30-day momentum
        - Volume ratio (current vs 20-day SMA)
        - RSI proximity to 50 (sweet spot)
        - News sentiment

        Returns sorted list of stock dicts.
        """
        if sector not in SECTOR_MAP:
            logger.error("Unknown sector: %s", sector)
            return []

        tickers = SECTOR_MAP[sector]["stocks"]
        min_volume = self.cfg["min_volume"]

        mom_days = self.cfg.get("stock_momentum_days", 60)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=mom_days + 30)  # extra buffer for indicators

        logger.info("Picking stocks for sector %s from %d candidates", sector, len(tickers))

        data = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            logger.warning("No stock data returned for sector %s", sector)
            return []

        # Handle single vs multi-ticker
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            volume = data["Volume"]
        else:
            close = data[["Close"]].rename(columns={"Close": tickers[0]})
            volume = data[["Volume"]].rename(columns={"Volume": tickers[0]})

        scored: List[Dict[str, Any]] = []

        for tic in tickers:
            if tic not in close.columns:
                continue

            prices = close[tic].dropna()
            vols = volume[tic].dropna() if tic in volume.columns else None

            if len(prices) < mom_days:
                continue

            # --- Momentum: N-day return (config-driven) ---
            momentum = float(prices.iloc[-1] / prices.iloc[-mom_days] - 1)

            # --- Volume ratio ---
            if vols is not None and len(vols) >= 20:
                current_vol = float(vols.iloc[-1])
                vol_sma_20 = float(vols.iloc[-20:].mean())
                volume_ratio = current_vol / vol_sma_20 if vol_sma_20 > 0 else 1.0
                if current_vol < min_volume:
                    continue  # skip low-volume stocks
            else:
                volume_ratio = 1.0

            # --- RSI (14-period) ---
            rsi = self._compute_rsi(prices, period=14)

            # RSI score: 1.0 when RSI == 50, 0.0 at extremes
            rsi_score = 1.0 - abs(rsi - 50.0) / 50.0

            # --- News sentiment ---
            sentiment = 0.0
            if include_sentiment:
                sentiment = self._get_news_sentiment(tic)

            scored.append({
                "ticker": tic,
                "price": round(float(prices.iloc[-1]), 2),
                "momentum": round(momentum, 4),
                "volume_ratio": round(volume_ratio, 2),
                "rsi": round(rsi, 1),
                "rsi_score": round(rsi_score, 4),
                "sentiment": round(sentiment, 4),
            })

        if not scored:
            return []

        # Normalise each factor to [0, 1] and compute final score
        scored = self._normalise_and_rank(scored)

        # Sort by final_score descending
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_pipeline(
        self,
        top_sectors: int = 3,
        stocks_per_sector: int = 5,
        include_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete sector scan → stock pick pipeline.

        Returns::

            {
                "scan_time": "2024-...",
                "benchmark": "SPY",
                "sector_rankings": [...],
                "top_sectors": [...],
                "recommended_tickers": [
                    {"sector": "...", "ticker": "...", "final_score": 0.8, ...}, ...
                ],
            }
        """
        top_sectors = top_sectors or self.cfg["top_sectors"]
        stocks_per_sector = stocks_per_sector or self.cfg["stocks_per_sector"]

        # Step 0: Detect market regime (arXiv 2601.19504)
        regime_result = self.regime_detector.detect()
        regime_adjustments = self.regime_detector.get_adjustments(regime_result)
        logger.info(
            "Regime: %s (confidence=%.1f%%, exposure=%.0f%%)",
            regime_result["regime_label"],
            regime_result["confidence"] * 100,
            regime_adjustments["exposure_scale"] * 100,
        )

        # Step 1: Scan sectors
        sector_rankings = self.scan_sectors()

        min_required = min(top_sectors, 3)
        if len(sector_rankings) < min_required:
            raise ValueError(
                f"Only {len(sector_rankings)} valid sectors found (minimum {min_required} required). "
                "Market data may be unavailable."
            )

        # Step 1.5: Apply regime-based sector adjustment
        sector_rankings = self._apply_regime_bias(sector_rankings, regime_adjustments)

        top = sector_rankings[:top_sectors]

        # Step 2: Pick stocks from each top sector
        all_picks: List[Dict[str, Any]] = []
        for sector_info in top:
            sector_name = sector_info["sector"]
            picks = self.pick_stocks(
                sector_name,
                top_n=stocks_per_sector,
                include_sentiment=include_sentiment,
            )
            for pick in picks:
                pick["sector"] = sector_name
                pick["sector_score"] = sector_info["composite_score"]
            all_picks.extend(picks)

        # Sort all picks by final_score
        all_picks.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return {
            "scan_time": datetime.now().isoformat(),
            "benchmark": self.cfg["benchmark"],
            "regime": {
                "state": regime_result["regime"],
                "label": regime_result["regime_label"],
                "confidence": regime_result["confidence"],
                "volatility": regime_result["volatility"],
                "exposure_scale": regime_adjustments["exposure_scale"],
                "description": regime_adjustments["description"],
            },
            "sector_rankings": sector_rankings,
            "top_sectors": [
                {
                    "sector": s["sector"],
                    "etf": s["etf"],
                    "composite_score": s["composite_score"],
                }
                for s in top
            ],
            "recommended_tickers": all_picks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_regime_bias(
        rankings: List[Dict[str, Any]],
        adjustments: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Adjust sector scores based on market regime.

        In high-vol regime: boost defensive sectors, penalise speculative.
        In low-vol regime: no adjustment (momentum signal is trusted).
        """
        regime = adjustments.get("regime", "low_volatility")
        if regime == "low_volatility":
            return rankings  # trust pure momentum in calm markets

        defensive_bias = adjustments.get("defensive_bias", 0.0)
        if defensive_bias <= 0:
            return rankings

        adjusted = []
        for r in rankings:
            entry = dict(r)
            sector_name = entry["sector"]
            trait = SECTOR_TRAITS.get(sector_name, "cyclical")

            # Apply bias based on sector trait
            if trait == "defensive":
                entry["composite_score"] += defensive_bias
            elif trait == "speculative":
                entry["composite_score"] -= defensive_bias * 1.5
            elif trait == "growth":
                entry["composite_score"] -= defensive_bias * 0.5
            # cyclical: no adjustment

            entry["regime_adjusted"] = True
            adjusted.append(entry)

        adjusted.sort(key=lambda x: x["composite_score"], reverse=True)
        return adjusted

    @staticmethod
    def _compute_rsi(prices, period: int = 14) -> float:
        """Compute RSI for the last value of a price series."""
        if len(prices) < period + 1:
            return 50.0  # neutral default

        deltas = prices.diff().iloc[1:]
        gains = deltas.clip(lower=0)
        losses = (-deltas.clip(upper=0))

        avg_gain = float(gains.iloc[-period:].mean())
        avg_loss = float(losses.iloc[-period:].mean())

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _get_news_sentiment(self, ticker: str) -> float:
        """Fetch recent news headlines for a ticker and score sentiment.

        Uses FinBERT (arXiv 2502.14897) when available, falls back to
        keyword scorer. FinBERT has demonstrated +11% accuracy vs keywords.
        """
        try:
            tic_obj = yf.Ticker(ticker)
            news = tic_obj.news
            if not news:
                return 0.0

            headlines: List[str] = []
            for item in news[:5]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)

            if not headlines:
                return 0.0

            # Prefer FinBERT batch scoring (much more accurate)
            if self.finbert_scorer is not None:
                scores = self.finbert_scorer.score_batch(headlines)
            else:
                scores = [self.sentiment_scorer.score_text(h) for h in headlines]

            return float(np.mean(scores))

        except Exception as exc:
            logger.debug("Failed to get news for %s: %s", ticker, exc)
            return 0.0

    def _normalise_and_rank(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise factors to [0,1] and compute weighted final_score.

        Uses XGBoost-learned weights if available (arXiv 2508.18592),
        falls back to equal weighting otherwise.
        """
        if not stocks:
            return stocks

        # Get learned weights (or equal fallback)
        weights = self.stock_ranker.get_factor_weights()

        factors = ["momentum", "volume_ratio", "rsi_score", "sentiment"]

        for factor in factors:
            vals = [s[factor] for s in stocks]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin
            for s in stocks:
                if rng > 0:
                    s[f"{factor}_norm"] = (s[factor] - vmin) / rng
                else:
                    s[f"{factor}_norm"] = 0.5

        for s in stocks:
            weighted_sum = sum(
                weights.get(f, 0.25) * s[f"{f}_norm"]
                for f in factors
            )
            s["final_score"] = round(weighted_sum, 4)
            s["factor_weights_used"] = weights

        return stocks


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scanner = SectorScanner()

    print("\n=== Sector Rankings ===")
    rankings = scanner.scan_sectors()
    for i, r in enumerate(rankings, 1):
        print(f"  {i:2d}. {r['sector']:25s} ETF={r['etf']:5s}  score={r['composite_score']:+.4f}  {r['returns']}")

    print(f"\n=== Top 3 Sectors — Stock Picks ===")
    for sector_info in rankings[:3]:
        name = sector_info["sector"]
        print(f"\n--- {name} (score={sector_info['composite_score']:+.4f}) ---")
        picks = scanner.pick_stocks(name, top_n=5)
        for p in picks:
            print(
                f"  {p['ticker']:6s}  ${p['price']:>8.2f}  "
                f"mom={p['momentum']:+.2%}  vol_r={p['volume_ratio']:.1f}  "
                f"RSI={p['rsi']:.0f}  sent={p['sentiment']:+.2f}  "
                f"SCORE={p['final_score']:.3f}"
            )
