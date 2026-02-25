"""Autonomous trading decision engine.

Synthesizes all available signals (regime, sector momentum, stock factors,
FinBERT sentiment, RL model) into actionable buy/sell/hold decisions with
confidence scores and position sizing.

This is the "brain" that answers: "What should I buy/sell right now, and how much?"

Paper backing:
- arXiv 2512.10913: implementation quality > algorithm (167 papers)
- arXiv 2601.19504: regime-adaptive trading +1-4% alpha
- arXiv 2508.18592: multi-factor dynamic weighting
- Moskowitz-Grinblatt 1999: sector momentum 0.43%/month excess

Decision framework:
1. Regime check → sets exposure and strategy mode
2. Sector scan → identifies hot sectors
3. Stock ranking → picks best stocks within sectors
4. Signal aggregation → combines all signals into confidence score
5. Position sizing → Kelly fraction adjusted by confidence
6. Risk check → final guardrails before execution

Usage::

    from auto_trader import AutoTrader
    trader = AutoTrader()
    decisions = trader.decide()
    # Returns list of {ticker, action, confidence, size, reasons}
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from config import RISK_CONFIG, SECTOR_SCAN_CONFIG
from analysis.regime_detector import RegimeDetector
from core.risk_manager import RiskManager
from analysis.sector_scanner import SectorScanner
from analysis.sentiment_finbert import FinBERTScorer
from analysis.sentiment_scorer import SentimentScorer
from analysis.stock_ranker import StockRanker

logger = logging.getLogger(__name__)

# Decision thresholds
CONFIDENCE_THRESHOLDS = {
    "strong_buy": 0.6,
    "buy": 0.35,       # ~sector momentum + stock momentum sufficient
    "hold": 0.15,      # below this = no action
    "sell": -0.2,      # existing positions only
    "strong_sell": -0.4,
}

# Position sizing
MAX_POSITION_PCT = RISK_CONFIG.get("max_position_pct", 0.15)
KELLY_FRACTION = RISK_CONFIG.get("kelly_fraction", 0.25)


class AutoTrader:
    """Autonomous trading decision engine.

    Combines regime detection, sector scanning, stock ranking,
    sentiment analysis, and risk management into automated
    buy/sell/hold decisions.
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        sector_scanner: Optional[SectorScanner] = None,
        risk_manager: Optional[RiskManager] = None,
        stock_ranker: Optional[StockRanker] = None,
    ) -> None:
        self.regime_detector = regime_detector or RegimeDetector()
        self.risk_manager = risk_manager or RiskManager()
        self.stock_ranker = stock_ranker or StockRanker()

        if sector_scanner is not None:
            self.sector_scanner = sector_scanner
        else:
            self.sector_scanner = SectorScanner(
                finbert_scorer=FinBERTScorer(),
                regime_detector=self.regime_detector,
                stock_ranker=self.stock_ranker,
            )

    def decide(
        self,
        portfolio_value: float = 100_000.0,
        current_positions: Optional[Dict[str, float]] = None,
        top_sectors: int = 3,
        stocks_per_sector: int = 5,
        include_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """Make autonomous trading decisions.

        Returns a complete decision package::

            {
                "timestamp": "...",
                "regime": {...},
                "decisions": [
                    {
                        "ticker": "NVDA",
                        "action": "BUY",
                        "confidence": 0.82,
                        "position_size_pct": 0.08,
                        "position_size_usd": 8000,
                        "reasons": ["strong sector momentum", "high volume", ...],
                        "risk_check": {...},
                    },
                    ...
                ],
                "summary": "...",
                "portfolio_allocation": {...},
            }
        """
        current_positions = current_positions or {}
        timestamp = datetime.now().isoformat()

        # ----------------------------------------------------------
        # Step 1: Regime detection
        # ----------------------------------------------------------
        regime_result = self.regime_detector.detect()
        regime_adj = self.regime_detector.get_adjustments(regime_result)
        exposure_scale = regime_adj.get("exposure_scale", 1.0)

        logger.info(
            "AutoTrader: regime=%s confidence=%.0f%% exposure=%.0f%%",
            regime_result["regime_label"],
            regime_result["confidence"] * 100,
            exposure_scale * 100,
        )

        # ----------------------------------------------------------
        # Step 2: Sector scan + stock picks
        # ----------------------------------------------------------
        try:
            scan_result = self.sector_scanner.full_pipeline(
                top_sectors=top_sectors,
                stocks_per_sector=stocks_per_sector,
                include_sentiment=include_sentiment,
            )
        except ValueError as exc:
            logger.error("Sector scan failed: %s", exc)
            return {
                "timestamp": timestamp,
                "regime": regime_result,
                "decisions": [],
                "summary": f"Scan failed: {exc}",
                "error": str(exc),
            }

        recommended = scan_result.get("recommended_tickers", [])

        if not recommended:
            return {
                "timestamp": timestamp,
                "regime": regime_result,
                "decisions": [],
                "summary": "No stocks passed the screening criteria",
            }

        # ----------------------------------------------------------
        # Step 3: Score each candidate and make decisions
        # ----------------------------------------------------------
        decisions: List[Dict[str, Any]] = []

        for stock in recommended:
            ticker = stock["ticker"]
            decision = self._evaluate_stock(
                stock=stock,
                regime_result=regime_result,
                regime_adj=regime_adj,
                portfolio_value=portfolio_value,
                exposure_scale=exposure_scale,
                current_position=current_positions.get(ticker, 0.0),
            )
            if decision is not None:
                decisions.append(decision)

        # ----------------------------------------------------------
        # Step 4: Portfolio-level checks
        # ----------------------------------------------------------
        decisions = self._apply_portfolio_constraints(
            decisions, portfolio_value, exposure_scale, current_positions,
        )

        # Sort by confidence
        decisions.sort(key=lambda x: abs(x["confidence"]), reverse=True)

        # ----------------------------------------------------------
        # Step 5: Generate summary
        # ----------------------------------------------------------
        summary = self._generate_summary(decisions, regime_result, scan_result)

        return {
            "timestamp": timestamp,
            "regime": {
                "state": regime_result["regime_label"],
                "confidence": regime_result["confidence"],
                "volatility": regime_result["volatility"],
                "exposure_scale": exposure_scale,
            },
            "scan_result": {
                "top_sectors": scan_result.get("top_sectors", []),
                "total_candidates": len(recommended),
            },
            "decisions": decisions,
            "summary": summary,
            "portfolio_allocation": self._compute_allocation(
                decisions, portfolio_value, current_positions,
            ),
        }

    # ------------------------------------------------------------------
    # Internal: per-stock evaluation
    # ------------------------------------------------------------------

    def _evaluate_stock(
        self,
        stock: Dict[str, Any],
        regime_result: Dict[str, Any],
        regime_adj: Dict[str, Any],
        portfolio_value: float,
        exposure_scale: float,
        current_position: float,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single stock and produce a buy/sell/hold decision."""

        ticker = stock["ticker"]
        momentum = stock.get("momentum", 0)
        volume_ratio = stock.get("volume_ratio", 1.0)
        rsi = stock.get("rsi", 50)
        sentiment = stock.get("sentiment", 0)
        sector_score = stock.get("sector_score", 0)
        final_score = stock.get("final_score", 0)

        reasons: List[str] = []

        # ---- Signal 1: Sector momentum ----
        sector_signal = 0.0
        if sector_score > 0.05:
            sector_signal = min(sector_score * 5, 1.0)  # normalize to ~[0, 1]
            reasons.append(f"strong sector ({stock.get('sector', '?')}: {sector_score:+.3f})")
        elif sector_score > 0:
            sector_signal = sector_score * 3
            reasons.append(f"neutral sector ({sector_score:+.3f})")
        else:
            sector_signal = max(sector_score * 3, -1.0)
            reasons.append(f"weak sector ({sector_score:+.3f})")

        # ---- Signal 2: Stock momentum ----
        mom_signal = 0.0
        if momentum > 0.10:
            mom_signal = min(momentum * 3, 1.0)
            reasons.append(f"strong momentum ({momentum:+.1%})")
        elif momentum > 0:
            mom_signal = momentum * 2
        elif momentum < -0.05:
            mom_signal = max(momentum * 3, -1.0)
            reasons.append(f"negative momentum ({momentum:+.1%})")

        # ---- Signal 3: Volume confirmation ----
        vol_signal = 0.0
        if volume_ratio > 1.5:
            vol_signal = 0.3  # volume confirming the move
            reasons.append(f"volume surge ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.5:
            vol_signal = -0.2  # suspicious lack of volume
            reasons.append("low volume warning")

        # ---- Signal 4: RSI ----
        rsi_signal = 0.0
        if rsi > 70:
            rsi_signal = -0.3  # overbought
            reasons.append(f"overbought RSI={rsi:.0f}")
        elif rsi < 30:
            rsi_signal = 0.3 if momentum > 0 else -0.3  # oversold bounce or falling knife
            reasons.append(f"oversold RSI={rsi:.0f}")
        elif 40 <= rsi <= 60:
            rsi_signal = 0.1  # healthy zone
        # else neutral

        # ---- Signal 5: Sentiment (FinBERT) ----
        sent_signal = 0.0
        if abs(sentiment) > 0.3:
            sent_signal = sentiment * 0.5
            label = "positive" if sentiment > 0 else "negative"
            reasons.append(f"{label} sentiment ({sentiment:+.2f})")

        # ---- Signal 6: Regime adjustment ----
        regime_signal = 0.0
        if regime_result["regime"] == 1:  # high vol
            regime_signal = -0.2
            reasons.append("high-vol regime (reduced exposure)")

        # ---- Aggregate confidence ----
        # Weighted combination of all signals
        weights = {
            "sector": 0.25,
            "momentum": 0.25,
            "volume": 0.10,
            "rsi": 0.10,
            "sentiment": 0.15,
            "regime": 0.15,
        }
        raw_confidence = (
            weights["sector"] * sector_signal
            + weights["momentum"] * mom_signal
            + weights["volume"] * vol_signal
            + weights["rsi"] * rsi_signal
            + weights["sentiment"] * sent_signal
            + weights["regime"] * regime_signal
        )

        # Clamp to [-1, 1]
        confidence = max(-1.0, min(1.0, raw_confidence))

        # ---- Determine action ----
        if confidence >= CONFIDENCE_THRESHOLDS["strong_buy"]:
            action = "STRONG_BUY"
        elif confidence >= CONFIDENCE_THRESHOLDS["buy"]:
            action = "BUY"
        elif confidence >= CONFIDENCE_THRESHOLDS["hold"]:
            action = "HOLD"
        elif confidence <= CONFIDENCE_THRESHOLDS["strong_sell"]:
            action = "STRONG_SELL"
        elif confidence <= CONFIDENCE_THRESHOLDS["sell"]:
            action = "SELL"
        else:
            action = "HOLD"

        # Skip if no meaningful signal
        if action == "HOLD" and current_position == 0:
            return None

        # ---- Position sizing (half-Kelly) ----
        position_pct = self._compute_position_size(
            confidence=confidence,
            exposure_scale=exposure_scale,
        )
        position_usd = round(portfolio_value * position_pct, 2)

        return {
            "ticker": ticker,
            "action": action,
            "confidence": round(confidence, 4),
            "position_size_pct": round(position_pct, 4),
            "position_size_usd": position_usd,
            "price": stock.get("price", 0),
            "sector": stock.get("sector", ""),
            "reasons": reasons,
            "signals": {
                "sector": round(sector_signal, 3),
                "momentum": round(mom_signal, 3),
                "volume": round(vol_signal, 3),
                "rsi": round(rsi_signal, 3),
                "sentiment": round(sent_signal, 3),
                "regime": round(regime_signal, 3),
            },
        }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_position_size(
        confidence: float,
        exposure_scale: float = 1.0,
    ) -> float:
        """Compute position size using modified Kelly criterion.

        Kelly fraction = edge / odds
        We use quarter-Kelly for safety, scaled by confidence and regime.
        """
        if confidence <= 0:
            return 0.0

        # Approximate edge from confidence
        # confidence 0.5 = ~2.5% expected edge, confidence 1.0 = ~5%
        estimated_edge = confidence * 0.05

        # Quarter-Kelly
        kelly_size = estimated_edge * KELLY_FRACTION

        # Scale by exposure (regime-dependent)
        sized = kelly_size * exposure_scale

        # Cap at max position
        return min(sized, MAX_POSITION_PCT)

    # ------------------------------------------------------------------
    # Portfolio constraints
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_portfolio_constraints(
        decisions: List[Dict[str, Any]],
        portfolio_value: float,
        exposure_scale: float,
        current_positions: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Apply portfolio-level risk constraints."""

        max_total_exposure = RISK_CONFIG.get("max_exposure_pct", 0.80) * exposure_scale
        max_daily_trades = RISK_CONFIG.get("max_daily_trades", 20)

        # Limit number of new trades
        buy_decisions = [d for d in decisions if d["action"] in ("BUY", "STRONG_BUY")]
        if len(buy_decisions) > max_daily_trades:
            buy_decisions = buy_decisions[:max_daily_trades]
            logger.warning("Capped buy decisions at %d (max daily trades)", max_daily_trades)

        # Check total exposure
        total_new_exposure = sum(d["position_size_pct"] for d in buy_decisions)
        existing_exposure = sum(current_positions.values()) / portfolio_value if portfolio_value > 0 else 0

        if total_new_exposure + existing_exposure > max_total_exposure:
            # Scale down proportionally
            available = max(0, max_total_exposure - existing_exposure)
            if total_new_exposure > 0:
                scale = available / total_new_exposure
                for d in buy_decisions:
                    d["position_size_pct"] = round(d["position_size_pct"] * scale, 4)
                    d["position_size_usd"] = round(portfolio_value * d["position_size_pct"], 2)
                    d["reasons"].append(f"position scaled {scale:.0%} (exposure limit)")

        # Keep sell decisions + scaled buy decisions
        sell_decisions = [d for d in decisions if d["action"] in ("SELL", "STRONG_SELL")]
        hold_decisions = [d for d in decisions if d["action"] == "HOLD"]
        return buy_decisions + sell_decisions + hold_decisions

    # ------------------------------------------------------------------
    # Summary & allocation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        decisions: List[Dict[str, Any]],
        regime_result: Dict[str, Any],
        scan_result: Dict[str, Any],
    ) -> str:
        """Generate a human-readable summary of decisions."""
        buys = [d for d in decisions if "BUY" in d["action"]]
        sells = [d for d in decisions if "SELL" in d["action"]]
        holds = [d for d in decisions if d["action"] == "HOLD"]

        top_sectors = [s["sector"] for s in scan_result.get("top_sectors", [])]

        parts = [
            f"Regime: {regime_result['regime_label']} (confidence {regime_result['confidence']:.0%})",
            f"Top sectors: {', '.join(top_sectors)}",
        ]

        if buys:
            tickers = [f"{d['ticker']}({d['confidence']:+.2f})" for d in buys[:5]]
            total_alloc = sum(d["position_size_pct"] for d in buys)
            parts.append(f"BUY {len(buys)} stocks: {', '.join(tickers)} (total alloc {total_alloc:.1%})")

        if sells:
            tickers = [d["ticker"] for d in sells]
            parts.append(f"SELL {len(sells)}: {', '.join(tickers)}")

        if holds:
            parts.append(f"HOLD {len(holds)} positions")

        if not buys and not sells:
            parts.append("No actionable signals")

        return " | ".join(parts)

    @staticmethod
    def _compute_allocation(
        decisions: List[Dict[str, Any]],
        portfolio_value: float,
        current_positions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute target portfolio allocation."""
        buys = {
            d["ticker"]: d["position_size_pct"]
            for d in decisions
            if "BUY" in d["action"]
        }
        sells = {d["ticker"] for d in decisions if "SELL" in d["action"]}

        # Combine with current positions
        target = {}
        for ticker, value in current_positions.items():
            if ticker not in sells:
                target[ticker] = value / portfolio_value if portfolio_value > 0 else 0

        target.update(buys)

        cash_pct = max(0, 1.0 - sum(target.values()))

        return {
            "positions": {k: round(v, 4) for k, v in target.items()},
            "total_invested_pct": round(1.0 - cash_pct, 4),
            "cash_pct": round(cash_pct, 4),
            "num_positions": len(target),
        }


# ------------------------------------------------------------------
# API integration
# ------------------------------------------------------------------

def run_auto_decide(
    portfolio_value: float = 100_000.0,
    current_positions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Entry point for server.py."""
    trader = AutoTrader()
    return trader.decide(
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    trader = AutoTrader()
    result = trader.decide(
        portfolio_value=100_000,
        top_sectors=3,
        stocks_per_sector=5,
        include_sentiment=True,
    )

    print(f"\n{'='*70}")
    print(f"AUTO TRADER DECISION")
    print(f"{'='*70}")
    print(f"Time:    {result['timestamp']}")
    print(f"Regime:  {result['regime']['state']} (confidence {result['regime']['confidence']:.0%})")
    print(f"Summary: {result['summary']}")

    if result.get("decisions"):
        print(f"\n--- Decisions ({len(result['decisions'])}) ---")
        for d in result["decisions"]:
            print(
                f"  [{d['action']:11s}]  {d['ticker']:6s}  "
                f"confidence={d['confidence']:+.3f}  "
                f"size={d['position_size_pct']:.1%} (${d['position_size_usd']:,.0f})  "
                f"price=${d['price']:.2f}"
            )
            for reason in d["reasons"]:
                print(f"              -> {reason}")

    alloc = result.get("portfolio_allocation", {})
    if alloc:
        print(f"\n--- Portfolio Allocation ---")
        print(f"  Positions:  {alloc.get('num_positions', 0)}")
        print(f"  Invested:   {alloc.get('total_invested_pct', 0):.1%}")
        print(f"  Cash:       {alloc.get('cash_pct', 1):.1%}")
