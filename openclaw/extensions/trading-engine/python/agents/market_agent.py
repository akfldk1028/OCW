"""Market structure analyst agent — regime + sector momentum.

Wraps the existing RegimeDetector and SectorScanner into an agent
interface that publishes results to the EventBus.

Responsibilities:
    - HMM regime detection (low_vol / high_vol)
    - 14 sector ETF momentum scanning
    - Defensive/offensive sector bias based on regime
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus

logger = logging.getLogger("trading-engine.agents.market")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="market-agent")


@dataclass
class MarketView:
    """Output of MarketAgent.analyze()."""

    regime: str            # "low_volatility" or "high_volatility"
    regime_confidence: float
    volatility: float
    exposure_scale: float  # 1.0 for low_vol, 0.7 for high_vol
    top_sectors: List[Dict[str, Any]]
    sector_scores: List[Dict[str, Any]]
    adjustments: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "volatility": self.volatility,
            "exposure_scale": self.exposure_scale,
            "top_sectors": self.top_sectors,
            "sector_scores": self.sector_scores,
            "adjustments": self.adjustments,
            "timestamp": self.timestamp,
        }


class MarketAgent:
    """Market structure analyst — regime + sector momentum.

    Reuses:
        - ``RegimeDetector.detect()`` for HMM regime
        - ``SectorScanner.scan_sectors()`` for sector momentum
        - ``RegimeDetector.get_adjustments()`` for strategy bias

    Usage::

        agent = MarketAgent(regime_detector, sector_scanner, event_bus)
        view = await agent.analyze(top_n=3)
    """

    def __init__(
        self,
        regime_detector: Any,
        sector_scanner: Any,
        event_bus: EventBus,
    ) -> None:
        self._regime = regime_detector
        self._scanner = sector_scanner
        self._bus = event_bus
        self._last_view: Optional[MarketView] = None

    async def analyze(self, top_n: int = 3) -> MarketView:
        """Run full market structure analysis.

        1. HMM regime detection
        2. 14-sector ETF momentum scan
        3. Apply defensive/offensive bias

        Publishes ``market.regime`` and ``market.sector`` events.
        """
        loop = asyncio.get_running_loop()

        # Run CPU-heavy analysis in thread pool
        regime_result, sector_rankings = await asyncio.gather(
            loop.run_in_executor(_executor, self._regime.detect),
            loop.run_in_executor(_executor, self._scanner.scan_sectors),
        )

        adjustments = self._regime.get_adjustments(regime_result)

        # Apply regime bias to sector scores
        biased_sectors = self._apply_regime_bias(sector_rankings, adjustments)
        top_sectors = biased_sectors[:top_n]

        view = MarketView(
            regime=regime_result.get("regime_label", "unknown"),
            regime_confidence=regime_result.get("confidence", 0.0),
            volatility=regime_result.get("volatility", 0.0),
            exposure_scale=adjustments.get("exposure_scale", 1.0),
            top_sectors=top_sectors,
            sector_scores=biased_sectors,
            adjustments=adjustments,
            timestamp=time.time(),
        )

        self._last_view = view

        # Publish events
        await self._bus.publish("market.regime", {
            "regime": view.regime,
            "confidence": view.regime_confidence,
            "volatility": view.volatility,
            "exposure_scale": view.exposure_scale,
            "adjustments": adjustments,
        })
        await self._bus.publish("market.sector", {
            "top_sectors": [s.get("sector", s.get("etf", "?")) for s in top_sectors],
            "scores": top_sectors,
            "regime": view.regime,
        })

        logger.info(
            "MarketAgent: regime=%s (%.0f%%), top=%s",
            view.regime,
            view.regime_confidence * 100,
            [s.get("sector", "?") for s in top_sectors],
        )

        return view

    @property
    def last_view(self) -> Optional[MarketView]:
        return self._last_view

    def _apply_regime_bias(
        self,
        sector_rankings: List[Dict[str, Any]],
        adjustments: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Adjust sector scores based on regime (defensive/offensive bias).

        Mirrors SectorScanner.full_pipeline() regime adjustment logic.
        """
        defensive_bias = adjustments.get("defensive_bias", 0.0)
        if defensive_bias == 0.0:
            return sector_rankings

        # Sector trait mapping (same as sector_scanner.py)
        traits = {
            "Technology": "growth",
            "Semiconductors": "growth",
            "Semis": "growth",
            "Financials": "growth",
            "Healthcare": "defensive",
            "Consumer Discretionary": "growth",
            "ConsDisc": "growth",
            "Communication": "growth",
            "Industrials": "growth",
            "Energy": "growth",
            "Consumer Staples": "defensive",
            "Staples": "defensive",
            "Biotech": "speculative",
            "Innovation": "speculative",
            "Materials": "growth",
            "Utilities": "defensive",
            "RealEstate": "defensive",
            "Real Estate": "defensive",
        }

        biased = []
        for s in sector_rankings:
            entry = dict(s)
            sector_name = entry.get("sector", "")
            trait = traits.get(sector_name, "growth")
            score = entry.get("composite_score", entry.get("score", 0.0))

            if trait == "defensive":
                score += defensive_bias
            elif trait == "speculative":
                score -= 1.5 * defensive_bias
            elif trait == "growth":
                score -= 0.5 * defensive_bias

            entry["composite_score"] = score
            entry["score"] = score
            entry["regime_bias"] = trait
            biased.append(entry)

        biased.sort(key=lambda x: x.get("score", x.get("composite_score", 0)), reverse=True)
        return biased
