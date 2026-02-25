"""FRED-based macro regime detector — All-Weather 4-quadrant classification.

Inspired by Bridgewater All-Weather and IQC 2025 winner strategy.
Classifies macro environment into 4 regimes based on growth + inflation direction:

    GOLDILOCKS  : Growth rising + Inflation falling  → Risk-on (crypto aggressive)
    REFLATION   : Growth rising + Inflation rising   → Mixed (neutral)
    STAGFLATION : Growth falling + Inflation rising  → Risk-off (crypto defensive)
    DEFLATION   : Growth falling + Inflation falling → Easing expected (neutral-positive)

FRED indicators used:
    Growth:    T10Y2Y (yield curve), UNRATE (unemployment, inverted)
    Inflation: T10YIE (10Y breakeven inflation), CPIAUCSL (CPI YoY)

Updates: Daily (most FRED data is daily or monthly with daily availability).
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("trading-engine.macro_regime")

FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs
SERIES = {
    "yield_curve": "T10Y2Y",     # 10Y-2Y spread (daily, leading indicator)
    "breakeven_10y": "T10YIE",   # 10Y breakeven inflation (daily)
    "unemployment": "UNRATE",    # Unemployment rate (monthly)
    "cpi": "CPIAUCSL",           # CPI-U All Items (monthly)
}


class MacroRegime(str, Enum):
    GOLDILOCKS = "goldilocks"    # Growth↑ Inflation↓
    REFLATION = "reflation"      # Growth↑ Inflation↑
    STAGFLATION = "stagflation"  # Growth↓ Inflation↑
    DEFLATION = "deflation"      # Growth↓ Inflation↓
    UNKNOWN = "unknown"


# Crypto exposure scaling per regime
REGIME_EXPOSURE = {
    MacroRegime.GOLDILOCKS: 1.2,   # Aggressive: increase position sizes
    MacroRegime.REFLATION: 1.0,    # Neutral: default sizing
    MacroRegime.STAGFLATION: 0.6,  # Defensive: reduce exposure
    MacroRegime.DEFLATION: 0.9,    # Slightly cautious but easing ahead
    MacroRegime.UNKNOWN: 1.0,
}

# Trailing stop tightness per regime
REGIME_TRAIL_MULTIPLIER = {
    MacroRegime.GOLDILOCKS: 1.0,   # Normal trailing
    MacroRegime.REFLATION: 1.0,
    MacroRegime.STAGFLATION: 0.75, # Tighter trails in risk-off
    MacroRegime.DEFLATION: 0.9,
    MacroRegime.UNKNOWN: 1.0,
}


class MacroRegimeDetector:
    """Detect macro regime from FRED economic data.

    Caches results for `cache_ttl` seconds to avoid hammering the API.
    FRED rate limit: 120 requests per minute.
    """

    def __init__(self, cache_ttl: float = 3600.0 * 6) -> None:
        self._api_key = os.environ.get(FRED_API_KEY_ENV, "")
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0.0
        self._current_regime = MacroRegime.UNKNOWN
        self._regime_scores: Dict[str, float] = {}

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def regime(self) -> MacroRegime:
        return self._current_regime

    @property
    def exposure_scale(self) -> float:
        return REGIME_EXPOSURE.get(self._current_regime, 1.0)

    @property
    def trail_multiplier(self) -> float:
        return REGIME_TRAIL_MULTIPLIER.get(self._current_regime, 1.0)

    def detect(self) -> MacroRegime:
        """Fetch FRED data and classify macro regime.

        Returns cached result if within TTL.
        """
        if not self.available:
            logger.debug("[macro] No FRED_API_KEY set, returning UNKNOWN")
            return MacroRegime.UNKNOWN

        if time.time() - self._cache_time < self._cache_ttl and self._current_regime != MacroRegime.UNKNOWN:
            return self._current_regime

        try:
            growth_dir = self._assess_growth()
            inflation_dir = self._assess_inflation()
            self._current_regime = self._classify(growth_dir, inflation_dir)
            self._cache_time = time.time()
            logger.info(
                "[macro] Regime: %s (growth=%s, inflation=%s)",
                self._current_regime.value, growth_dir, inflation_dir,
            )
        except Exception as exc:
            logger.warning("[macro] Detection failed: %s", exc)

        return self._current_regime

    def _assess_growth(self) -> str:
        """Assess growth direction from yield curve and unemployment."""
        yc = self._fetch_series_delta("yield_curve", lookback=90)
        ur = self._fetch_series_delta("unemployment", lookback=90)

        # Yield curve steepening = growth improving
        # Unemployment falling = growth improving (invert)
        growth_score = 0.0
        if yc is not None:
            growth_score += 0.6 * (1.0 if yc > 0.05 else (-1.0 if yc < -0.05 else 0.0))
        if ur is not None:
            growth_score += 0.4 * (-1.0 if ur > 0.1 else (1.0 if ur < -0.1 else 0.0))

        self._regime_scores["growth"] = growth_score
        return "rising" if growth_score > 0 else "falling"

    def _assess_inflation(self) -> str:
        """Assess inflation direction from breakeven rates and CPI."""
        be = self._fetch_series_delta("breakeven_10y", lookback=90)
        cpi = self._fetch_series_delta("cpi", lookback=90)

        inflation_score = 0.0
        if be is not None:
            inflation_score += 0.6 * (1.0 if be > 0.1 else (-1.0 if be < -0.1 else 0.0))
        if cpi is not None:
            # CPI is index level, compute YoY-ish change direction
            inflation_score += 0.4 * (1.0 if cpi > 0.5 else (-1.0 if cpi < -0.5 else 0.0))

        self._regime_scores["inflation"] = inflation_score
        return "rising" if inflation_score > 0 else "falling"

    def _classify(self, growth: str, inflation: str) -> MacroRegime:
        """Map growth/inflation directions to 4-quadrant regime."""
        if growth == "rising" and inflation == "falling":
            return MacroRegime.GOLDILOCKS
        elif growth == "rising" and inflation == "rising":
            return MacroRegime.REFLATION
        elif growth == "falling" and inflation == "rising":
            return MacroRegime.STAGFLATION
        else:
            return MacroRegime.DEFLATION

    def _fetch_series_delta(self, series_key: str, lookback: int = 90) -> Optional[float]:
        """Fetch a FRED series and return the delta over lookback days.

        Returns latest_value - value_N_days_ago, or None on failure.
        """
        series_id = SERIES.get(series_key)
        if not series_id:
            return None

        cache_key = f"{series_id}_{lookback}"
        if cache_key in self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache[cache_key]

        try:
            import requests

            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": lookback + 30,  # extra buffer for weekends/holidays
            }
            resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])

            # Filter out missing values
            valid = [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
            if len(valid) < 2:
                return None

            latest = valid[0][1]
            # Find observation closest to N days ago
            oldest_available = valid[-1][1] if len(valid) <= lookback else valid[min(lookback, len(valid) - 1)][1]
            delta = latest - oldest_available

            self._cache[cache_key] = delta
            return delta
        except Exception as exc:
            logger.debug("[macro] FRED fetch failed for %s: %s", series_id, exc)
            return None

    def get_context(self) -> Dict[str, Any]:
        """Return current macro context for pipeline injection."""
        return {
            "macro_regime": self._current_regime.value,
            "exposure_scale": self.exposure_scale,
            "trail_multiplier": self.trail_multiplier,
            "scores": dict(self._regime_scores),
            "available": self.available,
            "cache_age_s": time.time() - self._cache_time if self._cache_time > 0 else -1,
        }
