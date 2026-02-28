"""FRED-based macro regime detector — All-Weather classification.

Inspired by Bridgewater All-Weather and IQC 2025 winner strategy.
Classifies macro environment into regimes based on growth + inflation + liquidity:

    GOLDILOCKS  : Growth rising + Inflation falling  → Risk-on (crypto aggressive)
    REFLATION   : Growth rising + Inflation rising   → Mixed (neutral)
    STAGFLATION : Growth falling + Inflation rising  → Risk-off (crypto defensive)
    DEFLATION   : Growth falling + Inflation falling → Easing expected (neutral-positive)

FRED indicators:
    Growth:    T10Y2Y (yield curve), UNRATE (unemployment, inverted)
    Inflation: T10YIE (10Y breakeven inflation), CPIAUCSL (CPI YoY)
    Liquidity: WALCL (Fed balance sheet), WTREGEN (TGA), RRPONTSYD (reverse repo)
    Financial: BAMLH0A0HYM2 (HY spread), NFCI (financial conditions)
Non-FRED:
    DXY:       yfinance DX-Y.NYB (dollar index, BTC inverse correlation -0.4~-0.8)

Updates: Daily (FRED), 6h cache.
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
    # Growth
    "yield_curve": "T10Y2Y",           # 10Y-2Y spread (daily, leading indicator)
    "unemployment": "UNRATE",          # Unemployment rate (monthly)
    # Inflation
    "breakeven_10y": "T10YIE",         # 10Y breakeven inflation (daily)
    "cpi": "CPIAUCSL",                 # CPI-U All Items (monthly)
    # Liquidity (Fed Net Liquidity = WALCL - WTREGEN - RRPONTSYD)
    "fed_balance_sheet": "WALCL",      # Fed total assets (weekly, Wed)
    "tga": "WTREGEN",                  # Treasury General Account (weekly)
    "reverse_repo": "RRPONTSYD",       # Overnight reverse repo (daily)
    # Financial conditions
    "hy_spread": "BAMLH0A0HYM2",      # ICE BofA HY spread (daily)
    "nfci": "NFCI",                    # Chicago Fed NFCI (weekly, negative=loose)
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
        if time.time() - self._cache_time < self._cache_ttl and self._current_regime != MacroRegime.UNKNOWN:
            return self._current_regime

        try:
            # Core 4-quadrant (needs FRED API key)
            if self.available:
                growth_dir = self._assess_growth()
                inflation_dir = self._assess_inflation()
                self._current_regime = self._classify(growth_dir, inflation_dir)
                # Additional signals
                self._assess_liquidity()
                self._assess_financial_conditions()

            # DXY — no API key needed (yfinance)
            self._assess_dxy()

            self._cache_time = time.time()
            liq = self._regime_scores.get("net_liquidity_direction", "N/A")
            dxy = self._regime_scores.get("dxy_direction", "N/A")
            logger.info(
                "[macro] Regime: %s (growth=%s, inflation=%s, liquidity=%s, dxy=%s)",
                self._current_regime.value,
                self._regime_scores.get("growth", "N/A"),
                self._regime_scores.get("inflation", "N/A"),
                liq, dxy,
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

    def _assess_liquidity(self) -> None:
        """Assess Fed Net Liquidity direction (WALCL - WTREGEN - RRPONTSYD).

        BTC correlates 0.6~0.9 with global liquidity, 70-107 day lag.
        Rising net liquidity = bullish for crypto.
        """
        try:
            walcl = self._fetch_series_values("fed_balance_sheet", count=14)
            tga = self._fetch_series_values("tga", count=14)
            rrp = self._fetch_series_values("reverse_repo", count=14)

            if walcl and tga and rrp:
                # Latest net liquidity
                net_now = walcl[0] - tga[0] - rrp[0]
                # ~2 weeks ago (approximate, weekly data)
                net_prev = walcl[-1] - tga[-1] - rrp[-1]
                delta_pct = (net_now - net_prev) / abs(net_prev) if net_prev != 0 else 0

                self._regime_scores["net_liquidity"] = net_now
                self._regime_scores["net_liquidity_delta_pct"] = round(delta_pct, 4)
                self._regime_scores["net_liquidity_direction"] = "expanding" if delta_pct > 0.005 else ("contracting" if delta_pct < -0.005 else "flat")
                logger.info("[macro] Net Liquidity: $%.1fT (Δ%.2f%%, %s)",
                            net_now / 1e6, delta_pct * 100,
                            self._regime_scores["net_liquidity_direction"])
            else:
                self._regime_scores["net_liquidity_direction"] = "unknown"
        except Exception as exc:
            logger.debug("[macro] Liquidity assessment failed: %s", exc)
            self._regime_scores["net_liquidity_direction"] = "unknown"

    def _assess_financial_conditions(self) -> None:
        """Assess financial stress from HY spread and NFCI.

        HY spread widening = risk-off = bearish crypto.
        NFCI negative = loose conditions = bullish crypto.
        """
        try:
            hy_delta = self._fetch_series_delta("hy_spread", lookback=30)
            nfci_val = self._fetch_series_values("nfci", count=2)

            if hy_delta is not None:
                # HY spread widening > 0.5pp in 30 days = stress
                stress = "elevated" if hy_delta > 0.5 else ("easing" if hy_delta < -0.3 else "normal")
                self._regime_scores["hy_spread_delta"] = round(hy_delta, 3)
                self._regime_scores["financial_stress"] = stress
                logger.info("[macro] HY spread Δ: %.2f bps (%s)", hy_delta * 100, stress)

            if nfci_val and len(nfci_val) >= 1:
                nfci = nfci_val[0]
                self._regime_scores["nfci"] = round(nfci, 3)
                # NFCI: negative = loose, positive = tight
                self._regime_scores["nfci_condition"] = "tight" if nfci > 0 else "loose"
        except Exception as exc:
            logger.debug("[macro] Financial conditions failed: %s", exc)

    def _assess_dxy(self) -> None:
        """Assess DXY (US Dollar Index) via yfinance. No API key needed.

        BTC-DXY correlation: -0.4 to -0.8. Dollar up = BTC down.
        """
        try:
            import yfinance as yf
            dxy = yf.download("DX-Y.NYB", period="3mo", interval="1d", progress=False)
            if dxy is None or dxy.empty:
                self._regime_scores["dxy_direction"] = "unknown"
                return

            # Handle yfinance multi-level columns (newer versions)
            if hasattr(dxy.columns, "levels") and len(dxy.columns.levels) > 1:
                close = dxy["Close"].iloc[:, 0].dropna()
            else:
                close = dxy["Close"].dropna()
            if len(close) < 10:
                self._regime_scores["dxy_direction"] = "unknown"
                return

            latest = float(close.iloc[-1])
            month_ago = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
            week_ago = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])

            delta_month_pct = (latest - month_ago) / month_ago
            delta_week_pct = (latest - week_ago) / week_ago

            self._regime_scores["dxy"] = round(latest, 2)
            self._regime_scores["dxy_1m_pct"] = round(delta_month_pct * 100, 2)
            self._regime_scores["dxy_1w_pct"] = round(delta_week_pct * 100, 2)

            if delta_month_pct > 0.02:
                direction = "strengthening"  # bearish for BTC
            elif delta_month_pct < -0.02:
                direction = "weakening"  # bullish for BTC
            else:
                direction = "stable"

            self._regime_scores["dxy_direction"] = direction
            logger.info("[macro] DXY: %.1f (1m: %+.1f%%, 1w: %+.1f%%, %s → BTC %s)",
                        latest, delta_month_pct * 100, delta_week_pct * 100,
                        direction,
                        "bearish" if direction == "strengthening" else "bullish" if direction == "weakening" else "neutral")
        except Exception as exc:
            logger.debug("[macro] DXY fetch failed: %s", exc)
            self._regime_scores["dxy_direction"] = "unknown"

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

    def _fetch_series_values(self, series_key: str, count: int = 10) -> Optional[list]:
        """Fetch latest N values from a FRED series. Returns list [newest, ..., oldest]."""
        series_id = SERIES.get(series_key)
        if not series_id or not self._api_key:
            return None

        cache_key = f"{series_id}_vals_{count}"
        if cache_key in self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache[cache_key]

        try:
            import requests
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": count + 10,
            }
            resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            valid = [float(o["value"]) for o in obs if o["value"] != "."][:count]
            if valid:
                self._cache[cache_key] = valid
            return valid or None
        except Exception as exc:
            logger.debug("[macro] FRED fetch values failed for %s: %s", series_id, exc)
            return None

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
        """Return current macro context for pipeline/snapshot injection."""
        scores = dict(self._regime_scores)
        return {
            "macro_regime": self._current_regime.value,
            "exposure_scale": self.exposure_scale,
            "trail_multiplier": self.trail_multiplier,
            "scores": scores,
            "available": self.available,
            "cache_age_s": time.time() - self._cache_time if self._cache_time > 0 else -1,
            # New macro signals
            "dxy": scores.get("dxy", None),
            "dxy_direction": scores.get("dxy_direction", "unknown"),
            "dxy_1m_pct": scores.get("dxy_1m_pct", None),
            "net_liquidity_direction": scores.get("net_liquidity_direction", "unknown"),
            "net_liquidity_delta_pct": scores.get("net_liquidity_delta_pct", None),
            "financial_stress": scores.get("financial_stress", "unknown"),
            "nfci": scores.get("nfci", None),
        }
