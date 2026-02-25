"""LLM-based and keyword-fallback sentiment scoring.

Provides a ``SentimentScorer`` class that can score arbitrary text for
financial sentiment.  When an LLM gateway is available it delegates to
it via HTTP; otherwise it falls back to a curated keyword dictionary.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Keyword dictionaries  (used by the fallback scorer)
# -----------------------------------------------------------------------

_POSITIVE_WORDS: Dict[str, float] = {
    # strong positive
    "surge": 0.8, "surges": 0.8, "surging": 0.8,
    "soar": 0.8, "soars": 0.8, "soaring": 0.8,
    "rally": 0.7, "rallies": 0.7, "rallying": 0.7,
    "boom": 0.7, "booming": 0.7,
    "breakout": 0.7, "breakthrough": 0.7,
    "skyrocket": 0.9, "skyrockets": 0.9,
    "outperform": 0.6, "outperforms": 0.6, "outperformed": 0.6,
    "upgrade": 0.6, "upgraded": 0.6, "upgrades": 0.6,
    "bullish": 0.7,
    "record": 0.5, "record-high": 0.7,
    "beat": 0.5, "beats": 0.5, "beating": 0.5,
    "exceed": 0.5, "exceeds": 0.5, "exceeded": 0.5,
    # moderate positive
    "gain": 0.4, "gains": 0.4, "gained": 0.4,
    "rise": 0.4, "rises": 0.4, "rising": 0.4, "rose": 0.4,
    "up": 0.2, "higher": 0.3, "high": 0.2,
    "grow": 0.4, "grows": 0.4, "growth": 0.4, "growing": 0.4,
    "profit": 0.4, "profits": 0.4, "profitable": 0.4,
    "revenue": 0.3, "earnings": 0.3,
    "strong": 0.3, "strength": 0.3,
    "positive": 0.4, "optimistic": 0.5, "optimism": 0.5,
    "recover": 0.4, "recovery": 0.4, "recovering": 0.4,
    "improve": 0.3, "improves": 0.3, "improved": 0.3, "improvement": 0.3,
    "dividend": 0.3, "buyback": 0.4,
    "innovation": 0.3, "innovative": 0.3,
    "success": 0.4, "successful": 0.4,
    "expansion": 0.3, "expand": 0.3, "expanding": 0.3,
    "opportunity": 0.3, "opportunities": 0.3,
    "momentum": 0.3,
}

_NEGATIVE_WORDS: Dict[str, float] = {
    # strong negative
    "crash": -0.9, "crashes": -0.9, "crashing": -0.9,
    "plunge": -0.8, "plunges": -0.8, "plunging": -0.8,
    "collapse": -0.8, "collapses": -0.8, "collapsed": -0.8,
    "bankrupt": -0.9, "bankruptcy": -0.9,
    "fraud": -0.9, "fraudulent": -0.9,
    "scandal": -0.8,
    "default": -0.7, "defaults": -0.7, "defaulted": -0.7,
    "downgrade": -0.6, "downgrades": -0.6, "downgraded": -0.6,
    "bearish": -0.7,
    "recession": -0.7, "recessionary": -0.7,
    "crisis": -0.7,
    # moderate negative
    "loss": -0.5, "losses": -0.5,
    "decline": -0.5, "declines": -0.5, "declining": -0.5, "declined": -0.5,
    "drop": -0.4, "drops": -0.4, "dropped": -0.4, "dropping": -0.4,
    "fall": -0.4, "falls": -0.4, "falling": -0.4, "fell": -0.4,
    "down": -0.2, "lower": -0.3, "low": -0.2,
    "miss": -0.5, "misses": -0.5, "missed": -0.5,
    "weak": -0.4, "weakness": -0.4, "weaken": -0.4,
    "negative": -0.4, "pessimistic": -0.5, "pessimism": -0.5,
    "risk": -0.3, "risks": -0.3, "risky": -0.4,
    "concern": -0.3, "concerns": -0.3, "concerned": -0.3,
    "volatile": -0.3, "volatility": -0.3,
    "sell": -0.3, "selloff": -0.6, "sell-off": -0.6,
    "underperform": -0.5, "underperforms": -0.5,
    "layoff": -0.5, "layoffs": -0.5,
    "cut": -0.3, "cuts": -0.3, "cutting": -0.3,
    "debt": -0.3,
    "lawsuit": -0.5, "litigation": -0.4,
    "warning": -0.4, "warn": -0.4, "warns": -0.4,
    "inflation": -0.3, "inflationary": -0.3,
    "tariff": -0.3, "tariffs": -0.3,
    "uncertainty": -0.4,
    "investigation": -0.5,
}

# Merge into a single lookup for fast access
_WORD_SCORES: Dict[str, float] = {**_POSITIVE_WORDS, **_NEGATIVE_WORDS}

# Negation words that flip the sign of the next sentiment word
_NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely", "don't", "doesn't", "didn't", "wasn't", "weren't", "isn't", "aren't", "won't", "wouldn't", "shouldn't", "couldn't", "can't", "cannot"}

# Intensifiers that amplify the score
_INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "highly": 1.3, "significantly": 1.3, "sharply": 1.4, "dramatically": 1.5, "massively": 1.5, "strongly": 1.3}


class SentimentScorer:
    """Score financial text for sentiment on a -1 (bearish) to +1 (bullish)
    scale.

    Parameters
    ----------
    gateway_url : str or None
        URL of an OpenClaw LLM gateway endpoint.  When provided the
        scorer will attempt an HTTP POST to ``{gateway_url}/sentiment``
        with a JSON body ``{"text": "..."}`` and expect a JSON response
        ``{"score": float}``.  Falls back to keyword scoring on failure.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self.gateway_url = gateway_url
        self.timeout = timeout

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def score_text(self, text: str) -> float:
        """Return a sentiment score in [-1, 1] for the given *text*.

        Tries the LLM gateway first (if configured), falls back to the
        keyword-based scorer.
        """
        if self.gateway_url:
            llm_score = self._score_via_llm(text)
            if llm_score is not None:
                return llm_score

        return self._keyword_score(text)

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts and return a list of scores."""
        if self.gateway_url:
            llm_scores = self._score_batch_via_llm(texts)
            if llm_scores is not None:
                return llm_scores

        return [self._keyword_score(t) for t in texts]

    # ----------------------------------------------------------------
    # Keyword-based fallback scorer
    # ----------------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str) -> float:
        """Compute a sentiment score purely from keyword matching.

        The algorithm:
        1. Tokenise into lowercase words.
        2. Walk through the token list maintaining a context window for
           negation and intensifiers.
        3. Accumulate weighted scores; return the ``tanh``-squashed mean
           so the result stays within [-1, 1].
        """
        tokens = re.findall(r"[a-z'\-]+", text.lower())
        if not tokens:
            return 0.0

        total_score = 0.0
        hit_count = 0
        negate = False
        intensifier = 1.0

        for token in tokens:
            if token in _NEGATORS:
                negate = True
                continue

            if token in _INTENSIFIERS:
                intensifier = _INTENSIFIERS[token]
                continue

            if token in _WORD_SCORES:
                raw = _WORD_SCORES[token] * intensifier
                if negate:
                    raw *= -0.75  # partial flip -- negation weakens rather than fully reversing
                total_score += raw
                hit_count += 1

            # Reset modifiers after consuming a content word
            negate = False
            intensifier = 1.0

        if hit_count == 0:
            return 0.0

        mean_score = total_score / hit_count
        # Squash through tanh to keep within [-1, 1]
        return float(math.tanh(mean_score))

    # ----------------------------------------------------------------
    # LLM gateway helpers
    # ----------------------------------------------------------------

    def _score_via_llm(self, text: str) -> Optional[float]:
        """Attempt to score a single text via the OpenClaw LLM gateway."""
        try:
            import httpx  # type: ignore[import-untyped]

            resp = httpx.post(
                f"{self.gateway_url}/sentiment",
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            score = float(resp.json()["score"])
            return max(-1.0, min(1.0, score))
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM gateway call failed, falling back to keywords: %s", exc)
            return None

    def _score_batch_via_llm(self, texts: List[str]) -> Optional[List[float]]:
        """Attempt to score a batch of texts via the LLM gateway."""
        try:
            import httpx  # type: ignore[import-untyped]

            resp = httpx.post(
                f"{self.gateway_url}/sentiment/batch",
                json={"texts": texts},
                timeout=self.timeout * 2,
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
            return [max(-1.0, min(1.0, float(s))) for s in scores]
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM batch gateway call failed, falling back to keywords: %s", exc)
            return None
