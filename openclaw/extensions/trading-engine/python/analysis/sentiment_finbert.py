"""FinBERT-based financial sentiment scoring.

Replaces keyword-based sentiment (which has 0% predictive power per
XGBoost feature importance) with a transformer model fine-tuned on
financial text.

Paper backing:
- arXiv 2502.14897: market-derived sentiment labels +11% accuracy
- ProsusAI/finbert: pre-trained on Financial PhraseBank (4,840 sentences)

The model classifies text into positive/negative/neutral and returns
a continuous score in [-1, 1].

Usage::

    from sentiment_finbert import FinBERTScorer
    scorer = FinBERTScorer()
    score = scorer.score_text("NVIDIA beats earnings expectations")  # ~0.85
    scores = scorer.score_batch(["stock rallies", "market crashes"])
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Model identifier on HuggingFace
_MODEL_NAME = "ProsusAI/finbert"


class FinBERTScorer:
    """FinBERT-based financial sentiment scorer.

    Lazy-loads the model on first use to avoid slow startup.
    Thread-safe for inference (model is in eval mode).
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._tokenizer = None
        self._model = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Lazy-load the FinBERT model."""
        if self._loaded:
            return True

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Loading FinBERT model (%s) on %s...", _MODEL_NAME, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info("FinBERT loaded successfully")
            return True

        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            return False

    def score_text(self, text: str) -> float:
        """Score a single text for financial sentiment.

        Returns a float in [-1, 1]:
          -1.0 = strongly negative
           0.0 = neutral
          +1.0 = strongly positive
        """
        if not self._ensure_loaded():
            return 0.0

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # FinBERT outputs: [positive, negative, neutral]
            probs = torch.softmax(logits, dim=-1).squeeze()

            # Convert to [-1, 1] score
            # score = P(positive) - P(negative)
            score = float(probs[0] - probs[1])
            return max(-1.0, min(1.0, score))

        except Exception as exc:
            logger.debug("FinBERT scoring failed for text: %s", exc)
            return 0.0

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts efficiently in a batch."""
        if not texts:
            return []

        if not self._ensure_loaded():
            return [0.0] * len(texts)

        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            # score = P(positive) - P(negative) for each
            scores = (probs[:, 0] - probs[:, 1]).cpu().tolist()
            return [max(-1.0, min(1.0, s)) for s in scores]

        except Exception as exc:
            logger.debug("FinBERT batch scoring failed: %s", exc)
            return [0.0] * len(texts)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scorer = FinBERTScorer()

    test_headlines = [
        "NVIDIA beats earnings expectations, stock surges 15%",
        "Federal Reserve signals potential rate cuts in 2025",
        "Tech stocks plunge amid recession fears",
        "Apple reports record iPhone sales in holiday quarter",
        "Oil prices crash as demand outlook weakens",
        "Semiconductor sector shows strong momentum",
        "Market volatility increases as geopolitical tensions rise",
        "Amazon announces massive layoffs affecting 10,000 workers",
        "Bitcoin rally continues as institutional adoption grows",
        "Goldman Sachs downgrades banking sector outlook",
    ]

    print(f"\n{'='*70}")
    print(f"FINBERT SENTIMENT SCORING TEST")
    print(f"{'='*70}")

    for headline in test_headlines:
        score = scorer.score_text(headline)
        bar = "+" * int(max(0, score) * 20) + "-" * int(max(0, -score) * 20)
        label = "POS" if score > 0.1 else ("NEG" if score < -0.1 else "NEU")
        print(f"  [{label:3s}] {score:+.3f}  {bar:20s}  {headline}")

    # Batch test
    print(f"\nBatch scoring test:")
    batch_scores = scorer.score_batch(test_headlines)
    for headline, score in zip(test_headlines, batch_scores):
        print(f"  {score:+.3f}  {headline[:60]}")
