"""
NLI-based factual consistency reward.

Uses a cross-encoder NLI model to score whether each claim in the agent's
response is entailed by the retrieved context. Aggregates per-claim scores
into a single episode reward.

Model: cross-encoder/nli-deberta-v3-base (default)
  — strong NLI performance, runs on CPU for dev, GPU for scale.
  — swap to 'cross-encoder/nli-roberta-base' for lighter weight.

Design note: we score at the sentence level rather than full-response level
because long responses contain both grounded and hallucinated claims; full-
response scoring masks individual claim errors (the same insight behind Lynx's
claim-level decomposition approach).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


_LABEL_ORDER = None  # resolved lazily per model


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class NLIReward:
    """
    Scores factual consistency via NLI entailment probability.

    score(response, context_docs) → float in [0, 1]
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[int] = None,
        batch_size: int = 16,
        aggregation: str = "mean",   # "mean" | "min" | "harmonic"
    ):
        device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,
        )
        self._batch_size = batch_size
        self._aggregation = aggregation
        self._entailment_label = self._resolve_entailment_label()

    def score(self, response: str, context_docs: list[str]) -> float:
        """Return entailment score ∈ [0, 1]. Higher = more grounded."""
        if not response.strip() or not context_docs:
            return 0.0

        context = " ".join(context_docs)
        claims = _split_sentences(response)
        if not claims:
            return 0.0

        pairs = [{"text": context, "text_pair": claim} for claim in claims]
        raw = self._pipe(pairs, batch_size=self._batch_size, truncation=True, padding=True)

        scores = []
        for result in raw:
            label_scores = {r["label"]: r["score"] for r in result}
            entailment_prob = label_scores.get(self._entailment_label, 0.0)
            scores.append(entailment_prob)

        return float(self._aggregate(np.array(scores)))

    def score_with_breakdown(
        self, response: str, context_docs: list[str]
    ) -> dict:
        """Returns per-claim scores for interpretability / trace logging."""
        if not response.strip() or not context_docs:
            return {"total": 0.0, "claims": []}

        context = " ".join(context_docs)
        claims = _split_sentences(response)
        pairs = [{"text": context, "text_pair": claim} for claim in claims]
        raw = self._pipe(pairs, batch_size=self._batch_size, truncation=True, padding=True)

        claim_scores = []
        for claim, result in zip(claims, raw):
            label_scores = {r["label"]: r["score"] for r in result}
            claim_scores.append({
                "claim": claim,
                "entailment": label_scores.get(self._entailment_label, 0.0),
                "contradiction": label_scores.get("contradiction", 0.0),
                "neutral": label_scores.get("neutral", 0.0),
            })

        total = float(self._aggregate(
            np.array([c["entailment"] for c in claim_scores])
        ))
        return {"total": total, "claims": claim_scores}

    # ------------------------------------------------------------------
    def _aggregate(self, scores: np.ndarray) -> float:
        if self._aggregation == "min":
            return float(scores.min())
        if self._aggregation == "harmonic":
            if scores.min() <= 0:
                return 0.0
            return float(len(scores) / np.sum(1.0 / scores))
        return float(scores.mean())

    def _resolve_entailment_label(self) -> str:
        sample = self._pipe(
            [{"text": "The sky is blue.", "text_pair": "The sky is blue."}],
            batch_size=1, truncation=True, padding=True,
        )
        labels = [r["label"] for r in sample[0]]
        for candidate in ("entailment", "ENTAILMENT", "LABEL_0"):
            if candidate in labels:
                return candidate
        return labels[0]
