"""
Composite reward combining NLI consistency, citation accuracy, length
penalty, refusal bonus, and optional tool-use bonus.

Reward design is a first-class concern here (Rebecca's explicit interest).
Each component is independently interpretable for reward shaping analysis.
"""

from __future__ import annotations

import re
from typing import Optional

from factenv.rewards.nli_reward import NLIReward


_CITATION_PATTERN = re.compile(r"\[(\d+)\]|\(source[:\s]\d+\)", re.IGNORECASE)
_REFUSAL_PHRASES = [
    "i don't know", "i cannot", "not enough information",
    "the context does not", "cannot determine",
]


class CompositeReward:
    def __init__(self, weights: dict):
        """
        weights keys: nli, citation, length_penalty, refusal_bonus, tool_use_bonus
        All weights should sum to ≤ 1.0 for interpretable total ∈ [-1, 1].
        """
        self.weights = weights
        self._nli = NLIReward()

    def score(
        self,
        response: str,
        context_docs: list[str],
        reference_answer: Optional[str] = None,
        tool_use_bonus: bool = False,
    ) -> dict:
        """Returns a breakdown dict with keys for each component + 'total'."""
        breakdown = {}

        # 1. NLI factual consistency (primary signal)
        nli_detail = self._nli.score_with_breakdown(response, context_docs)
        breakdown["nli"] = nli_detail["total"]
        breakdown["nli_claims"] = nli_detail["claims"]

        # 2. Citation accuracy: reward if response cites sources explicitly
        breakdown["citation"] = self._citation_score(response, context_docs)

        # 3. Length penalty: discourage verbose hedging over substance
        breakdown["length_penalty"] = self._length_penalty(response)

        # 4. Refusal bonus: reward calibrated "I don't know" when context lacks answer
        breakdown["refusal_bonus"] = self._refusal_bonus(response, context_docs)

        # 5. Tool-use bonus: reward agents that used retrieval before answering
        breakdown["tool_use_bonus"] = float(tool_use_bonus)

        total = sum(
            self.weights.get(k, 0.0) * breakdown[k]
            for k in ["nli", "citation", "length_penalty", "refusal_bonus", "tool_use_bonus"]
        )
        breakdown["total"] = float(total)
        return breakdown

    # ------------------------------------------------------------------
    def _citation_score(self, response: str, context_docs: list[str]) -> float:
        if not context_docs:
            return 0.0
        citations_found = len(_CITATION_PATTERN.findall(response))
        return min(1.0, citations_found / max(1, len(context_docs)))

    def _length_penalty(self, response: str) -> float:
        words = len(response.split())
        if words < 10:
            return -0.5   # too short to be useful
        if words > 300:
            return -0.3   # probably padding
        return 0.0        # acceptable range

    def _refusal_bonus(self, response: str, context_docs: list[str]) -> float:
        """
        Grant a small bonus when the agent correctly refuses to answer
        unanswerable questions instead of hallucinating.
        Detect unanswerability heuristically: if NLI score is very low across
        all docs, the agent *should* express uncertainty.
        """
        lower = response.lower()
        is_refusal = any(phrase in lower for phrase in _REFUSAL_PHRASES)
        if not is_refusal:
            return 0.0
        # Only reward refusal if context genuinely lacks the answer
        nli_score = self._nli.score(response, context_docs)
        return 0.5 if nli_score < 0.3 else -0.2  # penalize false refusals
