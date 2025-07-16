"""
LLM-as-judge evaluation layer — a second signal on top of the NLI reward.

Provides interpretable, rubric-based scoring that complements the
NLI classifier: the NLI reward is fast and differentiable; the judge
is slower but gives explainable per-criterion scores (Glider-style).

Criteria:
  factual_accuracy  — claims match the context
  completeness      — all parts of the question are addressed
  grounding         — answer explicitly cites sources
  calibration       — agent expresses appropriate uncertainty
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


JUDGE_SYSTEM = """You are an expert evaluator assessing LLM-generated answers
for factual accuracy and grounding. You will be given a question, context
documents, and a candidate answer.

Score each criterion from 0.0 to 1.0 and return ONLY valid JSON.
Criteria:
  factual_accuracy  : Are all claims in the answer supported by the context?
  completeness      : Does the answer address all aspects of the question?
  grounding         : Does the answer cite or reference the context?
  calibration       : Does the answer appropriately express uncertainty when needed?

Return JSON with keys: factual_accuracy, completeness, grounding, calibration, rationale
"""

JUDGE_HUMAN = """Question: {query}

Context documents:
{context}

Candidate answer:
{response}

Score each criterion 0.0–1.0 and explain briefly in 'rationale'. Return JSON only."""


class LLMJudge:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str = "ollama",
    ):
        self._llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.0)

    def evaluate(self, query: str, context_docs: list[str], response: str) -> dict:
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(context_docs))
        messages = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=JUDGE_HUMAN.format(
                query=query, context=context, response=response
            )),
        ]
        raw = self._llm.invoke(messages).content
        return self._parse(raw)

    def _parse(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"error": "unparseable", "raw": text}
        try:
            data = json.loads(match.group())
            score = sum(
                float(data.get(k, 0))
                for k in ["factual_accuracy", "completeness", "grounding", "calibration"]
            ) / 4.0
            data["composite"] = round(score, 4)
            return data
        except json.JSONDecodeError:
            return {"error": "json_error", "raw": text}
