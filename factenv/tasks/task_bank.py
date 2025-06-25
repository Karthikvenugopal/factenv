"""
Task bank with three difficulty tiers:
  EASY   — single-hop, clean context, unambiguous answer
  MEDIUM — multi-hop, mild distractors
  HARD   — adversarial context (deliberate contradictions), requires tool use

The curriculum scheduler promotes/demotes agents between tiers based on rolling
success rate, implementing the adaptive curriculum learning Rebecca described.

Tasks are sourced from built-in samples here; swap in HotpotQA / NQ / TriviaQA
loaders in tasks/loaders.py for production use.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


DIFFICULTY_SCORE = {Difficulty.EASY: 0.2, Difficulty.MEDIUM: 0.5, Difficulty.HARD: 0.9}


@dataclass
class Task:
    query: str
    context_docs: list[str]
    reference_answer: str
    difficulty: float                    # 0.0 – 1.0
    tier: Difficulty
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in seed tasks (replace/extend with dataset loaders for scale)
# ---------------------------------------------------------------------------

_SEED_TASKS: list[dict] = [
    # EASY — direct factual lookup
    {
        "tier": Difficulty.EASY,
        "query": "What year was the Eiffel Tower completed?",
        "context_docs": [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "It was designed by Gustave Eiffel and completed in 1889 for the 1889 World's Fair.",
        ],
        "reference_answer": "The Eiffel Tower was completed in 1889.",
    },
    {
        "tier": Difficulty.EASY,
        "query": "Who wrote Pride and Prejudice?",
        "context_docs": [
            "Pride and Prejudice is a romantic novel by Jane Austen, published in 1813.",
            "The novel follows the character development of Elizabeth Bennet.",
        ],
        "reference_answer": "Pride and Prejudice was written by Jane Austen.",
    },
    # MEDIUM — multi-hop
    {
        "tier": Difficulty.MEDIUM,
        "query": "Which country invented the language used to write the first stored-program computer?",
        "context_docs": [
            "The Manchester Baby (SSEM) was the first computer to run a stored program, in 1948.",
            "It was built at the University of Manchester in England.",
            "Its program was written in machine code, which originated from US/UK computing research.",
            "Alan Turing, a British mathematician, formalized the concept of algorithms.",
        ],
        "reference_answer": "The United Kingdom, specifically England, where the Manchester Baby ran the first stored program.",
    },
    {
        "tier": Difficulty.MEDIUM,
        "query": "What is the boiling point of the element whose atomic number is one more than carbon?",
        "context_docs": [
            "Carbon has atomic number 6.",
            "Nitrogen has atomic number 7, directly after carbon on the periodic table.",
            "Nitrogen's boiling point is −195.79 °C (77.36 K) at standard pressure.",
        ],
        "reference_answer": "Nitrogen (atomic number 7) has a boiling point of −195.79 °C.",
    },
    # HARD — adversarial / contradictory context
    {
        "tier": Difficulty.HARD,
        "query": "What is the capital of Australia?",
        "context_docs": [
            "Sydney is the largest city in Australia and a major financial hub.",
            "Many people mistakenly believe Sydney is the capital of Australia.",
            "Canberra became the capital of Australia in 1913, chosen as a compromise between Sydney and Melbourne.",
            "Melbourne served as the temporary capital from 1901 to 1927.",
        ],
        "reference_answer": "Canberra is the capital of Australia.",
    },
    {
        "tier": Difficulty.HARD,
        "query": "How many moons does Mars have, and what are their names?",
        "context_docs": [
            "Mars has two small moons.",
            "Early telescopes suggested Mars might have no moons.",
            "The moons are called Phobos and Deimos, discovered in 1877 by Asaph Hall.",
            "Some sources incorrectly list only Phobos as Mars's moon.",
        ],
        "reference_answer": "Mars has two moons: Phobos and Deimos, discovered in 1877.",
    },
]


class TaskBank:
    """
    Provides sampled tasks with optional difficulty filtering.
    Supports curriculum-aware sampling via `options={'tier': Difficulty.EASY}`.
    """

    def __init__(self, tasks: list[Task] | None = None, seed: int | None = None):
        self._rng = random.Random(seed)
        self._tasks = tasks or self._build_defaults()

    def sample(self, options: dict | None = None) -> Task:
        pool = self._tasks
        if options and "tier" in options:
            pool = [t for t in pool if t.tier == options["tier"]] or self._tasks
        task = self._rng.choice(pool)
        # Return a fresh copy so task_id is unique per episode
        return Task(
            query=task.query,
            context_docs=task.context_docs,
            reference_answer=task.reference_answer,
            difficulty=task.difficulty,
            tier=task.tier,
            metadata=dict(task.metadata),
        )

    def add(self, task: Task):
        self._tasks.append(task)

    def __len__(self):
        return len(self._tasks)

    @staticmethod
    def _build_defaults() -> list[Task]:
        tasks = []
        for t in _SEED_TASKS:
            tasks.append(Task(
                query=t["query"],
                context_docs=t["context_docs"],
                reference_answer=t["reference_answer"],
                difficulty=DIFFICULTY_SCORE[t["tier"]],
                tier=t["tier"],
            ))
        return tasks
