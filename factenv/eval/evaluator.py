"""
Batch evaluator: runs the agent over a task bank and reports aggregate metrics.
Outputs an MLflow run with per-tier NLI scores, judge scores, and promotion events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mlflow

from factenv.tasks.task_bank import TaskBank, Difficulty
from factenv.eval.judge import LLMJudge
from factenv.rewards.composite_reward import CompositeReward


@dataclass
class EvalResult:
    task_id: str
    tier: Difficulty
    query: str
    response: str
    nli_score: float
    judge_scores: dict
    reward: float


class Evaluator:
    """
    Runs a batch evaluation loop and logs results to MLflow.

    Usage:
        evaluator = Evaluator(task_bank, agent_fn=agent.act)
        results = evaluator.run(n_episodes=50)
    """

    def __init__(
        self,
        task_bank: TaskBank,
        agent_fn: Callable,
        reward_fn: CompositeReward | None = None,
        judge: LLMJudge | None = None,
        mlflow_experiment: str = "factenv-eval",
    ):
        self._bank = task_bank
        self._agent_fn = agent_fn
        self._reward_fn = reward_fn or CompositeReward({
            "nli": 0.6, "citation": 0.2, "length_penalty": 0.1, "refusal_bonus": 0.1,
        })
        self._judge = judge
        mlflow.set_experiment(mlflow_experiment)

    def run(self, n_episodes: int = 100, tier: Difficulty | None = None) -> list[EvalResult]:
        results = []
        options = {"tier": tier} if tier else None

        with mlflow.start_run():
            mlflow.log_param("n_episodes", n_episodes)
            mlflow.log_param("tier", tier.value if tier else "all")

            for i in range(n_episodes):
                task = self._bank.sample(options)
                info = {"query": task.query, "docs": task.context_docs, "history": []}
                action = self._agent_fn(info)
                response = action["content"] if isinstance(action, dict) else action

                reward_breakdown = self._reward_fn.score(
                    response=response,
                    context_docs=task.context_docs,
                    reference_answer=task.reference_answer,
                )

                judge_scores = {}
                if self._judge:
                    judge_scores = self._judge.evaluate(task.query, task.context_docs, response)

                result = EvalResult(
                    task_id=task.task_id,
                    tier=task.tier,
                    query=task.query,
                    response=response,
                    nli_score=reward_breakdown["nli"],
                    judge_scores=judge_scores,
                    reward=reward_breakdown["total"],
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    self._log_metrics(results, step=i + 1)

            self._log_metrics(results, step=n_episodes)

        return results

    def _log_metrics(self, results: list[EvalResult], step: int):
        if not results:
            return
        avg_nli = sum(r.nli_score for r in results) / len(results)
        avg_reward = sum(r.reward for r in results) / len(results)
        mlflow.log_metric("avg_nli_score", avg_nli, step=step)
        mlflow.log_metric("avg_reward", avg_reward, step=step)

        for tier in Difficulty:
            tier_results = [r for r in results if r.tier == tier]
            if tier_results:
                mlflow.log_metric(
                    f"avg_nli_{tier.value}",
                    sum(r.nli_score for r in tier_results) / len(tier_results),
                    step=step,
                )
