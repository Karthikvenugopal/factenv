"""
Evaluate a trained agent across all difficulty tiers.

Run:
    python scripts/evaluate.py --episodes 60 --model gpt-4o-mini
"""

import argparse

from factenv.tasks.task_bank import TaskBank, Difficulty
from factenv.agent.rag_agent import RAGAgent
from factenv.eval.evaluator import Evaluator
from factenv.eval.judge import LLMJudge
from factenv.rewards.composite_reward import CompositeReward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--ollama", action="store_true")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    base_url = "http://localhost:11434/v1" if args.ollama else None
    api_key = "ollama" if args.ollama else None

    task_bank = TaskBank()
    agent = RAGAgent(model=args.model, base_url=base_url, api_key=api_key)
    judge = None if args.no_judge else LLMJudge(model=args.model, base_url=base_url, api_key=api_key)

    evaluator = Evaluator(
        task_bank=task_bank,
        agent_fn=agent.act,
        judge=judge,
        mlflow_experiment="factenv-eval",
    )

    results = evaluator.run(n_episodes=args.episodes)

    print("\n=== Evaluation Results ===")
    for tier in Difficulty:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue
        avg_nli = sum(r.nli_score for r in tier_results) / len(tier_results)
        avg_reward = sum(r.reward for r in tier_results) / len(tier_results)
        print(f"  {tier.value:8s}  n={len(tier_results):3d}  avg_nli={avg_nli:.3f}  avg_reward={avg_reward:.3f}")

    overall_nli = sum(r.nli_score for r in results) / len(results)
    overall_reward = sum(r.reward for r in results) / len(results)
    print(f"\n  {'OVERALL':8s}  n={len(results):3d}  avg_nli={overall_nli:.3f}  avg_reward={overall_reward:.3f}")


if __name__ == "__main__":
    main()
