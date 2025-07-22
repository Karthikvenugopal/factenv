"""
Training loop demo: run the RAGAgent through FactualConsistencyEnv with the
curriculum scheduler. Logs reward curves and tier promotions to MLflow.

Run:
    python scripts/train.py --episodes 200 --model gpt-4o-mini
    python scripts/train.py --episodes 200 --model llama3.2:3b --ollama
"""

import argparse
import mlflow

from factenv.env.rag_env import FactualConsistencyEnv, EnvConfig
from factenv.tasks.task_bank import TaskBank
from factenv.curriculum.scheduler import CurriculumScheduler, CurriculumConfig
from factenv.agent.rag_agent import RAGAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--ollama", action="store_true", help="Use local Ollama endpoint")
    p.add_argument("--experiment", type=str, default="factenv-train")
    return p.parse_args()


def main():
    args = parse_args()

    task_bank = TaskBank()
    curriculum = CurriculumScheduler(CurriculumConfig(
        promote_threshold=0.75,
        demote_threshold=0.30,
        window_size=15,
    ))
    env = FactualConsistencyEnv(task_bank, EnvConfig(max_steps=1, log_traces=True))
    agent = RAGAgent(
        model=args.model,
        base_url="http://localhost:11434/v1" if args.ollama else None,
        api_key="ollama" if args.ollama else None,
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "model": args.model,
            "episodes": args.episodes,
            "ollama": args.ollama,
        })

        for episode in range(args.episodes):
            tier = curriculum.current_tier
            obs, info = env.reset(options={"tier": tier})
            action = agent.act(info)
            response = action["content"] if isinstance(action, dict) else action
            _, reward, _, _, step_info = env.step(response)

            status = curriculum.record(reward)

            mlflow.log_metrics({
                "reward":    reward,
                "nli_score": step_info["reward_breakdown"]["nli"],
                "difficulty": info["difficulty"],
            }, step=episode)

            if status.get("changed"):
                direction = status["direction"]
                new_tier = status["new_tier"].value
                print(f"[ep {episode:4d}] Tier {direction}: → {new_tier}  (success_rate={status['success_rate']:.2f})")
                mlflow.log_metric("tier_idx", curriculum._tier_idx, step=episode)
            elif episode % 20 == 0:
                stats = curriculum.stats()
                print(
                    f"[ep {episode:4d}] tier={stats['current_tier'].value}  "
                    f"reward={reward:.3f}  nli={step_info['reward_breakdown']['nli']:.3f}"
                )


if __name__ == "__main__":
    main()
