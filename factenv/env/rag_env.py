"""
Single-turn RAG environment: the agent retrieves context and generates one response per episode.
State  = (query, retrieved_docs, step)
Action = generated response string
Reward = composite factual consistency score
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from factenv.rewards.composite_reward import CompositeReward
from factenv.tasks.task_bank import Task, TaskBank
from factenv.eval.trace_logger import TraceLogger


@dataclass
class EnvConfig:
    max_steps: int = 1          # single-turn; extend to >1 for agentic loops
    top_k_docs: int = 3
    reward_weights: dict = field(default_factory=lambda: {
        "nli": 0.6,
        "citation": 0.2,
        "length_penalty": 0.1,
        "refusal_bonus": 0.1,
    })
    log_traces: bool = True


class FactualConsistencyEnv(gym.Env):
    """
    Gymnasium environment wrapping a RAG factual-consistency task.

    Observation space: Dict with keys
        - query        : str
        - context_docs : list[str]   (top-k retrieved passages)
        - step         : int

    Action space: Text (str) — the agent's response.
    Gymnasium's Text space requires gymnasium>=0.29; we use a simple
    custom encoding so the env stays compatible with older versions too.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_bank: TaskBank, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.task_bank = task_bank
        self.reward_fn = CompositeReward(self.config.reward_weights)
        self.logger = TraceLogger(enabled=self.config.log_traces)

        # Gymnasium spaces (text encoded as fixed-dim float32 for RL libs that
        # need Box spaces; real text is carried in `info`).
        self.observation_space = gym.spaces.Dict({
            "query_len":   gym.spaces.Box(0, 512,  shape=(1,), dtype=np.int32),
            "n_docs":      gym.spaces.Box(0, 20,   shape=(1,), dtype=np.int32),
            "step":        gym.spaces.Box(0, 100,  shape=(1,), dtype=np.int32),
            "difficulty":  gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        # Action space is a Discrete placeholder; real LLM text arrives via step()
        self.action_space = gym.spaces.Discrete(1)

        self._current_task: Task | None = None
        self._step = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_task = self.task_bank.sample(options)
        self._step = 0
        self.logger.start_episode(self._current_task)
        obs = self._make_obs()
        info = {
            "query":    self._current_task.query,
            "docs":     self._current_task.context_docs,
            "task_id":  self._current_task.task_id,
            "difficulty": self._current_task.difficulty,
        }
        return obs, info

    def step(self, action: str):
        """
        action: the LLM-generated response string.
        Returns standard (obs, reward, terminated, truncated, info).
        """
        assert self._current_task is not None, "Call reset() first."
        self._step += 1

        reward_breakdown = self.reward_fn.score(
            response=action,
            context_docs=self._current_task.context_docs,
            reference_answer=self._current_task.reference_answer,
        )
        reward = reward_breakdown["total"]

        terminated = self._step >= self.config.max_steps
        truncated = False

        self.logger.log_step(
            step=self._step,
            action=action,
            reward=reward_breakdown,
            task=self._current_task,
        )

        obs = self._make_obs()
        info = {
            "reward_breakdown": reward_breakdown,
            "query":  self._current_task.query,
            "docs":   self._current_task.context_docs,
            "action": action,
            "task_id": self._current_task.task_id,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._current_task:
            print(f"Query   : {self._current_task.query}")
            print(f"Step    : {self._step}/{self.config.max_steps}")

    # ------------------------------------------------------------------
    def _make_obs(self) -> dict:
        task = self._current_task
        return {
            "query_len":  np.array([len(task.query)],       dtype=np.int32),
            "n_docs":     np.array([len(task.context_docs)], dtype=np.int32),
            "step":       np.array([self._step],             dtype=np.int32),
            "difficulty": np.array([task.difficulty],        dtype=np.float32),
        }
