"""
Multi-turn environment: the agent issues tool calls (retrieve, calculate, respond)
across multiple steps before a terminal RESPOND action closes the episode.
Designed to test long-horizon factual consistency and tool-use grounding.

Connects to Rebecca's MEMTRACK interest: state is tracked across turns and the
full interaction trace is logged for offline analysis (TRAIL-style).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from factenv.rewards.composite_reward import CompositeReward
from factenv.tasks.task_bank import Task, TaskBank
from factenv.agent.tool_registry import ToolRegistry
from factenv.eval.trace_logger import TraceLogger


ACTIONS = ["RETRIEVE", "CALCULATE", "RESPOND", "CLARIFY"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class MultiTurnConfig:
    max_steps: int = 6
    top_k_docs: int = 3
    step_penalty: float = -0.02   # small cost per extra turn to encourage efficiency
    reward_weights: dict = field(default_factory=lambda: {
        "nli": 0.55,
        "citation": 0.20,
        "length_penalty": 0.10,
        "refusal_bonus": 0.05,
        "tool_use_bonus": 0.10,
    })
    log_traces: bool = True


class MultiTurnFactualEnv(gym.Env):
    """
    Multi-turn environment where an LLM agent can:
      RETRIEVE  — issue a sub-query to the retrieval index
      CALCULATE — call a calculator tool
      CLARIFY   — ask a clarifying question (incurs step cost)
      RESPOND   — produce a final answer (terminates episode)

    The episode is also terminated if max_steps is exceeded (truncated=True).
    History of tool calls and retrieved snippets is carried in `info` so that
    prompt-based LLM agents can include it in their context window.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task_bank: TaskBank,
        tool_registry: ToolRegistry,
        config: MultiTurnConfig | None = None,
    ):
        super().__init__()
        self.config = config or MultiTurnConfig()
        self.task_bank = task_bank
        self.tools = tool_registry
        self.reward_fn = CompositeReward(self.config.reward_weights)
        self.logger = TraceLogger(enabled=self.config.log_traces)

        self.observation_space = gym.spaces.Dict({
            "query_len":       gym.spaces.Box(0, 512,  shape=(1,), dtype=np.int32),
            "history_len":     gym.spaces.Box(0, 1000, shape=(1,), dtype=np.int32),
            "step":            gym.spaces.Box(0, 100,  shape=(1,), dtype=np.int32),
            "difficulty":      gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "n_retrieved_docs":gym.spaces.Box(0, 50,   shape=(1,), dtype=np.int32),
        })
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        self._current_task: Task | None = None
        self._step = 0
        self._history: list[dict] = []
        self._retrieved_docs: list[str] = []

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_task = self.task_bank.sample(options)
        self._step = 0
        self._history = []
        self._retrieved_docs = list(self._current_task.context_docs)
        self.logger.start_episode(self._current_task)
        return self._make_obs(), self._make_info()

    def step(self, action: dict):
        """
        action: {
            "type": "RETRIEVE" | "CALCULATE" | "CLARIFY" | "RESPOND",
            "content": str   # sub-query, expression, clarification, or final answer
        }
        """
        assert self._current_task is not None, "Call reset() first."
        self._step += 1
        action_type = action.get("type", "RESPOND")
        content = action.get("content", "")

        reward = self.config.step_penalty
        terminated = False
        truncated = False
        tool_result = None

        if action_type == "RETRIEVE":
            tool_result = self.tools.retrieve(content, top_k=self.config.top_k_docs)
            self._retrieved_docs.extend(tool_result)
            reward += 0.0  # neutral; reward comes at RESPOND time

        elif action_type == "CALCULATE":
            tool_result = self.tools.calculate(content)

        elif action_type == "CLARIFY":
            tool_result = "clarification logged"

        elif action_type == "RESPOND":
            breakdown = self.reward_fn.score(
                response=content,
                context_docs=self._retrieved_docs,
                reference_answer=self._current_task.reference_answer,
                tool_use_bonus=len(self._history) > 0,
            )
            reward += breakdown["total"]
            terminated = True
            self.logger.log_step(self._step, action, breakdown, self._current_task)

        self._history.append({
            "step": self._step,
            "action_type": action_type,
            "content": content,
            "tool_result": tool_result,
        })

        if self._step >= self.config.max_steps and not terminated:
            truncated = True

        info = self._make_info()
        info["last_tool_result"] = tool_result
        return self._make_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Query: {self._current_task.query}  Step: {self._step}")
        for h in self._history:
            print(f"  [{h['step']}] {h['action_type']}: {h['content'][:80]}")

    # ------------------------------------------------------------------
    def _make_obs(self) -> dict:
        history_text_len = sum(len(str(h)) for h in self._history)
        return {
            "query_len":        np.array([len(self._current_task.query)], dtype=np.int32),
            "history_len":      np.array([history_text_len],              dtype=np.int32),
            "step":             np.array([self._step],                    dtype=np.int32),
            "difficulty":       np.array([self._current_task.difficulty], dtype=np.float32),
            "n_retrieved_docs": np.array([len(self._retrieved_docs)],     dtype=np.int32),
        }

    def _make_info(self) -> dict:
        return {
            "query":         self._current_task.query,
            "docs":          self._retrieved_docs,
            "history":       self._history,
            "task_id":       self._current_task.task_id,
            "difficulty":    self._current_task.difficulty,
        }
