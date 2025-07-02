"""
Episode trace logger — inspired by TRAIL (Trace Reasoning and Agentic Issue
Localization). Records full decision traces per episode to a JSONL file so that
post-hoc analysis can identify *where* in the reasoning chain errors originate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, enabled: bool = True, log_dir: str = "runs/traces"):
        self.enabled = enabled
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._episode: dict | None = None
        self._steps: list[dict] = []
        self._episode_count = 0

    def start_episode(self, task: Any):
        if not self.enabled:
            return
        self._episode = {
            "episode_id": self._episode_count,
            "task_id": task.task_id,
            "query": task.query,
            "tier": task.tier.value if hasattr(task.tier, "value") else str(task.tier),
            "difficulty": task.difficulty,
            "start_time": time.time(),
        }
        self._steps = []
        self._episode_count += 1

    def log_step(self, step: int, action: Any, reward: Any, task: Any):
        if not self.enabled or self._episode is None:
            return
        self._steps.append({
            "step": step,
            "action": action if isinstance(action, str) else str(action),
            "reward": reward,
            "timestamp": time.time(),
        })

    def flush(self):
        if not self.enabled or self._episode is None:
            return
        record = {
            **self._episode,
            "steps": self._steps,
            "end_time": time.time(),
        }
        out_path = self._log_dir / f"episode_{self._episode['episode_id']:06d}.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._episode = None
        self._steps = []
