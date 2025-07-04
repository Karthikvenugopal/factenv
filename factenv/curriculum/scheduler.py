"""
Adaptive curriculum scheduler.

Tracks rolling success rate per difficulty tier and automatically promotes
or demotes the agent to a harder/easier tier based on configurable thresholds.

This implements the curriculum learning approach Rebecca explicitly named as
a research interest, adapted for LLM-agent training loops.

Promotion logic:
  - Promote to next tier when rolling success rate ≥ promote_threshold
  - Demote to previous tier when rolling success rate ≤ demote_threshold
  - "Success" = episode reward ≥ success_reward_threshold

Window size controls how quickly the curriculum reacts (smaller = more
reactive, larger = more stable).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from factenv.tasks.task_bank import Difficulty


TIER_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


@dataclass
class CurriculumConfig:
    promote_threshold: float = 0.75    # promote when 75% success over window
    demote_threshold: float = 0.30     # demote when below 30% success
    window_size: int = 20              # rolling window of episodes per tier
    success_reward_threshold: float = 0.6  # episode reward considered a success
    min_episodes_before_change: int = 10   # don't change tier too quickly


class CurriculumScheduler:
    """
    Wraps a TaskBank and maintains the current difficulty tier.
    Call `record(reward)` after each episode; call `current_tier` to get
    the tier to sample from for the next episode.
    """

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self._tier_idx = 0   # start at EASY
        self._windows: dict[Difficulty, deque] = {
            t: deque(maxlen=self.config.window_size) for t in Difficulty
        }
        self._episodes_at_tier = 0
        self._history: list[dict] = []

    @property
    def current_tier(self) -> Difficulty:
        return TIER_ORDER[self._tier_idx]

    def record(self, reward: float) -> dict:
        """
        Record episode reward and potentially update tier.
        Returns a status dict describing any tier change.
        """
        tier = self.current_tier
        success = reward >= self.config.success_reward_threshold
        self._windows[tier].append(float(success))
        self._episodes_at_tier += 1

        status = {"tier": tier, "reward": reward, "success": success, "changed": False}

        if self._episodes_at_tier < self.config.min_episodes_before_change:
            self._history.append(status)
            return status

        rate = self._success_rate(tier)
        status["success_rate"] = rate

        if rate >= self.config.promote_threshold and self._tier_idx < len(TIER_ORDER) - 1:
            self._tier_idx += 1
            self._episodes_at_tier = 0
            status["changed"] = True
            status["direction"] = "promote"
            status["new_tier"] = TIER_ORDER[self._tier_idx]

        elif rate <= self.config.demote_threshold and self._tier_idx > 0:
            self._tier_idx -= 1
            self._episodes_at_tier = 0
            status["changed"] = True
            status["direction"] = "demote"
            status["new_tier"] = TIER_ORDER[self._tier_idx]

        self._history.append(status)
        return status

    def stats(self) -> dict:
        return {
            "current_tier": self.current_tier,
            "tier_idx": self._tier_idx,
            "episodes_at_tier": self._episodes_at_tier,
            "success_rates": {
                t.value: self._success_rate(t) for t in Difficulty
            },
        }

    def _success_rate(self, tier: Difficulty) -> float:
        window = self._windows[tier]
        if not window:
            return 0.0
        return sum(window) / len(window)
