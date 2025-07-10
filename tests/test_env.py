"""Unit tests for environment + reward pipeline (no LLM calls needed)."""

import pytest
from factenv.tasks.task_bank import TaskBank, Difficulty
from factenv.env.rag_env import FactualConsistencyEnv, EnvConfig
from factenv.curriculum.scheduler import CurriculumScheduler, CurriculumConfig
from factenv.agent.tool_registry import ToolRegistry, MockVectorStore


def make_env(log_traces=False):
    return FactualConsistencyEnv(
        TaskBank(),
        EnvConfig(max_steps=1, log_traces=log_traces),
    )


def test_env_reset_returns_obs_and_info():
    env = make_env()
    obs, info = env.reset()
    assert "query" in info
    assert "docs" in info
    assert obs["step"][0] == 0


def test_env_step_terminates_after_max_steps():
    env = make_env()
    env.reset()
    _, _, terminated, truncated, info = env.step("The answer is 42.")
    assert terminated or truncated


def test_env_step_returns_reward_breakdown():
    env = make_env()
    env.reset()
    _, reward, _, _, info = env.step("Paris is the capital of France.")
    assert "reward_breakdown" in info
    assert "nli" in info["reward_breakdown"]
    assert isinstance(reward, float)


def test_curriculum_promotes_on_high_success():
    sched = CurriculumScheduler(CurriculumConfig(
        promote_threshold=0.6,
        demote_threshold=0.2,
        window_size=5,
        min_episodes_before_change=5,
    ))
    assert sched.current_tier == Difficulty.EASY
    for _ in range(5):
        sched.record(0.9)   # all successes
    assert sched.current_tier == Difficulty.MEDIUM


def test_curriculum_demotes_on_low_success():
    sched = CurriculumScheduler(CurriculumConfig(
        promote_threshold=0.8,
        demote_threshold=0.3,
        window_size=5,
        min_episodes_before_change=3,
    ))
    sched._tier_idx = 1    # start at MEDIUM
    for _ in range(5):
        sched.record(0.1)  # all failures
    assert sched.current_tier == Difficulty.EASY


def test_tool_registry_calculate():
    registry = ToolRegistry()
    assert registry.calculate("2 + 2") == "4"
    assert registry.calculate("10 / 2") == "5.0"


def test_tool_registry_retrieve_returns_passages():
    store = MockVectorStore(["passage A", "passage B", "passage C"])
    registry = ToolRegistry(vector_store=store)
    results = registry.retrieve("any query", top_k=2)
    assert len(results) == 2
    assert results[0] == "passage A"


def test_task_bank_samples_by_tier():
    bank = TaskBank(seed=42)
    for _ in range(10):
        task = bank.sample({"tier": Difficulty.HARD})
        assert task.tier == Difficulty.HARD
