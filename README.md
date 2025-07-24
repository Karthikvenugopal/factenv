# FactEnv

A Gymnasium-compatible reinforcement learning environment for training factually grounded RAG agents.

The core idea: frame retrieval-augmented generation as a Markov Decision Process where the **reward signal is a claim-level NLI entailment score** — penalizing hallucination at the claim level rather than treating the full response as a single unit (the same granularity insight behind [Lynx](https://arxiv.org/abs/2407.06037)).

---

## Design

### Environment as MDP

| Component | Definition |
|-----------|-----------|
| **State** | (query, retrieved docs, conversation history, step index) |
| **Action** | LLM-generated response string (or structured tool call in multi-turn mode) |
| **Reward** | Composite factual consistency score (see below) |
| **Episode** | Terminates on `RESPOND` action or `max_steps` exceeded |

Two environment variants:

- **`FactualConsistencyEnv`** — single-turn: agent receives context and produces one answer
- **`MultiTurnFactualEnv`** — multi-turn: agent can `RETRIEVE`, `CALCULATE`, `CLARIFY`, or `RESPOND` across multiple steps; state is tracked across turns (similar to [MEMTRACK](https://arxiv.org/abs/2504.00298) evaluation settings)

### Reward Design

The composite reward has five independently interpretable components:

```
reward = w_nli   × NLI_entailment_score        # primary: claim-level factual consistency
       + w_cite  × citation_accuracy            # does the answer reference sources?
       + w_len   × length_penalty               # penalize padding/over-hedging
       + w_ref   × calibrated_refusal_bonus     # reward correct "I don't know"
       + w_tool  × tool_use_bonus               # reward agents that retrieved before answering
```

**NLI scorer**: `cross-encoder/nli-deberta-v3-base` (HuggingFace) — splits the response into sentences, scores each claim independently against the retrieved context, then aggregates. This surfaces individual hallucinated claims rather than masking them behind an average.

### Curriculum Learning

`CurriculumScheduler` tracks a rolling success-rate window per difficulty tier and automatically promotes or demotes the agent:

```
EASY  (single-hop, clean context)
  ↕  promote ≥ 75% success over last 20 episodes
MEDIUM (multi-hop, mild distractors)
  ↕  demote  ≤ 30% success
HARD  (adversarial context, deliberate contradictions)
```

### Trace Logging

Every episode writes a structured JSONL trace to `runs/traces/` — query, retrieved docs, each step's action and reward breakdown. Designed for post-hoc issue localization (inspired by [TRAIL](https://arxiv.org/abs/2504.04948)).

### LLM-as-Judge

`LLMJudge` provides a secondary, interpretable rubric-based evaluation layer:

| Criterion | Description |
|-----------|-------------|
| `factual_accuracy` | Are all claims supported by context? |
| `completeness` | Does the answer address the full question? |
| `grounding` | Does the answer cite sources? |
| `calibration` | Is uncertainty expressed appropriately? |

Inspired by [Glider](https://arxiv.org/abs/2410.14669).

---

## Quickstart

```bash
pip install -e .
```

### Single-turn environment

```python
from factenv import FactualConsistencyEnv
from factenv.tasks.task_bank import TaskBank
from factenv.curriculum.scheduler import CurriculumScheduler

task_bank = TaskBank()
curriculum = CurriculumScheduler()
env = FactualConsistencyEnv(task_bank)

obs, info = env.reset(options={"tier": curriculum.current_tier})
response = "Your LLM response here."
obs, reward, terminated, truncated, info = env.step(response)

status = curriculum.record(reward)
print(f"Reward: {reward:.3f} | NLI: {info['reward_breakdown']['nli']:.3f}")
```

### Multi-turn environment with tool use

```python
from factenv.env.multi_turn_env import MultiTurnFactualEnv
from factenv.agent.tool_registry import ToolRegistry, MockVectorStore

store = MockVectorStore(passages=["passage A", "passage B"])
tools = ToolRegistry(vector_store=store)
env = MultiTurnFactualEnv(task_bank, tool_registry=tools)

obs, info = env.reset()
# Retrieve step
obs, r, done, _, info = env.step({"type": "RETRIEVE", "content": "sub-query"})
# Final answer step
obs, r, done, _, info = env.step({"type": "RESPOND", "content": "Final answer citing [doc_1]."})
```

### Run the training loop

```bash
# With OpenAI
python scripts/train.py --episodes 200 --model gpt-4o-mini

# With local Ollama (llama3.2:3b — same model used in your content moderation pipeline)
python scripts/train.py --episodes 200 --model llama3.2:3b --ollama
```

### Run evaluation

```bash
python scripts/evaluate.py --episodes 60 --model gpt-4o-mini
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
factenv/
├── factenv/
│   ├── env/
│   │   ├── rag_env.py           # Single-turn Gymnasium environment
│   │   └── multi_turn_env.py    # Multi-turn tool-use environment
│   ├── rewards/
│   │   ├── nli_reward.py        # Claim-level NLI entailment scorer
│   │   └── composite_reward.py  # Multi-component reward function
│   ├── curriculum/
│   │   └── scheduler.py         # Adaptive difficulty scheduler
│   ├── tasks/
│   │   └── task_bank.py         # Task definitions across 3 difficulty tiers
│   ├── agent/
│   │   ├── rag_agent.py         # LangGraph-based RAG agent
│   │   └── tool_registry.py     # RETRIEVE / CALCULATE tools
│   └── eval/
│       ├── trace_logger.py      # JSONL episode trace logging
│       ├── judge.py             # LLM-as-judge rubric evaluator
│       └── evaluator.py         # Batch eval loop + MLflow logging
├── scripts/
│   ├── train.py                 # Training loop demo
│   └── evaluate.py              # Evaluation script
└── tests/
    └── test_env.py              # Unit tests (no LLM calls required)
```

---

## Connections to Patronus Research

| This project | Patronus paper |
|---|---|
| Claim-level NLI reward signal | [Lynx](https://arxiv.org/abs/2407.06037) — hallucination evaluation at claim level |
| Multi-turn state tracking | [MEMTRACK](https://arxiv.org/abs/2504.00298) — long-term memory in agent environments |
| Episode trace logging | [TRAIL](https://arxiv.org/abs/2504.04948) — trace reasoning and issue localization |
| LLM-as-judge rubric scoring | [Glider](https://arxiv.org/abs/2410.14669) — grading LLM interactions with explainable ranking |
