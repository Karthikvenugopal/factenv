# FactEnv

A Gymnasium-compatible reinforcement learning environment for training and
evaluating factually grounded RAG agents.

The core idea: frame retrieval-augmented generation as a Markov Decision
Process where the reward signal is a claim-level factual-consistency score, so
hallucinations are penalized at the individual-claim level rather than across
the whole response.

## Status

Early scaffold. Planned components:

- `env/`        — Gymnasium environments (single-turn + multi-turn tool use)
- `rewards/`    — claim-level NLI reward + composite reward shaping
- `curriculum/` — adaptive difficulty scheduler
- `tasks/`      — task bank across easy / medium / hard tiers
- `agent/`      — LangGraph RAG agent + tool registry
- `eval/`       — trace logging, LLM-as-judge, batch evaluation

## Install

```bash
pip install -e .
```
