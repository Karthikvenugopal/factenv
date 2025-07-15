"""
LLM-backed RAG agent using LangGraph for state management.

This is the agent that interacts with FactEnv. It wraps an OpenAI-compatible
LLM (or Ollama local model) and exposes act() which maps an env info dict to
a structured action dict.

Graph nodes:
  retrieve  → issue sub-queries to the vector store
  reason    → chain-of-thought reasoning over retrieved docs
  respond   → produce final grounded answer

The graph can be extended with additional tool nodes (calculator, code exec)
without changing the environment interface.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


SYSTEM_PROMPT = """You are a factual QA agent. Your job is to answer questions
accurately based ONLY on the provided context documents.

Rules:
- If the context contains the answer, cite it with [doc_N] notation.
- If the context does NOT contain enough information, say so explicitly.
- Never fabricate facts not present in the context.
- Be concise. Aim for 1-3 sentences.
"""


class AgentState(TypedDict):
    query: str
    docs: list[str]
    history: list[dict]
    reasoning: str
    response: str
    action_type: str


class RAGAgent:
    """
    LangGraph-based RAG agent compatible with both FactualConsistencyEnv
    (single-turn) and MultiTurnFactualEnv (multi-turn tool use).

    Usage:
        agent = RAGAgent(model="gpt-4o-mini")
        action = agent.act(info)   # info from env.reset() or env.step()
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,    # set to http://localhost:11434/v1 for Ollama
        api_key: str = "ollama",
        temperature: float = 0.0,
    ):
        self._llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self._graph = self._build_graph()

    def act(self, info: dict) -> dict | str:
        """
        Given env info, return an action.
        For MultiTurnFactualEnv: returns {"type": ..., "content": ...}
        For FactualConsistencyEnv: returns response string directly.
        """
        state = AgentState(
            query=info.get("query", ""),
            docs=info.get("docs", []),
            history=info.get("history", []),
            reasoning="",
            response="",
            action_type="RESPOND",
        )
        result = self._graph.invoke(state)
        return {"type": result["action_type"], "content": result["response"]}

    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason_node)
        graph.add_node("respond", self._respond_node)
        graph.set_entry_point("reason")
        graph.add_edge("reason", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def _reason_node(self, state: AgentState) -> AgentState:
        context = "\n\n".join(
            f"[doc_{i+1}] {doc}" for i, doc in enumerate(state["docs"])
        )
        history_text = ""
        if state["history"]:
            history_text = "\n".join(
                f"Step {h['step']}: {h['action_type']} — {h.get('content','')[:100]}"
                for h in state["history"]
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context documents:\n{context}\n\n"
                f"{'Prior steps:' + history_text if history_text else ''}\n\n"
                f"Question: {state['query']}\n\n"
                f"First, briefly reason about what the context tells you (1-2 sentences)."
            )),
        ]
        reasoning = self._llm.invoke(messages).content
        return {**state, "reasoning": reasoning}

    def _respond_node(self, state: AgentState) -> AgentState:
        context = "\n\n".join(
            f"[doc_{i+1}] {doc}" for i, doc in enumerate(state["docs"])
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context documents:\n{context}\n\n"
                f"Your reasoning: {state['reasoning']}\n\n"
                f"Question: {state['query']}\n\n"
                f"Now provide your final answer, citing sources with [doc_N]."
            )),
        ]
        response = self._llm.invoke(messages).content
        return {**state, "response": response, "action_type": "RESPOND"}
