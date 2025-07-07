"""
Tool registry for the multi-turn environment.

Provides RETRIEVE and CALCULATE tools that the LLM agent can call via
structured action dicts. Designed to be swappable: plug in a real vector
store (Weaviate/Pinecone) for production, use the mock for tests/dev.
"""

from __future__ import annotations

import ast
import operator as op
from typing import Protocol


class VectorStore(Protocol):
    def search(self, query: str, top_k: int) -> list[str]: ...


class MockVectorStore:
    """Returns a fixed passage for any query — used in tests and demos."""

    def __init__(self, passages: list[str]):
        self._passages = passages

    def search(self, query: str, top_k: int) -> list[str]:
        return self._passages[:top_k]


_SAFE_OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.Pow:  op.pow,
    ast.USub: op.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression without exec/eval abuse."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    return _eval(ast.parse(expr, mode="eval").body)


class ToolRegistry:
    def __init__(self, vector_store: VectorStore | None = None):
        self._store = vector_store or MockVectorStore([])

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return self._store.search(query, top_k)

    def calculate(self, expression: str) -> str:
        try:
            result = _safe_eval(expression)
            return str(result)
        except Exception as e:
            return f"calculation error: {e}"

    def use(self, tool_name: str, **kwargs) -> str:
        if tool_name == "retrieve":
            return str(self.retrieve(**kwargs))
        if tool_name == "calculate":
            return self.calculate(**kwargs)
        raise ValueError(f"Unknown tool: {tool_name}")
