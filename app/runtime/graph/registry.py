from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.runtime.graph.state import AgentState

NodeFn = Callable[[AgentState], dict[str, Any] | Awaitable[dict[str, Any]]]


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}

    def register(self, name: str, fn: NodeFn) -> None:
        if name in self._nodes:
            raise ValueError(f"Node already registered: {name}")
        self._nodes[name] = fn

    def get(self, name: str) -> NodeFn:
        try:
            return self._nodes[name]
        except KeyError as e:
            raise KeyError(f"Node not found: {name}") from e

    def maybe_get(self, name: str) -> NodeFn | None:
        return self._nodes.get(name)


node_registry = NodeRegistry()


def register_node(name: str) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        node_registry.register(name, fn)
        return fn

    return decorator
