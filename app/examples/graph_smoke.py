import sys
from pathlib import Path

import anyio

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.core.workflow.nodes.router import router_node


async def main():
    state = {
        "messages": [],
        "user_id": "smoke",
        "route": {"needs_docs": False, "needs_history": False, "reasoning": "smoke"},
        "context": {"session_id": "s-smoke"},
    }
    out = await router_node(state)
    assert out["route"]["needs_docs"] is False
    assert out["route"]["needs_history"] is False
    assert "trace" in out and out["trace"].get("trace_id")


if __name__ == "__main__":
    anyio.run(main)
