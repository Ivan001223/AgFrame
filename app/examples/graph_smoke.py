import sys
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast

import anyio

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


async def main():
    from app.runtime.graph.state import AgentState
    from app.skills.common.router import router_node

    state = {
        "messages": [],
        "user_id": "smoke",
        "route": {"needs_docs": False, "needs_history": False, "reasoning": "smoke"},
        "context": {"session_id": "s-smoke"},
    }
    result = router_node(cast(AgentState, state))
    out: dict[str, Any]
    if isawaitable(result):
        out = await result
    else:
        out = result
    if out["route"]["needs_docs"] is not False:
        raise RuntimeError("router smoke expected needs_docs=False")
    if out["route"]["needs_history"] is not False:
        raise RuntimeError("router smoke expected needs_history=False")
    if "trace" not in out or not out["trace"].get("trace_id"):
        raise RuntimeError("router smoke expected trace.trace_id to exist")


if __name__ == "__main__":
    anyio.run(main)
