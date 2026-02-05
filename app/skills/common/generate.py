from __future__ import annotations

from typing import Any, Dict, List

import anyio
import time
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, convert_to_messages

from app.runtime.llm.llm_factory import get_llm
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState

_log = get_logger("workflow.generate")


@register_node("generate")
async def generate_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = state.get("context") or {}
    system_prompt = ctx.get("system_prompt") or "你是一个助理。"
    llm = get_llm(temperature=0, streaming=True)
    messages: List[BaseMessage] = list(state.get("messages") or [])
    
    # Convert BaseMessage subclasses to correct types (fix pydantic validation issue)
    raw_messages = []
    for m in messages:
        if hasattr(m, 'dict'):
            raw_messages.append(m.dict())
        elif hasattr(m, '__dict__'):
            raw_messages.append(dict(m))
        elif isinstance(m, dict):
            raw_messages.append(m)
        else:
            _log.warning(f"Unknown message type: {type(m)}, skipping")
    converted_messages = convert_to_messages(raw_messages)
    
    _log.info(f"Message count: {len(converted_messages)}")
    
    response = await anyio.to_thread.run_sync(
        lambda: llm.invoke([SystemMessage(content=system_prompt), *converted_messages])
    )
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    user_id = state.get("user_id") or ctx.get("user_id") or "-"
    session_id = ctx.get("session_id") or "-"
    bind_logger(_log, trace_id=str(trace_id or "-"), user_id=str(user_id), session_id=str(session_id), node="generate").info(
        "generated cost_ms=%d", int((time.perf_counter() - t0) * 1000)
    )
    return {"messages": [response]}
