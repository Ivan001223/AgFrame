from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from app.core.workflow.json_router import run_json_router


class MemoryRouteDecision(BaseModel):
    needs_docs: bool = Field(description="Whether to retrieve from knowledge base documents")
    needs_history: bool = Field(description="Whether to retrieve from conversation memory (summaries)")
    reasoning: str = Field(description="Short reasoning")


def route_memory(state: Dict[str, Any]) -> MemoryRouteDecision:
    messages = state.get("messages", [])
    if not messages:
        return MemoryRouteDecision(needs_docs=False, needs_history=False, reasoning="No messages")

    system_template = (
        "你是意图识别与路由器。\n"
        "判断用户这一轮问题是否需要：\n"
        "1) 检索用户上传的静态文档（needs_docs）\n"
        "2) 检索更早的对话记忆摘要（needs_history）\n\n"
        "判断依据：\n"
        "- 文档：提到“文档/上传/条款/参数/第几页/手册/说明书/规范”等，或明确要问文档内容。\n"
        "- 历史：提到“上次/之前/还记得/我遇到的报错/你说过/我们聊过”等，或需要回忆旧对话。\n"
        "- 两者都需要：同时提到文档与上次/之前。\n\n"
        "输出要求：仅输出合法 JSON，不要输出其他文字。\n"
        "JSON 字段：needs_docs(boolean), needs_history(boolean), reasoning(string)\n"
    )
    return run_json_router(
        messages,
        system_template=system_template,
        schema=MemoryRouteDecision,
        fallback_data={"needs_docs": False, "needs_history": False, "reasoning": "Error: {error}"},
        temperature=0,
        streaming=False,
        json_mode=True,
    )
