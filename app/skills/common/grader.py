from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.infrastructure.config.config_manager import config_manager
from app.runtime.llm.structured_output import StructuredOutputMode, invoke_structured
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState

_log = get_logger("workflow.grader")


class GraderResult(BaseModel):
    verdict: Literal["accept", "rewrite", "search"] = Field(description="下一步动作：通过/重写/搜索后重写")
    reasoning: str = Field(description="简短理由")
    issues: List[str] = Field(default_factory=list, description="问题列表，如 hallucination/not_answered/missing_info 等")
    rewrite_instructions: Optional[str] = Field(default=None, description="如果需要重写，给出改写指令（中文）")
    search_query: Optional[str] = Field(default=None, description="如果需要搜索，给出搜索 query")


def _get_last_user_query(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    return ""


def _get_last_ai_answer(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("ai", "assistant") and content:
            return str(content)
    return ""


def _format_docs(docs: List[Document], max_docs: int = 3, max_chars: int = 1200) -> str:
    out: List[str] = []
    for i, d in enumerate(list(docs)[:max_docs], start=1):
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or meta.get("file_path") or meta.get("url") or ""
        title = meta.get("title") or ""
        content = str(getattr(d, "page_content", "") or "")
        content = content[:max_chars]
        head = f"[Doc {i}] source={source} title={title}".strip()
        out.append(f"{head}\n{content}".strip())
    return "\n\n".join(out)


def _get_structured_mode() -> StructuredOutputMode:
    cfg = config_manager.get_config() or {}
    llm_cfg = cfg.get("llm") or {}
    mode = str(llm_cfg.get("structured_output_mode") or "native_first").strip().lower()
    if mode == StructuredOutputMode.PROMPT_ONLY.value:
        return StructuredOutputMode.PROMPT_ONLY
    return StructuredOutputMode.NATIVE_FIRST


@register_node("grader")
async def grader_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = dict(state.get("context") or {})
    trace = dict(state.get("trace") or {})
    trace_id = trace.get("trace_id") or ctx.get("trace_id")
    user_id = state.get("user_id") or ctx.get("user_id") or "-"
    session_id = ctx.get("session_id") or "-"

    messages = list(state.get("messages") or [])
    question = _get_last_user_query(messages)
    answer = _get_last_ai_answer(messages)
    retrieved_docs = list(state.get("retrieved_docs") or [])
    retrieved_memories = list(state.get("retrieved_memories") or [])
    citations = list(state.get("citations") or [])

    system_template = """你是“回答质量评估器”(grader)。\n\n你会收到用户问题、助手回答、以及可用上下文（检索到的文档/记忆与引用信息）。\n请判断：\n1) 是否回答了问题（覆盖关键点，避免答非所问）。\n2) 是否存在疑似幻觉（无依据事实、与上下文冲突、编造引用/来源）。\n3) 是否需要外部最新信息（例如实时数据、最新版本/政策、近期事件）或缺乏必要事实。\n\n决策规则：\n- 若回答基本正确且不需要补充信息：verdict=accept。\n- 若回答不充分/结构混乱/存在可修正问题：verdict=rewrite，并给出明确 rewrite_instructions。\n- 若缺少关键事实或需要外部信息：verdict=search，并给出 search_query（简短、可用于搜索）。\n\n输出要求：\n- 仅输出一个合法 JSON 对象，必须符合 schema。\n- 不要输出 Markdown、代码块、解释性文字。\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>\n\n<citations>\n{citations}\n</citations>\n\n<retrieved_docs>\n{retrieved_docs}\n</retrieved_docs>\n\n<retrieved_memories>\n{retrieved_memories}\n</retrieved_memories>\n"""

    payload = HumanMessage(
        content=system_template.format(
            question=question,
            answer=answer,
            citations=str(citations),
            retrieved_docs=_format_docs(retrieved_docs),
            retrieved_memories=_format_docs(retrieved_memories),
        )
    )

    try:
        result = await invoke_structured(
            [payload],
            system_template="你只负责输出 JSON。",
            schema=GraderResult,
            fallback_data={
                "verdict": "rewrite",
                "reasoning": "grader_fallback",
                "issues": ["grader_error"],
                "rewrite_instructions": "请基于已给出的上下文，重新回答用户问题；不确定的内容明确说明缺口，不要编造。",
                "search_query": None,
            },
            temperature=0,
            streaming=False,
            mode=_get_structured_mode(),
            sanitize_messages=False,
        )
        ctx["grade"] = result.model_dump()
        if result.verdict == "rewrite":
            ctx["self_correction"] = result.rewrite_instructions or ""
        elif result.verdict == "search":
            ctx["search_query"] = result.search_query or question
            ctx["self_correction"] = result.rewrite_instructions or ""

        attempts = int(trace.get("self_correction_attempts") or 0)
        if result.verdict != "accept":
            attempts += 1
        trace["self_correction_attempts"] = attempts

        bind_logger(
            _log,
            trace_id=str(trace_id or "-"),
            user_id=str(user_id),
            session_id=str(session_id),
            node="grader",
        ).info(
            "graded verdict=%s attempts=%d cost_ms=%d",
            result.verdict,
            attempts,
            int((time.perf_counter() - t0) * 1000),
        )
        return {"context": ctx, "trace": trace}
    except Exception as e:
        errors = list(state.get("errors") or [])
        errors.append(f"grader_error: {e}")
        ctx["grade"] = {
            "verdict": "rewrite",
            "reasoning": f"Error: {e}",
            "issues": ["grader_exception"],
            "rewrite_instructions": "请基于已给出的上下文，重新回答用户问题；不确定的内容明确说明缺口，不要编造。",
            "search_query": None,
        }
        bind_logger(
            _log,
            trace_id=str(trace_id or "-"),
            user_id=str(user_id),
            session_id=str(session_id),
            node="grader",
        ).info("graded exception cost_ms=%d", int((time.perf_counter() - t0) * 1000))
        return {"context": ctx, "errors": errors, "trace": trace}

