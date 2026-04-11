from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph

from app.infrastructure.config.settings import settings
from app.runtime.graph.nodes.bootstrap_request import bootstrap_request_node
from app.runtime.graph.nodes.human_interrupt import check_approval_node, human_interrupt_node
from app.runtime.graph.state import AgentState
from app.skills.common.assemble_prompt import assemble_prompt_node
from app.skills.common.generate import generate_node
from app.skills.common.grader import grader_node
from app.skills.common.router import router_node
from app.skills.memory.retrieve_memories import retrieve_memories_node
from app.skills.profile.retrieve_profile import retrieve_profile_node
from app.skills.rag.retrieve_docs import retrieve_docs_node
from app.skills.research.web_search import web_search_node


def _route_key(state: AgentState) -> Literal["none", "docs", "history", "both"]:
    flags = settings.feature_flags
    enable_docs_rag = flags.enable_docs_rag
    enable_chat_memory = flags.enable_chat_memory
    context = state.get("context") or {}
    route = state.get("route") or context.get("route") or {}
    route_payload = route if isinstance(route, dict) else {}
    needs_docs = bool(route_payload.get("needs_docs")) and enable_docs_rag
    needs_history = bool(route_payload.get("needs_history")) and enable_chat_memory
    if needs_docs and needs_history:
        return "both"
    if needs_docs:
        return "docs"
    if needs_history:
        return "history"
    return "none"


def _after_docs_key(state: AgentState) -> Literal["profile", "memories"]:
    enable_chat_memory = settings.feature_flags.enable_chat_memory
    context = state.get("context") or {}
    route = state.get("route") or context.get("route") or {}
    route_payload = route if isinstance(route, dict) else {}
    if bool(route_payload.get("needs_history")) and enable_chat_memory:
        return "memories"
    return "profile"


def _get_max_self_correction_attempts() -> int:
    return settings.self_correction.max_attempts


def _grader_key(state: AgentState) -> Literal["accept", "rewrite", "search"]:
    trace = state.get("trace") or {}
    attempts = int(trace.get("self_correction_attempts") or 0)
    if attempts >= _get_max_self_correction_attempts():
        return "accept"
    grade_value = (state.get("context") or {}).get("grade")
    grade: dict[str, Any] = cast(dict[str, Any], grade_value) if isinstance(grade_value, dict) else {}
    verdict = str(grade.get("verdict") or "accept").strip().lower()
    if verdict == "search":
        return "search"
    if verdict == "rewrite":
        return "rewrite"
    return "accept"


def _should_interrupt(state: AgentState) -> bool:
    context = state.get("context") or {}
    return bool(context.get("require_human_approval", False))


def _after_generate_key(state: AgentState) -> Literal["interrupt", "grade", "end"]:
    if _should_interrupt(state):
        return "interrupt"
    if settings.feature_flags.enable_self_correction:
        return "grade"
    return "end"


def _check_approval(state: AgentState) -> Literal["approved", "pending"]:
    if state.get("interrupted") is False and str(state.get("next_step") or "") != "wait_approval":
        return "approved"
    action_required = state.get("action_required")
    action_required_payload = action_required if isinstance(action_required, dict) else {}
    if action_required_payload and action_required_payload.get("approved"):
        return "approved"
    return "pending"


def run_app(checkpointer: Any | None = None):
    """
    构建并编译 LangGraph 工作流应用。

    Returns:
        CompiledStateGraph: 编译后的工作流图
    """
    workflow = StateGraph(AgentState)
    flags = settings.feature_flags
    enable_self_correction = flags.enable_self_correction

    workflow.add_node("bootstrap_request", cast(Any, bootstrap_request_node))
    workflow.add_node("router", cast(Any, router_node))
    workflow.add_node("retrieve_docs", cast(Any, retrieve_docs_node))
    workflow.add_node("retrieve_memories", cast(Any, retrieve_memories_node))
    workflow.add_node("retrieve_profile", cast(Any, retrieve_profile_node))
    workflow.add_node("assemble", cast(Any, assemble_prompt_node))
    workflow.add_node("generate", cast(Any, generate_node))
    workflow.add_node("human_interrupt", cast(Any, human_interrupt_node))
    workflow.add_node("check_approval", cast(Any, check_approval_node))
    if enable_self_correction:
        workflow.add_node("grader", cast(Any, grader_node))
        workflow.add_node("web_search", cast(Any, web_search_node))

    workflow.set_entry_point("bootstrap_request")
    workflow.add_edge("bootstrap_request", "router")
    workflow.add_conditional_edges(
        "router",
        _route_key,
        cast(dict[Any, str], {
            "both": "retrieve_docs",
            "docs": "retrieve_docs",
            "history": "retrieve_memories",
            "none": "retrieve_profile",
        }),
    )
    workflow.add_conditional_edges(
        "retrieve_docs",
        _after_docs_key,
        cast(dict[Any, str], {"memories": "retrieve_memories", "profile": "retrieve_profile"}),
    )
    workflow.add_edge("retrieve_memories", "retrieve_profile")
    workflow.add_edge("retrieve_profile", "assemble")
    workflow.add_edge("assemble", "generate")

    generate_routes: dict[Any, str] = {
        "interrupt": "human_interrupt",
        "end": END,
    }
    if enable_self_correction:
        generate_routes["grade"] = "grader"
    workflow.add_conditional_edges(
        "generate",
        _after_generate_key,
        generate_routes,
    )
    workflow.add_edge("human_interrupt", END)
    workflow.add_conditional_edges(
        "check_approval",
        _check_approval,
        cast(dict[Any, str], {"approved": "generate", "pending": END}),
    )

    if enable_self_correction:
        workflow.add_conditional_edges(
            "grader",
            _grader_key,
            cast(dict[Any, str], {"accept": END, "rewrite": "retrieve_profile", "search": "web_search"}),
        )
        workflow.add_edge("web_search", "retrieve_profile")

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return workflow.compile(**compile_kwargs)

# 导出应用实例
app = run_app()
