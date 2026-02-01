from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph, END

from app.infrastructure.config.config_manager import config_manager
from app.skills.common.assemble_prompt import assemble_prompt_node
from app.skills.common.grader import grader_node
from app.skills.common.generate import generate_node
from app.skills.rag.retrieve_docs import retrieve_docs_node
from app.skills.rag.rerank_docs import rerank_docs_node
from app.skills.memory.retrieve_memories import retrieve_memories_node
from app.skills.profile.retrieve_profile import retrieve_profile_node
from app.skills.common.router import router_node
from app.skills.research.web_search import web_search_node
from app.runtime.graph.state import AgentState


def _route_key(state: AgentState) -> Literal["none", "docs", "history", "both"]:
    cfg = config_manager.get_config() or {}
    flags = cfg.get("feature_flags", {}) or {}
    enable_docs_rag = bool(flags.get("enable_docs_rag", True))
    enable_chat_memory = bool(flags.get("enable_chat_memory", True))
    route = state.get("route") or (state.get("context") or {}).get("route") or {}
    needs_docs = bool(route.get("needs_docs")) and enable_docs_rag
    needs_history = bool(route.get("needs_history")) and enable_chat_memory
    if needs_docs and needs_history:
        return "both"
    if needs_docs:
        return "docs"
    if needs_history:
        return "history"
    return "none"


def _after_docs_key(state: AgentState) -> Literal["profile", "memories"]:
    cfg = config_manager.get_config() or {}
    flags = cfg.get("feature_flags", {}) or {}
    enable_chat_memory = bool(flags.get("enable_chat_memory", True))
    route = state.get("route") or (state.get("context") or {}).get("route") or {}
    if bool(route.get("needs_history")) and enable_chat_memory:
        return "memories"
    return "profile"


def _get_max_self_correction_attempts() -> int:
    cfg = config_manager.get_config() or {}
    sc_cfg = cfg.get("self_correction", {}) or {}
    val = sc_cfg.get("max_attempts")
    if val is None:
        return 2
    try:
        return max(0, int(val))
    except Exception:
        return 2


def _grader_key(state: AgentState) -> Literal["accept", "rewrite", "search"]:
    trace = state.get("trace") or {}
    attempts = int(trace.get("self_correction_attempts") or 0)
    if attempts >= _get_max_self_correction_attempts():
        return "accept"
    grade = (state.get("context") or {}).get("grade") or {}
    verdict = str(grade.get("verdict") or "accept").strip().lower()
    if verdict == "search":
        return "search"
    if verdict == "rewrite":
        return "rewrite"
    return "accept"


def run_app():
    """
    构建并编译 LangGraph 工作流应用。
    
    Returns:
        CompiledStateGraph: 编译后的工作流图
    """
    workflow = StateGraph(AgentState)
    cfg = config_manager.get_config() or {}
    flags = cfg.get("feature_flags", {}) or {}
    enable_self_correction = bool(flags.get("enable_self_correction", False))

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("rerank_docs", rerank_docs_node)
    workflow.add_node("retrieve_memories", retrieve_memories_node)
    workflow.add_node("retrieve_profile", retrieve_profile_node)
    workflow.add_node("assemble", assemble_prompt_node)
    workflow.add_node("generate", generate_node)
    if enable_self_correction:
        workflow.add_node("grader", grader_node)
        workflow.add_node("web_search", web_search_node)

    # 定义边和流程
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _route_key,
        {
            "both": "retrieve_docs",
            "docs": "retrieve_docs",
            "history": "retrieve_memories",
            "none": "retrieve_profile",
        },
    )
    workflow.add_edge("retrieve_docs", "rerank_docs")
    workflow.add_conditional_edges(
        "rerank_docs",
        _after_docs_key,
        {"memories": "retrieve_memories", "profile": "retrieve_profile"},
    )
    workflow.add_edge("retrieve_memories", "retrieve_profile")
    workflow.add_edge("retrieve_profile", "assemble")
    workflow.add_edge("assemble", "generate")
    if enable_self_correction:
        workflow.add_edge("generate", "grader")
        workflow.add_conditional_edges(
            "grader",
            _grader_key,
            {"accept": END, "rewrite": "retrieve_profile", "search": "web_search"},
        )
        workflow.add_edge("web_search", "retrieve_profile")
    else:
        workflow.add_edge("generate", END)

    return workflow.compile()

# 导出应用实例
app = run_app()
