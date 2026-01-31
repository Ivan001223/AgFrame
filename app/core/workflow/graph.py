from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph, END

from app.core.config.config_manager import config_manager
from app.core.workflow.nodes.assemble_prompt import assemble_prompt_node
from app.core.workflow.nodes.generate import generate_node
from app.core.workflow.nodes.retrieve_docs import retrieve_docs_node
from app.core.workflow.nodes.rerank_docs import rerank_docs_node
from app.core.workflow.nodes.retrieve_memories import retrieve_memories_node
from app.core.workflow.nodes.router import router_node
from app.core.workflow.state import AgentState


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


def _after_docs_key(state: AgentState) -> Literal["assemble", "memories"]:
    cfg = config_manager.get_config() or {}
    flags = cfg.get("feature_flags", {}) or {}
    enable_chat_memory = bool(flags.get("enable_chat_memory", True))
    route = state.get("route") or (state.get("context") or {}).get("route") or {}
    if bool(route.get("needs_history")) and enable_chat_memory:
        return "memories"
    return "assemble"

def run_app():
    """
    构建并编译 LangGraph 工作流应用。
    
    Returns:
        CompiledStateGraph: 编译后的工作流图
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("rerank_docs", rerank_docs_node)
    workflow.add_node("retrieve_memories", retrieve_memories_node)
    workflow.add_node("assemble", assemble_prompt_node)
    workflow.add_node("generate", generate_node)

    # 定义边和流程
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _route_key,
        {
            "both": "retrieve_docs",
            "docs": "retrieve_docs",
            "history": "retrieve_memories",
            "none": "assemble",
        },
    )
    workflow.add_edge("retrieve_docs", "rerank_docs")
    workflow.add_conditional_edges(
        "rerank_docs",
        _after_docs_key,
        {"memories": "retrieve_memories", "assemble": "assemble"},
    )
    workflow.add_edge("retrieve_memories", "assemble")
    workflow.add_edge("assemble", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# 导出应用实例
app = run_app()
