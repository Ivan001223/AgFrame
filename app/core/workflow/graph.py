from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage

from app.core.workflow.state import AgentState
from app.core.workflow.memory_router import route_memory
from app.core.services.rag_engine import get_rag_engine
from app.core.services.chat_memory_engine import ChatSummaryIndex
from app.core.services.profile_engine import UserProfileEngine
from app.core.llm.llm_factory import get_llm
from app.core.database.schema import ensure_schema_if_possible
from app.core.database.stores import MySQLConversationStore

_chat_summary_index = ChatSummaryIndex()
_profile_engine = UserProfileEngine()


def _get_last_user_query(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    last = messages[-1]
    return str(getattr(last, "content", "") or "")


def router_node(state: AgentState) -> Dict[str, Any]:
    decision = route_memory(state)
    ctx = dict(state.get("context") or {})
    ctx["route"] = {"needs_docs": decision.needs_docs, "needs_history": decision.needs_history, "reasoning": decision.reasoning}
    return {"context": ctx}


def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages") or []
    query = _get_last_user_query(messages)
    ctx = dict(state.get("context") or {})
    route = ctx.get("route") or {}
    needs_docs = bool(route.get("needs_docs"))
    needs_history = bool(route.get("needs_history"))

    user_id = state.get("user_id") or ctx.get("user_id") or "default"

    def fetch_docs():
        if not needs_docs:
            return []
        return get_rag_engine().retrieve_context(query, k=3, fetch_k=20)

    def fetch_history():
        if not needs_history:
            return []
        return _chat_summary_index.retrieve(user_id=user_id, query=query, k=3, fetch_k=20)

    with ThreadPoolExecutor(max_workers=2) as ex:
        docs_f = ex.submit(fetch_docs)
        mem_f = ex.submit(fetch_history)
        docs = docs_f.result()
        memories = mem_f.result()

    ctx["retrieved_docs"] = docs
    ctx["retrieved_memories"] = memories
    return {"context": ctx}


def assemble_prompt_node(state: AgentState) -> Dict[str, Any]:
    ctx = dict(state.get("context") or {})
    user_id = state.get("user_id") or ctx.get("user_id") or "default"
    session_id = ctx.get("session_id")

    recent_history_lines: List[str] = []
    if session_id and ensure_schema_if_possible():
        try:
            store = MySQLConversationStore()
            recent_msgs = store.get_recent_messages(user_id=user_id, session_id=str(session_id), limit_messages=10)
            for m in recent_msgs:
                recent_history_lines.append(f"{m.get('role')}: {m.get('content')}")
        except Exception:
            recent_history_lines = []

    try:
        profile = _profile_engine.get_profile(user_id)
    except Exception:
        profile = {"basic_info": {}, "tech_profile": {}, "preferences": {}, "facts": []}

    docs = ctx.get("retrieved_docs") or []
    memories = ctx.get("retrieved_memories") or []

    doc_block_lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        ref = f"doc_id={meta.get('doc_id')}, parent_chunk_id={meta.get('parent_chunk_id')}, page={meta.get('page_num')}"
        doc_block_lines.append(f"[Doc {i}] ({ref})\n{d.page_content}")

    mem_block_lines: List[str] = []
    for i, m in enumerate(memories, start=1):
        meta = getattr(m, "metadata", {}) or {}
        ref = f"session_id={meta.get('session_id')}, msg_range={meta.get('start_msg_id')}..{meta.get('end_msg_id')}"
        mem_block_lines.append(f"[Memory {i}] ({ref})\n{m.page_content}")

    system_prompt = (
        "你是一个严谨的助理。回答时优先使用提供的上下文与用户画像。\n"
        "当引用文档内容时，尽量给出对应 Doc 编号；当引用历史记忆时，尽量给出 Memory 编号。\n"
        "如果上下文不足以回答细节，明确说明缺失点并给出下一步需要的信息。\n\n"
        f"<user_profile>\n{profile}\n</user_profile>\n\n"
        f"<recent_history>\n{chr(10).join(recent_history_lines) if recent_history_lines else ''}\n</recent_history>\n\n"
        f"<retrieved_docs>\n{'\n\n'.join(doc_block_lines) if doc_block_lines else ''}\n</retrieved_docs>\n\n"
        f"<retrieved_memories>\n{'\n\n'.join(mem_block_lines) if mem_block_lines else ''}\n</retrieved_memories>\n"
    )

    ctx["system_prompt"] = system_prompt
    return {"context": ctx}


def generate_node(state: AgentState) -> Dict[str, Any]:
    ctx = state.get("context") or {}
    system_prompt = ctx.get("system_prompt") or "你是一个助理。"
    llm = get_llm(temperature=0, streaming=True)
    messages: List[BaseMessage] = list(state.get("messages") or [])
    response = llm.invoke([SystemMessage(content=system_prompt), *messages])
    return {"messages": [response]}

def run_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_context_node)
    workflow.add_node("assemble", assemble_prompt_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "assemble")
    workflow.add_edge("assemble", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# 导出应用
app = run_app()
