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

# 初始化全局服务实例
# 聊天摘要索引服务，用于检索历史对话摘要
_chat_summary_index = ChatSummaryIndex()
# 用户画像引擎，用于获取和管理用户画像
_profile_engine = UserProfileEngine()


def _get_last_user_query(messages: List[BaseMessage]) -> str:
    """
    从消息列表中获取最后一条用户的查询内容。
    
    Args:
        messages: 消息对象列表
        
    Returns:
        str: 用户的最后一条查询文本，如果未找到则返回空字符串
    """
    for m in reversed(messages):
        # 兼容不同类型的消息对象属性
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    # 如果没找到明确的用户消息，尝试取最后一条消息的内容
    last = messages[-1]
    return str(getattr(last, "content", "") or "")


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    路由节点：分析用户意图，决定是否需要检索文档或历史记录。
    
    Args:
        state: 当前 Agent 的状态
        
    Returns:
        Dict: 更新后的状态上下文，包含路由决策结果
    """
    # 调用内存路由逻辑，判断是否需要检索文档或历史
    decision = route_memory(state)
    ctx = dict(state.get("context") or {})
    # 将路由决策结果存入上下文
    ctx["route"] = {"needs_docs": decision.needs_docs, "needs_history": decision.needs_history, "reasoning": decision.reasoning}
    return {"context": ctx}


def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    检索节点：根据路由结果，并行检索文档和历史记忆。
    
    Args:
        state: 当前 Agent 的状态
        
    Returns:
        Dict: 更新后的状态上下文，包含检索到的文档和记忆
    """
    messages = state.get("messages") or []
    query = _get_last_user_query(messages)
    ctx = dict(state.get("context") or {})
    route = ctx.get("route") or {}
    needs_docs = bool(route.get("needs_docs"))
    needs_history = bool(route.get("needs_history"))

    user_id = state.get("user_id") or ctx.get("user_id") or "default"

    def fetch_docs():
        """检索知识库文档"""
        if not needs_docs:
            return []
        # 使用 RAG 引擎检索上下文
        return get_rag_engine().retrieve_context(query, k=3, fetch_k=20)

    def fetch_history():
        """检索历史对话摘要"""
        if not needs_history:
            return []
        # 使用摘要索引检索历史记忆
        return _chat_summary_index.retrieve(user_id=user_id, query=query, k=3, fetch_k=20)

    # 使用线程池并行执行检索任务
    with ThreadPoolExecutor(max_workers=2) as ex:
        docs_f = ex.submit(fetch_docs)
        mem_f = ex.submit(fetch_history)
        docs = docs_f.result()
        memories = mem_f.result()

    ctx["retrieved_docs"] = docs
    ctx["retrieved_memories"] = memories
    return {"context": ctx}


def assemble_prompt_node(state: AgentState) -> Dict[str, Any]:
    """
    提示词组装节点：整合用户画像、近期历史、检索文档和记忆，构建系统提示词。
    
    Args:
        state: 当前 Agent 的状态
        
    Returns:
        Dict: 更新后的状态上下文，包含构建好的系统提示词
    """
    ctx = dict(state.get("context") or {})
    user_id = state.get("user_id") or ctx.get("user_id") or "default"
    session_id = ctx.get("session_id")

    # 获取最近的对话历史（即时上下文）
    recent_history_lines: List[str] = []
    if session_id and ensure_schema_if_possible():
        try:
            store = MySQLConversationStore()
            recent_msgs = store.get_recent_messages(user_id=user_id, session_id=str(session_id), limit_messages=10)
            for m in recent_msgs:
                recent_history_lines.append(f"{m.get('role')}: {m.get('content')}")
        except Exception:
            recent_history_lines = []

    # 获取用户画像信息
    try:
        profile = _profile_engine.get_profile(user_id)
    except Exception:
        profile = {"basic_info": {}, "tech_profile": {}, "preferences": {}, "facts": []}

    docs = ctx.get("retrieved_docs") or []
    memories = ctx.get("retrieved_memories") or []

    # 格式化文档块
    doc_block_lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        ref = f"doc_id={meta.get('doc_id')}, parent_chunk_id={meta.get('parent_chunk_id')}, page={meta.get('page_num')}"
        doc_block_lines.append(f"[Doc {i}] ({ref})\n{d.page_content}")

    # 格式化记忆块
    mem_block_lines: List[str] = []
    for i, m in enumerate(memories, start=1):
        meta = getattr(m, "metadata", {}) or {}
        ref = f"session_id={meta.get('session_id')}, msg_range={meta.get('start_msg_id')}..{meta.get('end_msg_id')}"
        mem_block_lines.append(f"[Memory {i}] ({ref})\n{m.page_content}")

    # 构建完整的系统提示词
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
    """
    生成节点：调用 LLM 生成最终回复。
    
    Args:
        state: 当前 Agent 的状态
        
    Returns:
        Dict: 包含生成回复的消息列表
    """
    ctx = state.get("context") or {}
    system_prompt = ctx.get("system_prompt") or "你是一个助理。"
    # 获取 LLM 实例，禁用随机性（temperature=0）并开启流式输出
    llm = get_llm(temperature=0, streaming=True)
    messages: List[BaseMessage] = list(state.get("messages") or [])
    # 调用 LLM，传入系统提示词和当前对话消息
    response = llm.invoke([SystemMessage(content=system_prompt), *messages])
    return {"messages": [response]}

def run_app():
    """
    构建并编译 LangGraph 工作流应用。
    
    Returns:
        CompiledStateGraph: 编译后的工作流图
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)      # 路由分析
    workflow.add_node("retrieve", retrieve_context_node) # 上下文检索
    workflow.add_node("assemble", assemble_prompt_node)  # 提示词组装
    workflow.add_node("generate", generate_node)  # 回复生成

    # 定义边和流程
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "assemble")
    workflow.add_edge("assemble", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# 导出应用实例
app = run_app()
