from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    """
    通用 Agent 图状态
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 核心路由逻辑
    next_step: str
    reasoning: str
    
    # 通用上下文
    context: Dict[str, Any]
    user_id: str
    
    # 在此添加自定义状态字段
