from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    """
    通用 Agent 图状态 (Graph State) 定义
    
    Attributes:
        messages: 对话消息列表，使用 add_messages reducer 处理追加逻辑
        next_step: 路由决策的下一步目标
        reasoning: 路由或决策背后的推理过程
        context: 传递的上下文数据字典
        user_id: 当前用户 ID
    """
    # 消息列表：使用 add_messages 策略，新消息会自动追加到列表中
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 核心路由逻辑：记录下一步走向和理由
    next_step: str
    reasoning: str
    
    # 通用上下文：存储跨节点共享的数据
    context: Dict[str, Any]
    user_id: str
    
    # 在此添加自定义状态字段

