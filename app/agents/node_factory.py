from typing import Any, Callable, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.runtime.llm.llm_factory import get_llm
from app.runtime.graph.state import AgentState


def build_system_prompt_template(system_prompt: str, messages_key: str = "messages") -> ChatPromptTemplate:
    """构建包含系统提示词和消息占位符的 Prompt 模板"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name=messages_key),
        ]
    )


def build_llm_chain(
    system_prompt: str,
    *,
    temperature: float = 0,
    tools: Optional[Sequence[Any]] = None,
    json_mode: bool = False,
):
    """
    构建标准的 LLM 执行链。
    Prompt -> LLM (bind tools)
    
    Args:
        system_prompt: 系统提示词
        temperature: 温度参数
        tools: 可用工具列表
        json_mode: 是否启用 JSON 模式
        
    Returns:
        Runnable: 可执行的 LangChain 对象
    """
    llm = get_llm(temperature=temperature, json_mode=json_mode)
    if tools:
        llm = llm.bind_tools(list(tools))
    prompt = build_system_prompt_template(system_prompt)
    return prompt | llm


def make_agent_node(chain, *, messages_key: str = "messages") -> Callable[[AgentState], dict]:
    """
    创建符合 LangGraph 签名的节点函数。
    
    Args:
        chain: LLM 执行链
        messages_key: 状态中存储消息的键名
        
    Returns:
        Callable: 节点函数，输入 State，输出更新后的 State
    """
    def node(state: AgentState):
        messages = state[messages_key]
        response = chain.invoke({messages_key: messages})
        return {"messages": [response]}

    return node

