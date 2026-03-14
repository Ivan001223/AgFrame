from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.infrastructure.utils.json_parser import parse_json_from_llm
from app.infrastructure.utils.message_utils import sanitize_messages_for_routing
from app.runtime.llm.llm_factory import get_llm

T = TypeVar("T", bound=BaseModel)


def run_json_router(
    messages: Iterable[Any],
    *,
    system_template: str,
    schema: type[T],
    fallback_data: dict[str, Any],
    temperature: float = 0,
    streaming: bool = False,
    json_mode: bool = True,
) -> T:
    """
    通用 JSON 路由器：使用 LLM 根据系统提示词和对话历史生成结构化 JSON 输出。
    
    Args:
        messages: 对话历史消息
        system_template: 系统提示词模板，定义路由规则和角色
        schema: 期望输出的 Pydantic 模型类
        fallback_data: 当解析失败或发生异常时的回退数据
        temperature: LLM 温度参数，默认为 0（确定性）
        streaming: 是否流式输出
        json_mode: 是否启用 LLM 的 JSON 模式
        
    Returns:
        T: 解析后的 Pydantic 模型实例
    """
    llm = get_llm(temperature=temperature, streaming=streaming, json_mode=json_mode)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm

    try:
        # 清洗消息，移除可能干扰路由的复杂内容
        sanitized_messages = sanitize_messages_for_routing(messages)
        response = chain.invoke({"messages": sanitized_messages})
        # 解析 LLM 返回的 JSON 字符串
        data = parse_json_from_llm(str(getattr(response, "content", response)))
        return schema(**data)
    except Exception as e:
        # 异常处理：使用回退数据
        fallback = dict(fallback_data)
        if "reasoning" in schema.model_fields and "reasoning" not in fallback:
            fallback["reasoning"] = f"Error: {e}"
        elif "reasoning" in fallback:
            fallback["reasoning"] = str(fallback["reasoning"]).format(error=e)
        return schema(**fallback)

