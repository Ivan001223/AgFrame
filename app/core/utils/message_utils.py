from typing import Iterable, List, Union, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _content_to_text(content: Any) -> str:
    """将消息内容（可能是多模态列表）转换为纯文本"""
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts)
    if content is None:
        return ""
    return str(content)


def sanitize_messages_for_routing(messages: Iterable[BaseMessage]) -> List[BaseMessage]:
    """
    清洗消息列表以用于路由判断。
    移除多模态内容（图片等），仅保留文本，减少 Token 消耗并提高稳定性。
    
    Args:
        messages: 原始消息列表
        
    Returns:
        List[BaseMessage]: 清洗后的纯文本消息列表
    """
    sanitized: List[BaseMessage] = []
    for msg in messages:
        content = _content_to_text(getattr(msg, "content", ""))
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            sanitized.append(HumanMessage(content=content))
        elif msg_type == "ai":
            sanitized.append(AIMessage(content=content))
        else:
            sanitized.append(HumanMessage(content=f"[{msg_type}]: {content}"))

    return sanitized

