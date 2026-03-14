from __future__ import annotations

from collections.abc import Sized
from typing import Any


def derive_session_title(
    messages: list[dict[str, Any]],
    provided_title: str | None = None,
    *,
    default: str = "新对话",
    max_len: int = 20,
) -> str:
    """
    推导会话标题。
    优先使用提供的标题，否则使用第一条用户消息的前 max_len 个字符。
    
    Args:
        messages: 消息列表
        provided_title: 用户提供的标题（可选）
        default: 默认标题
        max_len: 自动生成标题的最大长度
        
    Returns:
        str: 最终标题
    """
    if provided_title:
        return str(provided_title)
    first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
    if first_user_msg:
        content = str(first_user_msg.get("content", ""))
        return content[:max_len] + "..." if len(content) > max_len else content
    return default


def should_bump_updated_at(
    old_messages: Sized,
    new_messages: Sized,
) -> bool:
    """
    判断是否需要更新会话的 updated_at 时间戳。
    仅当新消息数量多于旧消息数量时（即有新回复），才更新时间戳。
    避免仅因为查看或加载而导致会话被置顶。
    
    Args:
        old_messages: 旧消息列表（或长度）
        new_messages: 新消息列表（或长度）
        
    Returns:
        bool: 是否需要更新时间戳
    """
    return len(new_messages) > len(old_messages)
