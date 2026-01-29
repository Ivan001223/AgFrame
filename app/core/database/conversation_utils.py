from __future__ import annotations

from collections.abc import Sized
from typing import Any, Dict, List, Optional


def derive_session_title(
    messages: List[Dict[str, Any]],
    provided_title: Optional[str] = None,
    *,
    default: str = "新对话",
    max_len: int = 20,
) -> str:
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
    return len(new_messages) > len(old_messages)
