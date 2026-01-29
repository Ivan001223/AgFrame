from typing import Iterable, List, Union, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _content_to_text(content: Any) -> str:
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

