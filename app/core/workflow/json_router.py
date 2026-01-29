from __future__ import annotations

from typing import Any, Dict, Iterable, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.core.llm.llm_factory import get_llm
from app.core.utils.json_parser import parse_json_from_llm
from app.core.utils.message_utils import sanitize_messages_for_routing

T = TypeVar("T", bound=BaseModel)


def run_json_router(
    messages: Iterable[Any],
    *,
    system_template: str,
    schema: Type[T],
    fallback_data: Dict[str, Any],
    temperature: float = 0,
    streaming: bool = False,
    json_mode: bool = True,
) -> T:
    llm = get_llm(temperature=temperature, streaming=streaming, json_mode=json_mode)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm

    try:
        sanitized_messages = sanitize_messages_for_routing(messages)
        response = chain.invoke({"messages": sanitized_messages})
        data = parse_json_from_llm(str(getattr(response, "content", response)))
        return schema(**data)
    except Exception as e:
        fallback = dict(fallback_data)
        if "reasoning" in schema.model_fields and "reasoning" not in fallback:
            fallback["reasoning"] = f"Error: {e}"
        elif "reasoning" in fallback:
            fallback["reasoning"] = str(fallback["reasoning"]).format(error=e)
        return schema(**fallback)

