from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, TypeVar

import anyio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.infrastructure.utils.json_parser import parse_json_from_llm
from app.infrastructure.utils.message_utils import sanitize_messages_for_routing
from app.runtime.llm.llm_factory import get_llm

T = TypeVar("T", bound=BaseModel)


class StructuredOutputMode(str, Enum):
    NATIVE_FIRST = "native_first"
    PROMPT_ONLY = "prompt_only"


async def invoke_structured(
    messages: Iterable[Any],
    *,
    system_template: str,
    schema: type[T],
    fallback_data: dict[str, Any],
    temperature: float = 0,
    streaming: bool = False,
    mode: StructuredOutputMode = StructuredOutputMode.NATIVE_FIRST,
    sanitize_messages: bool = True,
) -> T:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prepared_messages: list[Any]
    if sanitize_messages:
        prepared_messages = sanitize_messages_for_routing(messages)
    else:
        prepared_messages = list(messages)

    async def _invoke_with_llm(*, json_mode: bool, use_with_structured: bool) -> T:
        llm = get_llm(temperature=temperature, streaming=streaming, json_mode=json_mode)
        with_structured = getattr(llm, "with_structured_output", None)
        if use_with_structured and callable(with_structured):
            try:
                structured_llm = with_structured(schema)
                chain = prompt | structured_llm
                result = await anyio.to_thread.run_sync(lambda: chain.invoke({"messages": prepared_messages}))
                if isinstance(result, schema):
                    return result
            except Exception:
                pass

        chain = prompt | llm
        response = await anyio.to_thread.run_sync(lambda: chain.invoke({"messages": prepared_messages}))
        raw = str(getattr(response, "content", response))
        data = parse_json_from_llm(raw)
        if isinstance(data, dict):
            return schema(**data)
        return schema(**fallback_data)

    def _fallback_model(error: Exception) -> T:
        fallback = dict(fallback_data)
        if "reasoning" in schema.model_fields and "reasoning" not in fallback:
            fallback["reasoning"] = f"Error: {error}"
        elif "reasoning" in fallback:
            fallback["reasoning"] = str(fallback["reasoning"]).format(error=error)
        return schema(**fallback)

    try:
        if mode == StructuredOutputMode.PROMPT_ONLY:
            return await _invoke_with_llm(json_mode=False, use_with_structured=False)
        try:
            return await _invoke_with_llm(json_mode=True, use_with_structured=True)
        except Exception:
            return await _invoke_with_llm(json_mode=False, use_with_structured=False)
    except Exception as e:
        return _fallback_model(e)
