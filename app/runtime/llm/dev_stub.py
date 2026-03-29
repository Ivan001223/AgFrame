from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if str(role or "").lower() in {"human", "user"}:
            return _message_text(message).strip()
    return ""


class DevStubChatModel(BaseChatModel):
    model_name: str = "dev-stub"
    streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "dev-stub"

    def _build_reply(self, messages: list[BaseMessage]) -> str:
        prompt = _last_user_message(messages)
        if not prompt:
            return "[dev-stub] ready"

        normalized = " ".join(prompt.split())
        if "json" in normalized.lower():
            return "{}"
        if len(normalized) > 180:
            normalized = normalized[:177].rstrip() + "..."
        return f"[dev-stub] {normalized}"

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._build_reply(messages)
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=reply))
        if run_manager:
            run_manager.on_llm_new_token(reply, chunk=chunk)
        yield chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self._build_reply(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
