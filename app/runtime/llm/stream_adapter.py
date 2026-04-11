from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from inspect import isawaitable
from typing import Any


def coerce_stream_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                nested = getattr(item, "text", None)
                if nested:
                    parts.append(str(nested))
        return "".join(parts)
    return str(content or "")


async def stream_llm_events(
    llm: Any,
    messages: list[Any],
    *,
    on_event: Callable[[dict[str, Any]], Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    event_callback = on_event
    event_index = 0
    token_index = 0

    async def _emit(event: dict[str, Any]) -> None:
        nonlocal event_index
        payload = {"event_index": event_index, **event}
        event_index += 1
        if event_callback is not None:
            result = event_callback(payload)
            if isawaitable(result):
                await result

    stream_events_method = getattr(llm, "astream_events", None)
    if callable(stream_events_method):
        async for raw_event in stream_events_method(messages):
            raw = dict(raw_event or {}) if isinstance(raw_event, dict) else {"raw_event": raw_event}
            event_name = str(raw.get("event") or raw.get("event_type") or "provider_event")
            data = dict(raw.get("data") or {}) if isinstance(raw.get("data"), dict) else {}
            chunk = data.get("chunk") or data.get("output") or data.get("delta") or raw.get("chunk")
            text = coerce_stream_text(chunk)
            emitted = {
                "type": "provider_event",
                "provider_event_type": event_name,
                "text": text,
                "raw": raw,
            }
            await _emit(emitted)
            yield emitted
            if text:
                token_event = {
                    "type": "token",
                    "provider_event_type": event_name,
                    "token_index": token_index,
                    "text": text,
                    "raw": raw,
                }
                token_index += 1
                await _emit(token_event)
                yield token_event
        return

    stream_method = getattr(llm, "astream", None)
    if callable(stream_method):
        async for chunk in stream_method(messages):
            text = coerce_stream_text(chunk)
            event = {
                "type": "token",
                "provider_event_type": "astream_chunk",
                "token_index": token_index,
                "text": text,
                "raw": chunk,
            }
            token_index += 1
            await _emit(event)
            yield event

