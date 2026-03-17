from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class RerankerRuntimeStatus(TypedDict):
    configured: bool
    model_name: str | None
    pruning_scoring_source: str


class RuntimeStatus(TypedDict):
    reranker: RerankerRuntimeStatus


class SettingsPayload(TypedDict, total=False):
    runtime_status: RuntimeStatus


def build_runtime_status(config: dict[str, Any]) -> RuntimeStatus:
    from app.runtime.llm.reranker import ModelReranker

    reranker = ModelReranker(config=config)
    model_name = str(getattr(reranker, "model_name", "") or "").strip()
    configured = not bool(getattr(reranker, "_disabled", True))
    return {
        "reranker": {
            "configured": configured,
            "model_name": model_name or None,
            "pruning_scoring_source": "reranker_model" if configured else "local_phrase_fallback",
        },
    }


def dump_settings_with_runtime_status(raw_settings: Any) -> SettingsPayload:
    payload: SettingsPayload = raw_settings.model_dump()
    payload["runtime_status"] = build_runtime_status(payload)
    return payload
