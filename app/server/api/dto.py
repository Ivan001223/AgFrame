from __future__ import annotations

import os
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
    reranker_cfg = config.get("reranker") or {}
    local_models = config.get("local_models") or {}
    env_var = str(reranker_cfg.get("env_var") or "MODEL_PATH_RERANKER").strip()
    explicit_model = str(reranker_cfg.get("model_name") or local_models.get("rerank_model") or "").strip()
    env_model = str(os.getenv(env_var, "")).strip() if env_var else ""
    model_name = explicit_model or env_model
    configured = bool(model_name)
    return {
        "reranker": {
            "configured": configured,
            "model_name": model_name or None,
            "pruning_scoring_source": "lightweight_ranker",
        },
    }


def dump_settings_with_runtime_status(raw_settings: Any) -> SettingsPayload:
    payload: SettingsPayload = raw_settings.model_dump()
    payload["runtime_status"] = build_runtime_status(payload)
    return payload
