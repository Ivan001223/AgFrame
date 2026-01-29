from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ImportedModel:
    pretrained_source: str
    provider: str
    model_ref: str


def _snapshot_modelscope(model_id: str, *, cache_dir: Optional[str] = None, revision: Optional[str] = None) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as e:
        raise RuntimeError("modelscope 未安装或不可用") from e

    kwargs: dict[str, Any] = {"model_id": model_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    return snapshot_download(**kwargs)


def _snapshot_huggingface(repo_id: str, *, cache_dir: Optional[str] = None, revision: Optional[str] = None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return repo_id

    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    return snapshot_download(**kwargs)


def resolve_pretrained_source(
    *,
    provider: str,
    model_ref: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    modelscope_fallback_to_hf: bool = True,
) -> ImportedModel:
    normalized = (provider or "hf").lower()

    if normalized in {"local", "path"}:
        if not (os.path.isdir(model_ref) or os.path.isfile(model_ref)):
            raise FileNotFoundError(f"未找到本地模型路径: {model_ref}")
        return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)

    if normalized in {"modelscope", "ms"}:
        if os.path.exists(model_ref):
            return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)
        try:
            local_dir = _snapshot_modelscope(model_ref, cache_dir=cache_dir, revision=revision)
            return ImportedModel(pretrained_source=local_dir, provider=normalized, model_ref=model_ref)
        except Exception:
            if modelscope_fallback_to_hf:
                return ImportedModel(pretrained_source=model_ref, provider="hf", model_ref=model_ref)
            raise

    if normalized in {"huggingface", "hf"}:
        local_dir_or_id = _snapshot_huggingface(model_ref, cache_dir=cache_dir, revision=revision)
        return ImportedModel(pretrained_source=local_dir_or_id, provider=normalized, model_ref=model_ref)

    return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)

