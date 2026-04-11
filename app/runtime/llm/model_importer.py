from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib import import_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportedModel:
    """Resolved model import information."""

    pretrained_source: str
    provider: str
    model_ref: str


def require_pinned_revision(provider: str, model_ref: str, revision: str | None) -> None:
    normalized_provider = (provider or "").lower()
    if normalized_provider not in {"huggingface", "hf", "modelscope", "ms"}:
        return
    if os.path.exists(model_ref):
        return
    if revision and str(revision).strip():
        return
    raise ValueError(
        f"Remote model '{model_ref}' requires an explicit revision pin for provider '{provider}'."
    )


def _snapshot_modelscope(model_id: str, *, cache_dir: str | None = None, revision: str | None = None) -> str:
    require_pinned_revision("modelscope", model_id, revision)
    try:
        snapshot_download = import_module("modelscope.hub.snapshot_download").snapshot_download
    except Exception as exc:
        raise RuntimeError("modelscope is not installed or unavailable") from exc

    return str(
        snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
            revision=revision,
        )
    )


def _snapshot_huggingface(repo_id: str, *, cache_dir: str | None = None, revision: str | None = None) -> str:
    require_pinned_revision("hf", repo_id, revision)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        logger.debug("HF hub not available: %s", exc)
        return repo_id

    return str(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
        )
    )


def resolve_pretrained_source(
    *,
    provider: str,
    model_ref: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    modelscope_fallback_to_hf: bool = True,
) -> ImportedModel:
    """Resolve a model reference into a local pretrained source path or repo id."""

    normalized = (provider or "hf").lower()

    if normalized in {"local", "path"}:
        if not (os.path.isdir(model_ref) or os.path.isfile(model_ref)):
            raise FileNotFoundError(f"Local model path not found: {model_ref}")
        return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)

    if normalized in {"modelscope", "ms"}:
        if os.path.exists(model_ref):
            return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)
        try:
            local_dir = _snapshot_modelscope(model_ref, cache_dir=cache_dir, revision=revision)
            return ImportedModel(pretrained_source=local_dir, provider=normalized, model_ref=model_ref)
        except Exception as exc:
            if modelscope_fallback_to_hf:
                logger.warning("ModelScope download failed, falling back to HF: %s", exc)
                return ImportedModel(pretrained_source=model_ref, provider="hf", model_ref=model_ref)
            raise

    if normalized in {"huggingface", "hf"}:
        local_dir_or_id = _snapshot_huggingface(model_ref, cache_dir=cache_dir, revision=revision)
        return ImportedModel(pretrained_source=local_dir_or_id, provider=normalized, model_ref=model_ref)

    return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)
