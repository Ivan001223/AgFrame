from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class ImportedModel:
    """已导入的模型信息"""
    pretrained_source: str  # 模型在本地的实际路径
    provider: str           # 来源提供方 (local, modelscope, hf)
    model_ref: str          # 原始引用字符串


def _snapshot_modelscope(model_id: str, *, cache_dir: Optional[str] = None, revision: Optional[str] = None) -> str:
    """使用 ModelScope 下载模型快照"""
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
    """使用 HuggingFace Hub 下载模型快照"""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        # 如果 HF 库不可用，直接返回 ID，交给 Transformers 自动处理
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
    """
    解析模型来源并确保模型已下载到本地。
    
    支持的 Provider:
    - local / path: 本地文件路径
    - modelscope / ms: 阿里云 ModelScope
    - huggingface / hf: HuggingFace Hub
    """
    normalized = (provider or "hf").lower()

    # 1. 本地路径模式
    if normalized in {"local", "path"}:
        if not (os.path.isdir(model_ref) or os.path.isfile(model_ref)):
            raise FileNotFoundError(f"未找到本地模型路径: {model_ref}")
        return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)

    # 2. ModelScope 模式
    if normalized in {"modelscope", "ms"}:
        if os.path.exists(model_ref):
            return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)
        try:
            local_dir = _snapshot_modelscope(model_ref, cache_dir=cache_dir, revision=revision)
            return ImportedModel(pretrained_source=local_dir, provider=normalized, model_ref=model_ref)
        except Exception:
            if modelscope_fallback_to_hf:
                # 降级到 HF
                return ImportedModel(pretrained_source=model_ref, provider="hf", model_ref=model_ref)
            raise

    # 3. HuggingFace 模式 (默认)
    if normalized in {"huggingface", "hf"}:
        local_dir_or_id = _snapshot_huggingface(model_ref, cache_dir=cache_dir, revision=revision)
        return ImportedModel(pretrained_source=local_dir_or_id, provider=normalized, model_ref=model_ref)

    return ImportedModel(pretrained_source=model_ref, provider=normalized, model_ref=model_ref)

