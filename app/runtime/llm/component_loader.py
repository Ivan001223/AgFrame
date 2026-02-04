from __future__ import annotations

import os
from typing import Any, Optional

from transformers import AutoModel, AutoProcessor, AutoTokenizer

from app.runtime.llm.model_importer import resolve_pretrained_source
from app.runtime.llm.model_manager import torch_dtype_for_device


def _download_with_progress(pretrained_source: str, cache_dir: Optional[str] = None, desc: str = "下载模型"):
    """使用进度条下载 HuggingFace 模型"""
    try:
        from huggingface_hub import HfApi, snapshot_download
        from tqdm.auto import tqdm

        tqdm.write(f"📦 正在下载 {desc}...")

        api = HfApi()

        repo_info = api.repo_info(pretrained_source, repo_type="model")
        siblings = getattr(repo_info, 'siblings', [])
        if not siblings:
            siblings = getattr(repo_info, 'files', [])

        total_files = len(siblings)
        if total_files == 0:
            snapshot_download(pretrained_source, cache_dir=cache_dir)
            return

        with tqdm(total=total_files, desc=f"下载 {desc}", unit="文件") as pbar:
            for sibling in siblings:
                filename = sibling.rfilename if hasattr(sibling, 'rfilename') else sibling
                try:
                    api.hf_hub_download(
                        filename=filename,
                        repo_id=pretrained_source,
                        repo_type="model",
                        cache_dir=cache_dir,
                        resume_download=True,
                    )
                except Exception:
                    pass
                pbar.update(1)
    except Exception:
        pass


def resolve_pretrained_source_for_spec(spec: Any) -> str:
    """
    根据 ModelSpec 解析预训练模型源路径。
    自动处理 ModelScope/HuggingFace 的下载逻辑。
    """
    imported = resolve_pretrained_source(
        provider=spec.provider,
        model_ref=spec.model_ref,
        cache_dir=spec.cache_dir,
        revision=spec.revision,
        modelscope_fallback_to_hf=spec.modelscope_fallback_to_hf,
    )
    return imported.pretrained_source


def load_transformers_model(
    pretrained_source: str,
    *,
    trust_remote_code: bool,
    device: str,
    model_type: str = "auto",
    model_name: str = "模型",
) -> Any:
    """
    加载 Transformers 模型。
    支持自动模型 (AutoModel) 和序列分类模型 (AutoModelForSequenceClassification)。
    显示下载进度条。
    """
    import tempfile
    from tqdm.auto import tqdm

    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    if model_type == "sequence_classification":
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_source,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype_for_device(device),
        )
    else:
        model = AutoModel.from_pretrained(
            pretrained_source,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype_for_device(device),
        )
    model = model.to(device)
    model.eval()
    return model


def try_load_transformers_processor(pretrained_source: str, *, trust_remote_code: bool) -> Optional[Any]:
    """尝试加载 Transformers Processor，失败返回 None"""
    try:
        return AutoProcessor.from_pretrained(pretrained_source, trust_remote_code=trust_remote_code)
    except Exception:
        return None


def load_transformers_tokenizer(pretrained_source: str, *, trust_remote_code: bool) -> Any:
    """加载 Transformers Tokenizer"""
    return AutoTokenizer.from_pretrained(pretrained_source, trust_remote_code=trust_remote_code)


def load_sentence_transformers_embedder(
    pretrained_source: str,
    *,
    device: str,
    max_length: Optional[int] = None,
    model_name: str = "嵌入模型",
) -> Any:
    """加载 SentenceTransformer 嵌入模型，带下载进度条"""
    from sentence_transformers import SentenceTransformer
    import tempfile

    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    model = SentenceTransformer(pretrained_source, device=device)
    if max_length is not None:
        try:
            model.max_seq_length = int(max_length)
        except Exception:
            pass
    return model


def load_sentence_transformers_cross_encoder(
    pretrained_source: str,
    *,
    device: str,
    max_length: Optional[int] = None,
    model_name: str = "重排序模型",
) -> Any:
    """加载 SentenceTransformer CrossEncoder 模型，带下载进度条"""
    from sentence_transformers import CrossEncoder
    import tempfile

    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    return CrossEncoder(pretrained_source, device=device, max_length=max_length)

