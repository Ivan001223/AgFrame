from __future__ import annotations

import importlib
import logging
from typing import Any

from app.runtime.llm.model_importer import resolve_pretrained_source
from app.runtime.llm.model_manager import torch_dtype_for_device

logger = logging.getLogger(__name__)


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少可选依赖 'transformers'。如需本地 embeddings/reranker/Qwen 模型，请执行 "
            "`uv sync --group local-inference`。"
        ) from exc


def _require_tqdm() -> Any:
    return importlib.import_module("tqdm.auto").tqdm


def _require_sentence_transformers() -> Any:
    try:
        return importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少可选依赖 'sentence-transformers'。如需本地 embeddings/reranker 模型，请执行 "
            "`uv sync --group local-inference`。"
        ) from exc


def _download_with_progress(
    pretrained_source: str,
    cache_dir: str | None = None,
    *,
    revision: str | None = None,
    desc: str = "下载模型",
):
    """使用进度条下载 HuggingFace 模型"""
    tqdm = _require_tqdm()
    try:
        from huggingface_hub import HfApi, snapshot_download

        tqdm.write(f"📦 正在下载 {desc}...")

        api = HfApi()

        repo_info = api.repo_info(pretrained_source, repo_type="model", revision=revision)
        siblings = getattr(repo_info, 'siblings', [])
        if not siblings:
            siblings = getattr(repo_info, 'files', [])

        total_files = len(siblings)
        if total_files == 0:
            snapshot_download(pretrained_source, cache_dir=cache_dir, revision=revision)
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
                        revision=revision,
                        resume_download=True,
                    )
                except Exception as e:
                    logger.debug(f"Failed to download {filename}: {e}")
                pbar.update(1)
    except Exception as e:
        logger.warning(f"Model download failed: {e}")


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
    revision: str | None,
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

    transformers = _require_transformers()
    tqdm = _require_tqdm()

    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    if model_type == "sequence_classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_source,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype_for_device(device),
        )
    else:
        model = transformers.AutoModel.from_pretrained(
            pretrained_source,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype_for_device(device),
        )
    model = model.to(device)
    model.eval()
    return model


def try_load_transformers_processor(
    pretrained_source: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
) -> Any | None:
    """尝试加载 Transformers Processor，失败返回 None"""
    transformers = _require_transformers()
    try:
        return transformers.AutoProcessor.from_pretrained(
            pretrained_source,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logger.debug(f"Failed to load processor for {pretrained_source}: {e}")
        return None


def load_transformers_tokenizer(
    pretrained_source: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
) -> Any:
    """加载 Transformers Tokenizer"""
    transformers = _require_transformers()
    return transformers.AutoTokenizer.from_pretrained(
        pretrained_source,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def load_sentence_transformers_embedder(
    pretrained_source: str,
    *,
    device: str,
    max_length: int | None = None,
    model_name: str = "嵌入模型",
) -> Any:
    """加载 SentenceTransformer 嵌入模型，带下载进度条"""
    import tempfile

    tqdm = _require_tqdm()
    sentence_transformers = _require_sentence_transformers()
    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    model = sentence_transformers.SentenceTransformer(pretrained_source, device=device)
    if max_length is not None:
        try:
            model.max_seq_length = int(max_length)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to set max_seq_length to {max_length}: {e}")
    return model


def load_sentence_transformers_cross_encoder(
    pretrained_source: str,
    *,
    device: str,
    max_length: int | None = None,
    model_name: str = "重排序模型",
) -> Any:
    """加载 SentenceTransformer CrossEncoder 模型，带下载进度条"""
    import tempfile

    tqdm = _require_tqdm()
    sentence_transformers = _require_sentence_transformers()
    cache_dir = tempfile.gettempdir()
    tqdm.write(f"📦 正在下载 {model_name}...")

    _download_with_progress(pretrained_source, cache_dir=cache_dir, desc=f"下载 {model_name}")

    return sentence_transformers.CrossEncoder(
        pretrained_source, device=device, max_length=max_length
    )
