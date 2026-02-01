from __future__ import annotations

from typing import Any, Optional

from transformers import AutoModel, AutoProcessor, AutoTokenizer

from app.runtime.llm.model_importer import resolve_pretrained_source
from app.runtime.llm.model_manager import torch_dtype_for_device


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
) -> Any:
    """
    加载 Transformers 模型。
    支持自动模型 (AutoModel) 和序列分类模型 (AutoModelForSequenceClassification)。
    """
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
) -> Any:
    """加载 SentenceTransformer 嵌入模型"""
    from sentence_transformers import SentenceTransformer

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
) -> Any:
    """加载 SentenceTransformer CrossEncoder 模型"""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(pretrained_source, device=device, max_length=max_length)

