
import logging
import torch
from langchain_core.embeddings import Embeddings

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)
from app.runtime.llm.component_loader import (
    load_sentence_transformers_embedder,
    load_transformers_model,
    load_transformers_tokenizer,
    resolve_pretrained_source_for_spec,
    try_load_transformers_processor,
)
from app.runtime.llm.model_manager import (
    build_model_spec,
    get_best_device,
)


class ModelEmbeddings(Embeddings):
    """
    基于本地模型的 Embeddings 实现。
    支持 Transformers 和 SentenceTransformers 两种后端。
    负责将文本转换为向量表示。
    """
    def __init__(
        self,
        *,
        config: dict | None = None,
        model_name: str | None = None,
    ):
        cfg = config or settings.model_dump()
        emb_cfg = cfg.get("embeddings") or {}
        configured_model = emb_cfg.get("model_name") or cfg.get("local_models", {}).get("embedding_model")
        pooling = emb_cfg.get("pooling") or "auto"
        normalize = emb_cfg.get("normalize")
        max_length = emb_cfg.get("max_length")
        backend = emb_cfg.get("backend") or "transformers"
        batch_size = emb_cfg.get("batch_size")
        query_prefix = emb_cfg.get("query_prefix")
        doc_prefix = emb_cfg.get("doc_prefix")
        device = emb_cfg.get("device") or "auto"
        self._spec = build_model_spec(
            config=cfg,
            component_key="embeddings",
            env_var=emb_cfg.get("env_var") or "MODEL_PATH_EMBEDDING",
            config_path=("embeddings", "model_name"),
            explicit=model_name or configured_model,
            default=configured_model or "Qwen/Qwen3-Embedding-0.6B",
        )
        self.model_name = self._spec.model_ref
        if not self.model_name:
            raise ValueError("embeddings.model_name 未配置，且未传入 model_name")

        self._backend = str(backend)
        self._batch_size = 32 if batch_size is None else int(batch_size)
        self._query_prefix = "" if query_prefix is None else str(query_prefix)
        self._doc_prefix = "" if doc_prefix is None else str(doc_prefix)
        self._pooling = str(pooling)
        self._normalize = True if normalize is None else bool(normalize)
        self._max_length = 512 if max_length is None else int(max_length)

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._st_model = None
        self._loaded_source = None
        self._device = get_best_device() if str(device).lower() in {"auto", ""} else str(device)

    def _load_model(self):
        """懒加载模型：仅在首次使用时加载"""
        if self._backend == "sentence_transformers":
            if self._st_model is not None:
                return
            self._loaded_source = resolve_pretrained_source_for_spec(self._spec)
            logger.info(f"Loading embedding model: {self.model_name} (device: {self._device}, backend: sentence_transformers)")
            try:
                self._st_model = load_sentence_transformers_embedder(
                    self._loaded_source, device=self._device, max_length=self._max_length,
                    model_name=self.model_name
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
            return

        if self._model is None:
            self._loaded_source = resolve_pretrained_source_for_spec(self._spec)
            logger.info(f"Loading embedding model: {self.model_name} (device: {self._device})")
            try:
                self._model = load_transformers_model(
                    self._loaded_source,
                    trust_remote_code=self._spec.trust_remote_code,
                    device=self._device,
                    model_name=self.model_name,
                )
                self._processor = try_load_transformers_processor(
                    self._loaded_source, trust_remote_code=self._spec.trust_remote_code
                )
                if self._processor is None:
                    self._tokenizer = load_transformers_tokenizer(
                        self._loaded_source, trust_remote_code=self._spec.trust_remote_code
                    )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量计算文档列表的 embeddings。
        会自动添加 doc_prefix。
        """
        self._load_model()
        if not texts:
            return []
        prefixed = [self._doc_prefix + t for t in texts]
        if self._backend == "sentence_transformers":
            embeddings = self._st_model.encode(
                prefixed,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            return embeddings.detach().cpu().tolist()
        return self._embed_batch(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """
        计算单个查询的 embedding。
        会自动添加 query_prefix。
        """
        self._load_model()
        prefixed = self._query_prefix + text
        if self._backend == "sentence_transformers":
            embedding = self._st_model.encode(
                [prefixed],
                batch_size=1,
                normalize_embeddings=self._normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )[0]
            return embedding.detach().cpu().tolist()
        return self._embed_batch([prefixed])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        使用 Transformers 后端进行批量向量化。
        支持自定义 pooling 策略 (cls, mean, last_token)。
        """
        try:
            pooling = self._pooling
            if pooling == "auto":
                pooling = "mean"

            results: list[list[float]] = []
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                if self._processor is not None:
                    inputs = self._processor(
                        text=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                    )
                else:
                    inputs = self._tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                    )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.inference_mode():
                    if self._device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self._model(**inputs)
                    else:
                        outputs = self._model(**inputs)

                    if hasattr(outputs, "text_embeds"):
                        embedding_batch = outputs.text_embeds
                    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        embedding_batch = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs.get("attention_mask")
                        if pooling == "last_token":
                            if attention_mask is not None:
                                last_indices = attention_mask.sum(dim=1) - 1
                                batch_idx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
                                embedding_batch = token_embeddings[batch_idx, last_indices, :]
                            else:
                                embedding_batch = token_embeddings[:, -1, :]
                        elif pooling == "cls":
                            embedding_batch = token_embeddings[:, 0, :]
                        elif pooling == "mean":
                            if attention_mask is None:
                                embedding_batch = token_embeddings.mean(dim=1)
                            else:
                                mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
                                summed = (token_embeddings * mask).sum(dim=1)
                                denom = mask.sum(dim=1).clamp(min=1e-9)
                                embedding_batch = summed / denom
                        else:
                            raise ValueError(f"不支持的 embeddings.pooling: {pooling}")
                    else:
                        raise ValueError("模型输出不包含 text_embeds/pooler_output/last_hidden_state，无法生成 embedding")

                    embedding_batch = embedding_batch.float()
                    if self._normalize:
                        embedding_batch = torch.nn.functional.normalize(embedding_batch, p=2, dim=1)
                    results.extend(embedding_batch.detach().cpu().tolist())

            return results
        except Exception as e:
            preview = texts[0][:20] if texts else ""
            logger.error(f"Failed to embed text '{preview}...': {e}")
            raise e


_model_embeddings_instance = None


def get_embeddings() -> ModelEmbeddings:
    """
    获取 ModelEmbeddings 单例实例。
    避免重复加载模型到内存。

    Returns:
        ModelEmbeddings: Embeddings 模型单例
    """
    global _model_embeddings_instance
    if _model_embeddings_instance is None:
        _model_embeddings_instance = ModelEmbeddings()
    return _model_embeddings_instance


HFEmbeddings = ModelEmbeddings
Qwen3VLEmbeddings = ModelEmbeddings
