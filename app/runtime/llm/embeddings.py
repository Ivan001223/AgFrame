import importlib
import logging
import math
from hashlib import sha256
from typing import Any, cast

import httpx
from langchain_core.embeddings import Embeddings

from app.infrastructure.config.settings import settings
from app.runtime.llm.component_loader import (
    load_sentence_transformers_embedder,
    load_transformers_model,
    load_transformers_tokenizer,
    resolve_pretrained_source_for_spec,
    try_load_transformers_processor,
)
from app.runtime.llm.model_manager import build_model_spec, get_best_device

logger = logging.getLogger(__name__)


class ModelEmbeddings(Embeddings):
    """Embedding adapter supporting local, remote, and dev-stub modes."""

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
        base_url = str(emb_cfg.get("base_url") or "").rstrip("/")
        api_key = str(emb_cfg.get("api_key") or "")
        timeout_seconds = emb_cfg.get("timeout_seconds")

        self._spec = build_model_spec(
            config=cfg,
            component_key="embeddings",
            env_var=emb_cfg.get("env_var") or "MODEL_PATH_EMBEDDING",
            config_path=("embeddings", "model_name"),
            explicit=model_name or configured_model,
            default=configured_model or "Qwen/Qwen3-Embedding-0.6B",
        )
        resolved_model = self._spec.model_ref
        if not resolved_model:
            raise ValueError("embeddings.model_name is required")
        self.model_name = resolved_model
        self._use_dev_stub = self.model_name in {"dev-stub", "dev_stub"}

        self._backend = str(backend)
        self._batch_size = 32 if batch_size is None else int(batch_size)
        self._query_prefix = "" if query_prefix is None else str(query_prefix)
        self._doc_prefix = "" if doc_prefix is None else str(doc_prefix)
        self._pooling = str(pooling)
        self._normalize = True if normalize is None else bool(normalize)
        self._max_length = 512 if max_length is None else int(max_length)
        self._provider = str(emb_cfg.get("provider") or cfg.get("model_manager", {}).get("provider") or "hf")
        self._remote_base_url = base_url
        self._remote_api_key = api_key
        self._remote_timeout_seconds = 30 if timeout_seconds is None else int(timeout_seconds)
        self._use_remote_api = self._provider == "vllm" or bool(self._remote_base_url)

        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._st_model: Any | None = None
        self._loaded_source: str | None = None

        if self._use_dev_stub:
            self._device = "stub"
        elif self._use_remote_api:
            self._device = "remote"
        else:
            self._device = get_best_device() if str(device).lower() in {"auto", ""} else str(device)

    @staticmethod
    def _torch() -> Any:
        try:
            return importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing runtime dependency 'torch'. Run `uv sync` first.") from exc

    def _load_model(self) -> None:
        if self._use_dev_stub or self._use_remote_api:
            return

        loaded_source = resolve_pretrained_source_for_spec(self._spec)
        self._loaded_source = loaded_source

        if self._backend == "sentence_transformers":
            if self._st_model is not None:
                return
            logger.info(
                "Loading embedding model: %s (device=%s, backend=sentence_transformers)",
                self.model_name,
                self._device,
            )
            self._st_model = load_sentence_transformers_embedder(
                loaded_source,
                device=self._device,
                max_length=self._max_length,
                model_name=self.model_name,
            )
            return

        if self._model is not None:
            return

        logger.info("Loading embedding model: %s (device=%s)", self.model_name, self._device)
        self._model = load_transformers_model(
            loaded_source,
            revision=self._spec.revision,
            trust_remote_code=self._spec.trust_remote_code,
            device=self._device,
            model_name=self.model_name,
        )
        self._processor = try_load_transformers_processor(
            loaded_source,
            revision=self._spec.revision,
            trust_remote_code=self._spec.trust_remote_code,
        )
        if self._processor is None:
            self._tokenizer = load_transformers_tokenizer(
                loaded_source,
                revision=self._spec.revision,
                trust_remote_code=self._spec.trust_remote_code,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        if not texts:
            return []
        prefixed = [self._doc_prefix + text for text in texts]
        if self._use_dev_stub:
            return self._embed_stub(prefixed)
        if self._use_remote_api:
            return self._embed_remote(prefixed)
        if self._backend == "sentence_transformers":
            st_model = self._st_model
            if st_model is None:
                raise RuntimeError("Embedding model was not loaded")
            embeddings = st_model.encode(
                prefixed,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            return cast(list[list[float]], embeddings.detach().cpu().tolist())
        return self._embed_batch(prefixed)

    def embed_query(self, text: str) -> list[float]:
        self._load_model()
        prefixed = self._query_prefix + text
        if self._use_dev_stub:
            return self._embed_stub([prefixed])[0]
        if self._use_remote_api:
            return self._embed_remote([prefixed])[0]
        if self._backend == "sentence_transformers":
            st_model = self._st_model
            if st_model is None:
                raise RuntimeError("Embedding model was not loaded")
            embedding = st_model.encode(
                [prefixed],
                batch_size=1,
                normalize_embeddings=self._normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )[0]
            return cast(list[float], embedding.detach().cpu().tolist())
        return self._embed_batch([prefixed])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._model
        processor = self._processor
        tokenizer = self._tokenizer
        if model is None:
            raise RuntimeError("Embedding model was not loaded")
        if processor is None and tokenizer is None:
            raise RuntimeError("Embedding tokenizer was not loaded")

        pooling = "mean" if self._pooling == "auto" else self._pooling
        results: list[list[float]] = []
        torch = self._torch()

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            if processor is not None:
                inputs = processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                )
            else:
                inputs = cast(Any, tokenizer)(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}

            with torch.inference_mode():
                if self._device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

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
                            batch_index = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
                            embedding_batch = token_embeddings[batch_index, last_indices, :]
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
                        raise ValueError(f"Unsupported embeddings.pooling: {pooling}")
                else:
                    raise ValueError("Model output does not contain an embedding tensor")

                embedding_batch = embedding_batch.float()
                if self._normalize:
                    embedding_batch = torch.nn.functional.normalize(embedding_batch, p=2, dim=1)
                results.extend(cast(list[list[float]], embedding_batch.detach().cpu().tolist()))

        return results

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        if not self._remote_base_url:
            raise ValueError("embeddings.base_url is required for remote embedding")

        headers = {"Content-Type": "application/json"}
        if self._remote_api_key:
            headers["Authorization"] = f"Bearer {self._remote_api_key}"

        response = httpx.post(
            f"{self._remote_base_url}/v1/embeddings",
            headers=headers,
            json={"model": self.model_name, "input": texts},
            timeout=self._remote_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            raise ValueError("Remote embeddings response is missing data")

        ordered = sorted(
            [item for item in data if isinstance(item, dict)],
            key=lambda item: int(item.get("index", 0)),
        )
        vectors = [item.get("embedding") for item in ordered]
        if any(vector is None for vector in vectors):
            raise ValueError("Remote embeddings response is missing embedding values")
        return cast(list[list[float]], vectors)

    def _embed_stub(self, texts: list[str]) -> list[list[float]]:
        dim = int(settings.feature_flags.pgvector_dimension or 1024)
        return [self._stub_vector(text, dim=dim) for text in texts]

    def _stub_vector(self, text: str, *, dim: int) -> list[float]:
        values: list[float] = []
        seed = sha256(text.encode("utf-8")).digest()
        counter = 0

        while len(values) < dim:
            digest = sha256(seed + counter.to_bytes(4, "big")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index : index + 4]
                if len(chunk) < 4:
                    continue
                scalar = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                values.append((scalar * 2.0) - 1.0)
                if len(values) >= dim:
                    break
            counter += 1

        if self._normalize:
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            values = [value / norm for value in values]
        return values[:dim]


_model_embeddings_instance: ModelEmbeddings | None = None


def get_embeddings() -> ModelEmbeddings:
    global _model_embeddings_instance
    if _model_embeddings_instance is None:
        _model_embeddings_instance = ModelEmbeddings()
    return _model_embeddings_instance
