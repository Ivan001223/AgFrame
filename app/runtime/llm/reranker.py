import importlib
import logging
from typing import Any, cast

import httpx

from app.infrastructure.config.settings import settings
from app.runtime.llm.component_loader import (
    load_sentence_transformers_cross_encoder,
    load_transformers_model,
    load_transformers_tokenizer,
    resolve_pretrained_source_for_spec,
    try_load_transformers_processor,
)
from app.runtime.llm.model_manager import build_model_spec, get_best_device

logger = logging.getLogger(__name__)


class ModelReranker:
    """Reranker adapter supporting local, remote, and disabled modes."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        model_name: str | None = None,
    ):
        cfg = config or settings.model_dump()
        rr_cfg = cfg.get("reranker") or {}
        configured_model = rr_cfg.get("model_name") or cfg.get("local_models", {}).get("rerank_model")
        backend = rr_cfg.get("backend") or "transformers"
        batch_size = rr_cfg.get("batch_size")
        max_length = rr_cfg.get("max_length")
        query_prefix = rr_cfg.get("query_prefix")
        doc_prefix = rr_cfg.get("doc_prefix")
        window_size = rr_cfg.get("window_size")
        stride = rr_cfg.get("stride")
        device = rr_cfg.get("device") or "auto"
        transformers_model_type = rr_cfg.get("transformers_model_type") or "auto"
        base_url = str(rr_cfg.get("base_url") or "").rstrip("/")
        api_key = str(rr_cfg.get("api_key") or "")
        timeout_seconds = rr_cfg.get("timeout_seconds")

        self._spec = build_model_spec(
            config=cfg,
            component_key="reranker",
            env_var=rr_cfg.get("env_var") or "MODEL_PATH_RERANKER",
            config_path=("reranker", "model_name"),
            explicit=model_name or configured_model,
            default=configured_model or "",
        )
        self.model_name = self._spec.model_ref or ""
        self._disabled = not bool(self.model_name)
        self._backend = str(backend)
        self._batch_size = 16 if batch_size is None else int(batch_size)
        self._max_length = 512 if max_length is None else int(max_length)
        self._query_prefix = "" if query_prefix is None else str(query_prefix)
        self._doc_prefix = "" if doc_prefix is None else str(doc_prefix)
        self._window_size = None if window_size is None else int(window_size)
        self._stride = None if stride is None else int(stride)
        self._transformers_model_type = str(transformers_model_type)
        self._provider = str(rr_cfg.get("provider") or cfg.get("model_manager", {}).get("provider") or "hf")
        self._remote_base_url = base_url
        self._remote_api_key = api_key
        self._remote_timeout_seconds = 30 if timeout_seconds is None else int(timeout_seconds)
        self._use_remote_api = bool(self.model_name) and (self._provider == "vllm" or bool(self._remote_base_url))

        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._cross_encoder: Any | None = None
        self._loaded_source: str | None = None

        if self._disabled or self._use_remote_api:
            self._device = "remote" if self._use_remote_api else "cpu"
        else:
            self._device = get_best_device() if str(device).lower() in {"auto", ""} else str(device)

    @staticmethod
    def _torch() -> Any:
        try:
            return importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing runtime dependency 'torch'. Run `uv sync` first.") from exc

    def _load_model(self) -> None:
        if self._disabled or self._use_remote_api:
            return

        loaded_source = resolve_pretrained_source_for_spec(self._spec)
        self._loaded_source = loaded_source

        if self._backend == "sentence_transformers":
            if self._cross_encoder is not None:
                return
            logger.info(
                "Loading reranker model: %s (device=%s, backend=sentence_transformers)",
                self.model_name,
                self._device,
            )
            self._cross_encoder = load_sentence_transformers_cross_encoder(
                loaded_source,
                device=self._device,
                max_length=self._max_length,
                model_name=self.model_name,
            )
            return

        if self._model is not None:
            return

        logger.info("Loading reranker model: %s (device=%s)", self.model_name, self._device)
        self._model = load_transformers_model(
            loaded_source,
            revision=self._spec.revision,
            trust_remote_code=self._spec.trust_remote_code,
            device=self._device,
            model_type=self._transformers_model_type,
            model_name=self.model_name,
        )
        self._processor = try_load_transformers_processor(
            loaded_source,
            revision=self._spec.revision,
            trust_remote_code=self._spec.trust_remote_code,
        )
        self._tokenizer = load_transformers_tokenizer(
            loaded_source,
            revision=self._spec.revision,
            trust_remote_code=self._spec.trust_remote_code,
        )

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[tuple[str, float, int]]:
        if not documents:
            return []
        if self._disabled:
            return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]
        if self._use_remote_api:
            return self._rerank_remote(query, documents, top_k=top_k)

        self._load_model()
        query_text = self._query_prefix + query
        docs = [self._doc_prefix + doc for doc in documents]

        if self._backend == "sentence_transformers":
            cross_encoder = self._cross_encoder
            if cross_encoder is None:
                raise RuntimeError("Reranker model was not loaded")
            pairs = [(query_text, doc) for doc in docs]
            try:
                torch = self._torch()
                batch_scores = cross_encoder.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.detach().cpu().tolist()
                scores = [(documents[i], float(batch_scores[i]), i) for i in range(len(documents))]
                scores.sort(key=lambda item: item[1], reverse=True)
                return scores[:top_k]
            except Exception as exc:
                logger.warning("Reranking failed: %s", exc)
                return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]

        try:
            model = self._model
            tokenizer = self._tokenizer
            if model is None:
                raise RuntimeError("Reranker model was not loaded")

            if hasattr(model, "compute_score"):
                compute_scores: list[float] = []
                for start in range(0, len(docs), self._batch_size):
                    pairs = [(query_text, doc) for doc in docs[start : start + self._batch_size]]
                    torch = self._torch()
                    with torch.inference_mode():
                        batch_scores = model.compute_score(pairs)
                    if hasattr(batch_scores, "detach"):
                        batch_scores = batch_scores.detach().cpu().tolist()
                    compute_scores.extend(float(score) for score in batch_scores)
                ranked = [(documents[i], float(compute_scores[i]), i) for i in range(len(documents))]
                ranked.sort(key=lambda item: item[1], reverse=True)
                return ranked[:top_k]

            if hasattr(model, "predict"):
                predict_scores: list[float] = []
                for start in range(0, len(docs), self._batch_size):
                    pairs = [(query_text, doc) for doc in docs[start : start + self._batch_size]]
                    batch_scores = model.predict(pairs)
                    if hasattr(batch_scores, "detach"):
                        batch_scores = batch_scores.detach().cpu().tolist()
                    predict_scores.extend(float(score) for score in batch_scores)
                ranked = [(documents[i], float(predict_scores[i]), i) for i in range(len(documents))]
                ranked.sort(key=lambda item: item[1], reverse=True)
                return ranked[:top_k]

            if tokenizer is None or not callable(model):
                return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]

            transformer_scores = self._score_pairs_transformers(query_text, docs)
            ranked = [(documents[i], float(transformer_scores[i]), i) for i in range(len(documents))]
            ranked.sort(key=lambda item: item[1], reverse=True)
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]

    def _score_pairs_transformers(self, query: str, docs: list[str]) -> list[float]:
        if self._window_size is not None and self._window_size > 0:
            stride = self._stride or self._window_size
            return [self._score_single_with_windows(query, doc, stride=stride) for doc in docs]
        return self._score_pairs_transformers_no_window(query, docs)

    def _score_pairs_transformers_no_window(self, query: str, docs: list[str]) -> list[float]:
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Reranker model or tokenizer was not loaded")

        scores: list[float] = []
        torch = self._torch()
        for start in range(0, len(docs), self._batch_size):
            query_batch = [query] * len(docs[start : start + self._batch_size])
            doc_batch = docs[start : start + self._batch_size]
            inputs = tokenizer(
                query_batch,
                doc_batch,
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

                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise ValueError("Transformers reranker output does not contain logits")
                if logits.dim() == 2 and logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.dim() == 2 and logits.size(-1) >= 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(logits.size(0), -1)[:, -1]
                scores.extend(cast(list[float], batch_scores.detach().float().cpu().tolist()))

        return [float(score) for score in scores]

    def _score_single_with_windows(self, query: str, doc: str, *, stride: int) -> float:
        tokenizer = self._tokenizer
        window_size = self._window_size
        if tokenizer is None or window_size is None:
            raise RuntimeError("Reranker tokenizer or window size was not configured")

        tokens = tokenizer(doc, add_special_tokens=False, return_tensors=None)
        input_ids = tokens.get("input_ids") if isinstance(tokens, dict) else None
        if not input_ids:
            return 0.0
        token_ids = list(input_ids)
        best_score: float | None = None
        for start in range(0, len(token_ids), stride):
            window_ids = token_ids[start : start + window_size]
            if not window_ids:
                break
            window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
            score = self._score_pairs_transformers_no_window(query, [window_text])[0]
            best_score = score if best_score is None else max(best_score, score)
            if start + window_size >= len(token_ids):
                break
        return 0.0 if best_score is None else float(best_score)

    def _rerank_remote(self, query: str, documents: list[str], *, top_k: int) -> list[tuple[str, float, int]]:
        if not self._remote_base_url:
            raise ValueError("reranker.base_url is required for remote rerank")

        headers = {"Content-Type": "application/json"}
        if self._remote_api_key:
            headers["Authorization"] = f"Bearer {self._remote_api_key}"

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        response = None
        errors: list[str] = []
        for path in ("/v1/rerank", "/rerank"):
            try:
                response = httpx.post(
                    f"{self._remote_base_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=self._remote_timeout_seconds,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                response = None

        if response is None:
            logger.warning("Remote rerank failed: %s", "; ".join(errors))
            return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]

        body = response.json()
        results = body.get("results") or body.get("data") or []
        ranked: list[tuple[str, float, int]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", item.get("document_index", -1)) or -1)
            if index < 0 or index >= len(documents):
                continue
            score = float(item.get("relevance_score", item.get("score", 0.0)) or 0.0)
            ranked.append((documents[index], score, index))

        if not ranked:
            return [(doc, 0.0, index) for index, doc in enumerate(documents)][:top_k]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


_model_reranker_instance: ModelReranker | None = None


def get_reranker() -> ModelReranker:
    global _model_reranker_instance
    if _model_reranker_instance is None:
        _model_reranker_instance = ModelReranker()
    return _model_reranker_instance


HFReranker = ModelReranker
Qwen3VLReranker = ModelReranker
