
import torch

from app.infrastructure.config.settings import settings
from app.runtime.llm.component_loader import (
    load_sentence_transformers_cross_encoder,
    load_transformers_model,
    load_transformers_tokenizer,
    resolve_pretrained_source_for_spec,
    try_load_transformers_processor,
)
from app.runtime.llm.model_manager import (
    build_model_spec,
    get_best_device,
)


class ModelReranker:
    """
    基于本地模型的重排器 (Reranker) 实现。
    用于对初步召回的文档进行二次精排。
    支持 Transformers (CrossEncoder/SequenceClassification) 和 SentenceTransformers 后端。
    """
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
        self._spec = build_model_spec(
            config=cfg,
            component_key="reranker",
            env_var=rr_cfg.get("env_var") or "MODEL_PATH_RERANKER",
            config_path=("reranker", "model_name"),
            explicit=model_name or configured_model,
            default=configured_model or "",
        )
        self.model_name = self._spec.model_ref
        self._disabled = not bool(self.model_name)
        self._backend = str(backend)
        self._batch_size = 16 if batch_size is None else int(batch_size)
        self._max_length = 512 if max_length is None else int(max_length)
        self._query_prefix = "" if query_prefix is None else str(query_prefix)
        self._doc_prefix = "" if doc_prefix is None else str(doc_prefix)
        self._window_size = None if window_size is None else int(window_size)
        self._stride = None if stride is None else int(stride)
        self._transformers_model_type = str(transformers_model_type)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._cross_encoder = None
        self._loaded_source = None
        self._device = get_best_device() if str(device).lower() in {"auto", ""} else str(device)

    def _load_model(self):
        """懒加载模型：仅在首次使用时加载"""
        if self._disabled:
            return
        if self._backend == "sentence_transformers":
            if self._cross_encoder is not None:
                return
            self._loaded_source = resolve_pretrained_source_for_spec(self._spec)
            print(f"正在加载重排模型：{self.model_name}（设备：{self._device}，后端：sentence_transformers）...")
            try:
                self._cross_encoder = load_sentence_transformers_cross_encoder(
                    self._loaded_source, device=self._device, max_length=self._max_length,
                    model_name=self.model_name
                )
                print("重排模型加载完成。")
            except Exception as e:
                print(f"加载重排模型失败：{e}")
                raise e
            return

        if self._model is None:
            self._loaded_source = resolve_pretrained_source_for_spec(self._spec)
            print(f"正在加载重排模型：{self.model_name}（设备：{self._device}）...")
            try:
                self._model = load_transformers_model(
                    self._loaded_source,
                    trust_remote_code=self._spec.trust_remote_code,
                    device=self._device,
                    model_type=self._transformers_model_type,
                    model_name=self.model_name,
                )
                self._processor = try_load_transformers_processor(
                    self._loaded_source, trust_remote_code=self._spec.trust_remote_code
                )
                self._tokenizer = load_transformers_tokenizer(
                    self._loaded_source, trust_remote_code=self._spec.trust_remote_code
                )
                print("重排模型加载完成。")
            except Exception as e:
                print(f"加载重排模型失败：{e}")
                raise e

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[tuple[str, float, int]]:
        """
        基于查询对候选文档列表进行重排。
        
        Args:
            query: 查询字符串
            documents: 候选文档列表
            top_k: 返回前 K 个结果
            
        Returns:
            List[Tuple[str, float, int]]: 按分数降序排列的结果列表，每项为 (doc_content, score, original_index)
        """
        if not documents:
            return []
        if self._disabled:
            return [(doc, 0.0, i) for i, doc in enumerate(documents)][:top_k]
            
        self._load_model()

        q = self._query_prefix + query
        docs = [self._doc_prefix + d for d in documents]
        if self._backend == "sentence_transformers":
            pairs = [(q, d) for d in docs]
            try:
                batch_scores = self._cross_encoder.predict(
                    pairs, batch_size=self._batch_size, show_progress_bar=False
                )
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.detach().cpu().tolist()
                scores = [(documents[i], float(batch_scores[i]), i) for i in range(len(documents))]
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]
            except Exception as e:
                print(f"重排失败：{e}")
                return [(doc, 0.0, i) for i, doc in enumerate(documents)][:top_k]

        try:
            # 优先尝试 compute_score 接口
            if hasattr(self._model, "compute_score"):
                all_scores: list[float] = []
                for start in range(0, len(docs), self._batch_size):
                    pairs = [[q, d] for d in docs[start : start + self._batch_size]]
                    with torch.inference_mode():
                        batch_scores = self._model.compute_score(pairs)
                    if isinstance(batch_scores, torch.Tensor):
                        batch_scores = batch_scores.detach().cpu().tolist()
                    all_scores.extend([float(s) for s in batch_scores])
                scores = [(documents[i], float(all_scores[i]), i) for i in range(len(documents))]
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]

            # 其次尝试 predict 接口
            if hasattr(self._model, "predict"):
                all_scores: list[float] = []
                for start in range(0, len(docs), self._batch_size):
                    pairs = [[q, d] for d in docs[start : start + self._batch_size]]
                    batch_scores = self._model.predict(pairs)
                    if isinstance(batch_scores, torch.Tensor):
                        batch_scores = batch_scores.detach().cpu().tolist()
                    all_scores.extend([float(s) for s in batch_scores])
                scores = [(documents[i], float(all_scores[i]), i) for i in range(len(documents))]
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]

            if self._tokenizer is None or not hasattr(self._model, "__call__"):
                return [(doc, 0.0, i) for i, doc in enumerate(documents)][:top_k]

            # 默认使用 Transformers 序列分类逻辑
            all_scores = self._score_pairs_transformers(q, docs)
            scores = [(documents[i], float(all_scores[i]), i) for i in range(len(documents))]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception as e:
            print(f"重排失败：{e}")
            return [(doc, 0.0, i) for i, doc in enumerate(documents)][:top_k]

    def _score_pairs_transformers(self, query: str, docs: list[str]) -> list[float]:
        """使用 Transformers 模型对文本对打分（支持滑动窗口）"""
        if self._window_size is not None and self._window_size > 0:
            stride = self._stride or self._window_size
            return [self._score_single_with_windows(query, d, stride=stride) for d in docs]

        return self._score_pairs_transformers_no_window(query, docs)

    def _score_pairs_transformers_no_window(self, query: str, docs: list[str]) -> list[float]:
        """批量计算文本对分数（无滑动窗口）"""
        scores: list[float] = []
        for start in range(0, len(docs), self._batch_size):
            q_batch = [query] * len(docs[start : start + self._batch_size])
            d_batch = docs[start : start + self._batch_size]
            inputs = self._tokenizer(
                q_batch,
                d_batch,
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
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise ValueError("transformers reranker 输出不包含 logits")
                if logits.dim() == 2 and logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.dim() == 2 and logits.size(-1) >= 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(logits.size(0), -1)[:, -1]
                scores.extend(batch_scores.detach().float().cpu().tolist())
        return [float(s) for s in scores]

    def _score_single_with_windows(self, query: str, doc: str, *, stride: int) -> float:
        """对长文档使用滑动窗口计算最高分"""
        tokens = self._tokenizer(doc, add_special_tokens=False, return_tensors=None)
        input_ids = tokens.get("input_ids") if isinstance(tokens, dict) else None
        if not input_ids:
            return 0.0
        input_ids = input_ids[:]
        best = None
        for start in range(0, len(input_ids), stride):
            window_ids = input_ids[start : start + self._window_size]
            if not window_ids:
                break
            window_text = self._tokenizer.decode(window_ids, skip_special_tokens=True)
            score = self._score_pairs_transformers_no_window(query, [window_text])[0]
            best = score if best is None else max(best, score)
            if start + self._window_size >= len(input_ids):
                break
        return 0.0 if best is None else float(best)


HFReranker = ModelReranker
Qwen3VLReranker = ModelReranker
