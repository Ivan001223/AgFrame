# RAG 架构设计 (RAG Architecture)

## 架构概览 (Overview)

AgFrame 默认的 RAG (检索增强生成) 链路被设计为极致的轻量与透明：

```text
Dense Search (密集检索) + BM25 (稀疏检索)
  -> RRF Fusion (倒数排序融合)
  -> Candidate Pruning (轻量级候选裁剪)
  -> Parent Restore (父文档还原)
  -> Prompt Assembly (组装 Prompt)
```

我们的目标是在不引入沉重的大模型重排阶段 (Heavyweight Document Reranking) 的前提下，最大化透明度、召回率和可维护性。

## 设计原则 (Design Principles)

- 在主干路径上仅使用两种召回通道：密集向量检索与 BM25 稀疏检索。
- 使用 RRF (Reciprocal Rank Fusion) 进行排序融合，取代基于模型的重排。
- 保持子分块 (Chunk) 检索的细粒度，但在生成回答前还原完整的父上下文 (Parent context)。
- 使用轻量级裁剪策略 (Lightweight pruning) 减少 Prompt 中的噪声。

## 核心代码路径 (Main Code Paths)

- 摄入与分块 (Ingestion and chunking): `app/skills/rag/rag_engine.py`
- 混合检索与 RRF: `app/skills/rag/hybrid_retriever_service.py`
- 向量与稀疏检索适配器: `app/memory/vector_stores/pgvector_vectorstore.py`
- 上下文裁剪 (Pruning): `app/runtime/prompts/context_pruner.py`
- 本地轻量级打分器: `app/infrastructure/utils/lightweight_ranker.py`

## 当前默认配置 (Current Defaults)

### 分块策略 (Chunking)

当前的文档摄入默认参数：

- 父分块 (Parent chunk): `6000` 字符，`400` 重叠 (overlap)
- 子分块 (Child chunk): `1400` 字符，`120` 重叠 (overlap)

原因：
- 子分块能够提升召回的精确度。
- 父分块能为 LLM 生成提供更完整的上下文背景。

### 检索策略 (Retrieval)

`settings.rag.retrieval` 中的当前默认值：

- `dense_k=20`
- `sparse_k=20`
- `candidate_k=20`
- `final_k=3`
- `rrf_k=60`

### 裁剪策略 (Pruning)

`settings.prompt.context_pruning` 中的当前默认值：

- `method="auto"`
- `auto_reranker_min_lines=40`
- `auto_reranker_min_chars=2500`
- `neighbor_window=1`
- `reranker_window_radius=1`
- `max_lines_per_item=24`
- `score_threshold=0.18`

注意：
- 依然接受名为 `reranker` 的配置项以保持兼容。
- 但其背后的实现现已改为基于轻量级本地算法的评分，而非大模型推理。

## 调优顺序 (Tuning Order)

建议按以下顺序进行效果调优：

1. `dense_k` 与 `sparse_k`
2. `candidate_k`
3. 子分块大小与重叠度
4. `final_k`
5. 裁剪阈值 (Pruning thresholds)

原因：
- 召回阶段的遗漏无法在后续的裁剪或重排中被弥补。
- 大部分命中率 (Hit-rate) 问题首先源于分块策略和候选池深度。

## 实践起点建议 (Practical Starting Points)

**通用知识库：**
- 子分块：`1000-1600` 字符
- 重叠度：`80-180` 字符
- `dense_k=20-40`
- `sparse_k=20-40`
- `candidate_k=20-30`
- `final_k=3-6`

**代码或 API 文档：**
- 使用略小的子分块。
- 重叠度保持在 10% 左右。
- 如果强依赖专业术语，优先增加 `sparse_k`。

**长篇政策或重度依赖 OCR 的文档：**
- 保持父分块相对较大。
- 避免过大的子分块，以免稀释 BM25 的精确术语信号。
- 在增加检索深度前，优先清洗 OCR 输出的乱码。

## 遗留兼容项 (Legacy-Compatible Items)

以下配置保留仅为了向后兼容，不再属于推荐的默认主干路径：

- `reranker.*`
- `local_models.rerank_model`
- `context_pruning.method="reranker"` 的命名方式

## 推荐后续规划 (Recommended Next Steps)

- 基于真实业务数据构建评测 Benchmark。
- 逐步在 UI 和文档中将 `reranker` 的概念重命名为 `lightweight_ranker`。
- 为代码、日志和表格引入感知类型 (Type-aware) 的分块与裁剪策略。
