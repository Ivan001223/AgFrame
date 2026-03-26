# RAG 迁移指南 (RAG Migration Guide)

## 适用对象 (Who This Is For)

如果你正从一个旧版、较重的 RAG 架构迁移到 AgFrame 当前推荐的轻量级默认链路，请参考本指南。

典型情况：
- 你之前使用的是：密集检索 (Dense) + 稀疏检索 (Sparse) + 模型重排 (Model reranking)
- 你的配置文件中仍然存在 `reranker.*` 相关的项
- 你之前的图 (Graph) 依赖了一个独立的 `rerank_docs` 节点

## 目标状态 (Target State)

推荐的最终状态：

- 文档检索：`Dense + BM25 + RRF`
- 裁剪阶段：轻量级本地打分 / 启发式裁剪 (Lightweight local ranking / heuristic pruning)
- 父文档还原 (Parent restore)：开启
- 大模型重排 (Model reranker)：默认关闭

## 配置变更 (Configuration Changes)

**保留并可调优的项：**
- `embeddings.*`
- `rag.retrieval.*`
- `prompt.context_pruning.*`

**通常应置空（除非你需要遗留兼容）：**
- `reranker.model_name`
- `local_models.rerank_model`

**推荐的节点列表 (Node list)：**
```json
{
  "nodes": {
    "enabled": [
      "router",
      "retrieve_docs",
      "retrieve_memories",
      "assemble",
      "generate"
    ]
  }
}
```

如果你旧的配置中仍包含 `rerank_docs`，请将其移除。

## 行为变更 (Behavior Changes)

**旧模式：**
```text
Query
  -> Hybrid Retrieve (混合检索)
  -> Model Rerank (模型重排)
  -> Parent Restore (父文档还原)
```

**当前模式：**
```text
Query
  -> Dense + BM25
  -> RRF
  -> Candidate Pruning (轻量候选裁剪)
  -> Parent Restore (父文档还原)
```

**发生了什么改变：**
- 基于大模型的文档重排 (Document model reranking) 已从默认主干路径中移除。
- 排序质量现在主要取决于召回深度 (Recall depth)、分块策略 (Chunking) 和 RRF 融合。
- 裁剪阶段变得更轻量、更廉价（计算资源）。

## 重要的兼容性说明 (Important Compatibility Note)

`context_pruning.method="reranker"` 这个配置项依然被系统接受。
**但是，它不再意味着“使用基于大模型的重排器”。**

它现在已被重定向映射至轻量级的本地评分算法。这样做是为了让旧的配置文件能够继续工作，而不会强制在运行时引入沉重的模型推理依赖。

## 迁移检查清单 (Migration Checklist)

1. 设置 `reranker.model_name=""`
2. 设置 `local_models.rerank_model=""`
3. 从已启用的 nodes 列表中移除 `rerank_docs` (如果存在的话)
4. 保持 `prompt.context_pruning.method="auto"`，除非你有特殊理由不这么做
5. 在修改裁剪阈值之前，先验证底层检索的召回质量

## 验证检查清单 (Validation Checklist)

迁移完成后，请验证：

- BM25 依然能捕获精确的专业术语和标识符
- 密集检索依然能捕获同义词和语义变体
- 父文档还原 (Parent restore) 依然能为生成阶段提供足够丰富的上下文
- 延迟 (Latency) 得到改善或保持稳定
- Prompt 长度在可控范围内减少或保持一致

## 常见错误 (Common Mistakes)

- 在调优召回深度之前，先去调优了裁剪参数
- 误以为包含 `reranker` 字眼的配置项依然会调用大模型进行重排
- 过早地移除兼容性字段，导致旧部署环境崩溃

## 最简推荐配置 (Minimal Recommended Config)

```json
{
  "embeddings": {
    "model_name": "Qwen/Qwen3-Embedding-0.6B"
  },
  "rag": {
    "retrieval": {
      "mode": "hybrid",
      "dense_k": 20,
      "sparse_k": 20,
      "candidate_k": 20,
      "final_k": 3,
      "rrf_k": 60
    }
  },
  "prompt": {
    "context_pruning": {
      "enabled": true,
      "method": "auto"
    }
  },
  "reranker": {
    "model_name": ""
  },
  "local_models": {
    "rerank_model": ""
  }
}
```
