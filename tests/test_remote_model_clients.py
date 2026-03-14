from __future__ import annotations

from app.runtime.llm.embeddings import ModelEmbeddings
from app.runtime.llm.reranker import ModelReranker


def test_vllm_embeddings_client(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }

    def _post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("app.runtime.llm.embeddings.httpx.post", _post)

    client = ModelEmbeddings(
        config={
            "model_manager": {"provider": "vllm"},
            "embeddings": {
                "provider": "vllm",
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "base_url": "http://198.2.168.108:28800",
                "timeout_seconds": 9,
                "doc_prefix": "doc:",
            },
            "local_models": {},
        }
    )

    vectors = client.embed_documents(["a", "b"])

    assert captured["url"] == "http://198.2.168.108:28800/v1/embeddings"
    assert captured["json"]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert captured["json"]["input"] == ["doc:a", "doc:b"]
    assert captured["timeout"] == 9
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_vllm_reranker_client(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }

    def _post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("app.runtime.llm.reranker.httpx.post", _post)

    client = ModelReranker(
        config={
            "model_manager": {"provider": "vllm"},
            "reranker": {
                "provider": "vllm",
                "model_name": "Qwen/Qwen3-Reranker-0.6B",
                "base_url": "http://198.2.168.108:28880",
                "timeout_seconds": 7,
            },
            "local_models": {},
        }
    )

    ranked = client.rerank("query", ["doc1", "doc2"], top_k=1)

    assert captured["url"] == "http://198.2.168.108:28880/v1/rerank"
    assert captured["json"]["model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert captured["json"]["query"] == "query"
    assert captured["json"]["documents"] == ["doc1", "doc2"]
    assert captured["json"]["top_n"] == 1
    assert captured["timeout"] == 7
    assert ranked == [("doc2", 0.9, 1)]
