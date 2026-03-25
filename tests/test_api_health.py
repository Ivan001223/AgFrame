from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import health


def test_health_endpoints():
    class _Redis:
        async def ping(self):
            return True

    health.is_database_ready = lambda: True
    health.get_redis = lambda: _Redis()
    health._check_vectorstore = lambda: True
    health._check_llm = lambda: True
    health._check_embeddings_config = lambda: True
    health._check_reranker_config = lambda: True

    app = FastAPI()
    app.include_router(health.router)
    c = TestClient(app)

    r1 = c.get("/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "healthy"

    r2 = c.get("/health/ready")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ready"
    assert r2.json()["checks"]["database"] == "ready"
    assert r2.json()["checks"]["redis"] == "ready"
    assert r2.json()["checks"]["vectorstore"] == "ready"
    assert r2.json()["components"]["llm"] == "ready"
    assert r2.json()["components"]["embeddings"] == "configured"
    assert r2.json()["components"]["retrieval"] == "hybrid_rrf"
    assert r2.json()["components"]["context_pruning"] == "lightweight_ranker"
    assert r2.json()["components"]["reranker"] == "configured"

    r3 = c.get("/health/live")
    assert r3.status_code == 200
    assert r3.json()["status"] == "alive"


def test_health_ready_reports_dependency_failures():
    class _Redis:
        async def ping(self):
            raise RuntimeError("down")

    health.is_database_ready = lambda: False
    health.get_redis = lambda: _Redis()
    health._check_vectorstore = lambda: False
    health._check_llm = lambda: False
    health._check_embeddings_config = lambda: False
    health._check_reranker_config = lambda: False

    app = FastAPI()
    app.include_router(health.router)
    c = TestClient(app)

    r = c.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "not_ready"
    assert r.json()["checks"]["database"] == "not_ready"
    assert r.json()["checks"]["redis"] == "not_ready"
    assert r.json()["checks"]["vectorstore"] == "not_ready"
    assert r.json()["components"]["llm"] == "not_ready"
    assert r.json()["components"]["embeddings"] == "not_configured"
    assert r.json()["components"]["retrieval"] == "hybrid_rrf"
    assert r.json()["components"]["context_pruning"] == "lightweight_ranker"
    assert r.json()["components"]["reranker"] == "not_configured"
