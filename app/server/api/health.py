"""
健康检查端点
"""
from sqlalchemy import text

from fastapi import APIRouter, status

from app.infrastructure.database.orm import get_engine
from app.infrastructure.database.schema import is_database_ready
from app.infrastructure.config.settings import settings
from app.infrastructure.queue.redis_client import get_redis
from app.runtime.llm.llm_factory import get_llm

router = APIRouter(tags=["health"])


def _check_vectorstore() -> bool:
    """检查 pgvector 扩展和向量运算是否可用。"""
    if not is_database_ready():
        return False
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector"))
        return True
    except Exception:
        return False


def _check_llm() -> bool:
    """检查 LLM 配置或 provider 构造是否可用。"""
    try:
        llm = get_llm(temperature=0, streaming=False, json_mode=False)
        return llm is not None
    except Exception:
        return False


def _check_embeddings_config() -> bool:
    emb_cfg = settings.embeddings
    return bool(
        emb_cfg.base_url
        or emb_cfg.model_name
        or settings.local_models.embedding_model
    )


def _check_reranker_config() -> bool:
    rr_cfg = settings.reranker
    return bool(
        rr_cfg.base_url
        or rr_cfg.model_name
        or settings.local_models.rerank_model
    )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    健康检查端点。

    返回服务的基本状态信息。
    """
    return {
        "status": "healthy",
        "app_name": settings.general.app_name,
        "version": "0.1.1",
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    就绪检查端点。

    用于 Kubernetes 就绪探针，可以扩展检查数据库、Redis 等依赖。
    """
    db_ready = is_database_ready()
    try:
        await get_redis().ping()
        redis_ready = True
    except Exception:
        redis_ready = False
    vector_ready = _check_vectorstore()
    llm_ready = _check_llm()
    embeddings_ready = _check_embeddings_config()
    reranker_ready = _check_reranker_config()

    checks = {
        "app": "ready",
        "database": "ready" if db_ready else "not_ready",
        "redis": "ready" if redis_ready else "not_ready",
        "vectorstore": "ready" if vector_ready else "not_ready",
    }
    components = {
        "llm": "ready" if llm_ready else "not_ready",
        "embeddings": "configured" if embeddings_ready else "not_configured",
        "reranker": "configured" if reranker_ready else "not_configured",
    }

    all_ready = all(v == "ready" for v in checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "components": components,
    }


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    存活检查端点。

    用于 Kubernetes 存活探针，只检查应用是否运行。
    """
    return {"status": "alive"}
