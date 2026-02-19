"""
健康检查端点
"""
from fastapi import APIRouter, status

from app.infrastructure.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    健康检查端点。

    返回服务的基本状态信息。
    """
    return {
        "status": "healthy",
        "app_name": settings.general.app_name,
        "version": "1.0.1",
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    就绪检查端点。

    用于 Kubernetes 就绪探针，可以扩展检查数据库、Redis 等依赖。
    """
    checks = {
        "app": "ready",
    }

    # 可以扩展更多检查
    # 例如：数据库连接、Redis 连接等

    all_ready = all(v == "ready" for v in checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    存活检查端点。

    用于 Kubernetes 存活探针，只检查应用是否运行。
    """
    return {"status": "alive"}
