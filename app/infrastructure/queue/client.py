from __future__ import annotations

import os
from typing import Optional

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.infrastructure.config.config_manager import config_manager


def _redis_settings() -> RedisSettings:
    cfg = config_manager.get_config() or {}
    queue_cfg = cfg.get("queue") or {}
    url = (
        queue_cfg.get("redis_url")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    return RedisSettings.from_dsn(str(url))


_pool: Optional[ArqRedis] = None


async def get_arq_pool() -> ArqRedis:
    global _pool
    if _pool is not None:
        return _pool
    _pool = await create_pool(_redis_settings())
    return _pool


async def enqueue_ingest_pdf(task_id: str, file_path: str, user_id: str = None) -> str:
    pool = await get_arq_pool()
    job = await pool.enqueue_job("ingest_pdf", task_id, file_path, user_id)
    return str(job.job_id)
