from __future__ import annotations

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.infrastructure.config.settings import settings


def _redis_settings() -> RedisSettings:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://localhost:6379/0"
    return RedisSettings.from_dsn(str(url))


_pool: ArqRedis | None = None


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
