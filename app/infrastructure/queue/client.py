from __future__ import annotations

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.infrastructure.config.settings import settings

INGEST_QUEUE_NAME = "arq:queue:ingest"
RUNTIME_QUEUE_NAME = "arq:queue:runtime"
RESUME_QUEUE_NAME = "arq:queue:resume"


def _redis_settings() -> RedisSettings:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://:redissecret@localhost:6379/0"
    return RedisSettings.from_dsn(str(url))


_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    global _pool
    if _pool is not None:
        return _pool
    _pool = await create_pool(_redis_settings())
    return _pool


async def enqueue_ingest_pdf(
    task_id: str,
    storage_uri: str,
    user_id: str | None = None,
    knowledge_base_id: str | None = None,
) -> str:
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "ingest_pdf",
        task_id,
        storage_uri,
        user_id,
        knowledge_base_id,
        _job_id=f"ingest:{task_id}",
        _queue_name=INGEST_QUEUE_NAME,
    )
    return str(job.job_id) if job is not None else f"ingest:{task_id}"


async def enqueue_harness_run(run_id: str) -> str:
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "run_harness_task",
        run_id,
        _job_id=f"harness-run:{run_id}",
        _queue_name=RUNTIME_QUEUE_NAME,
    )
    return str(job.job_id) if job is not None else f"harness-run:{run_id}"


async def enqueue_harness_resume(run_id: str) -> str:
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "resume_harness_task",
        run_id,
        _job_id=f"harness-resume:{run_id}",
        _queue_name=RESUME_QUEUE_NAME,
    )
    return str(job.job_id) if job is not None else f"harness-resume:{run_id}"
