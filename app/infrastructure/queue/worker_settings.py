from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

from app.infrastructure.queue.arq_jobs import ingest_pdf, resume_harness_task, run_harness_task
from app.infrastructure.queue.client import INGEST_QUEUE_NAME, RESUME_QUEUE_NAME, RUNTIME_QUEUE_NAME


def _redis_settings() -> RedisSettings:
    url = os.getenv("REDIS_URL") or "redis://:redissecret@localhost:6379/0"
    return RedisSettings.from_dsn(url)


class _BaseWorkerSettings:
    redis_settings = _redis_settings()
    keep_result = 0
    job_timeout = 60 * 60

    @staticmethod
    async def on_startup(ctx: dict[str, Any]) -> None:
        return None

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        return None


class IngestWorkerSettings(_BaseWorkerSettings):
    functions = [ingest_pdf]
    queue_name = INGEST_QUEUE_NAME
    max_jobs = 2


class RuntimeWorkerSettings(_BaseWorkerSettings):
    functions = [run_harness_task]
    queue_name = RUNTIME_QUEUE_NAME
    max_jobs = 4


class ResumeWorkerSettings(_BaseWorkerSettings):
    functions = [resume_harness_task]
    queue_name = RESUME_QUEUE_NAME
    max_jobs = 2


class WorkerSettings(RuntimeWorkerSettings):
    pass
