from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

from app.infrastructure.queue.arq_jobs import ingest_pdf


def _redis_settings() -> RedisSettings:
    url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    return RedisSettings.from_dsn(url)


class WorkerSettings:
    functions = [ingest_pdf]
    redis_settings = _redis_settings()
    keep_result = 0
    job_timeout = 60 * 60
    max_jobs = 4

    async def on_startup(self, ctx: dict[str, Any]) -> None:
        return None

    async def on_shutdown(self, ctx: dict[str, Any]) -> None:
        return None

