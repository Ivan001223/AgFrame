from __future__ import annotations

from typing import Any

from redis.asyncio import Redis

from app.infrastructure.config.settings import settings


def _get_redis_url() -> str:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://localhost:6379/0"
    return str(url)


_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is not None:
        return _redis
    _redis = Redis.from_url(_get_redis_url(), decode_responses=True)
    return _redis


def task_key(task_id: str) -> str:
    return f"task:{task_id}"


async def init_task(task_id: str, fields: dict[str, Any]) -> None:
    r = get_redis()
    await r.hset(task_key(task_id), mapping={k: str(v) for k, v in (fields or {}).items()})


async def update_task(task_id: str, fields: dict[str, Any]) -> None:
    if not fields:
        return
    r = get_redis()
    await r.hset(task_key(task_id), mapping={k: str(v) for k, v in fields.items()})


async def get_task(task_id: str) -> dict[str, str]:
    r = get_redis()
    out = await r.hgetall(task_key(task_id))
    return dict(out or {})

