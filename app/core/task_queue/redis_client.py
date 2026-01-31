from __future__ import annotations

import os
from typing import Any, Dict, Optional

from redis.asyncio import Redis

from app.core.config.config_manager import config_manager


def _get_redis_url() -> str:
    cfg = config_manager.get_config() or {}
    queue_cfg = cfg.get("queue") or {}
    url = queue_cfg.get("redis_url") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    return str(url)


_redis: Optional[Redis] = None


def get_redis() -> Redis:
    global _redis
    if _redis is not None:
        return _redis
    _redis = Redis.from_url(_get_redis_url(), decode_responses=True)
    return _redis


def task_key(task_id: str) -> str:
    return f"task:{task_id}"


async def init_task(task_id: str, fields: Dict[str, Any]) -> None:
    r = get_redis()
    await r.hset(task_key(task_id), mapping={k: str(v) for k, v in (fields or {}).items()})


async def update_task(task_id: str, fields: Dict[str, Any]) -> None:
    if not fields:
        return
    r = get_redis()
    await r.hset(task_key(task_id), mapping={k: str(v) for k, v in fields.items()})


async def get_task(task_id: str) -> Dict[str, str]:
    r = get_redis()
    out = await r.hgetall(task_key(task_id))
    return dict(out or {})

