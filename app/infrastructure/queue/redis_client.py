from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from redis.asyncio import Redis

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


def _get_redis_url() -> str:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://:redissecret@localhost:6379/0"
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


def task_operation_key(operation_key: str) -> str:
    return f"taskop:{operation_key}"


def task_incidents_key() -> str:
    return "taskincidents"


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


async def claim_task_operation(
    operation_key: str,
    task_id: str,
    *,
    ttl_seconds: int = 3600,
) -> str:
    r = get_redis()
    key = task_operation_key(operation_key)
    claimed = await r.set(key, task_id, ex=ttl_seconds, nx=True)
    if claimed:
        return task_id
    existing = await r.get(key)
    return str(existing or task_id)


async def release_task_operation(operation_key: str, *, expected_task_id: str | None = None) -> None:
    if not operation_key:
        return
    r = get_redis()
    key = task_operation_key(operation_key)
    if expected_task_id is None:
        await r.delete(key)
        return
    existing = await r.get(key)
    if existing == expected_task_id:
        await r.delete(key)


async def append_task_incident(
    incident: dict[str, Any],
    *,
    max_items: int = 200,
) -> dict[str, Any]:
    r = get_redis()
    normalized = dict(incident or {})
    now = int(time.time())
    normalized.setdefault("incident_id", str(uuid.uuid4()))
    normalized.setdefault("handled", False)
    normalized.setdefault("archived", False)
    normalized.setdefault("handled_at", None)
    normalized.setdefault("archived_at", None)
    normalized.setdefault("updated_at", now)
    payload = json.dumps(normalized, ensure_ascii=True)
    key = task_incidents_key()
    await r.lpush(key, payload)
    await r.ltrim(key, 0, max_items - 1)
    return normalized


async def list_task_incidents(*, limit: int = 20) -> list[dict[str, Any]]:
    r = get_redis()
    raw_items = await r.lrange(task_incidents_key(), 0, max(0, limit - 1))
    out: list[dict[str, Any]] = []
    for item in raw_items:
        parsed = _parse_incident_payload(item)
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


async def update_task_incident(
    incident_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    if not incident_id:
        return None
    r = get_redis()
    key = task_incidents_key()
    raw_items = await r.lrange(key, 0, -1)
    now = int(time.time())
    for index, item in enumerate(raw_items):
        parsed = _parse_incident_payload(item)
        if not isinstance(parsed, dict):
            continue
        if str(parsed.get("incident_id") or "") != incident_id:
            continue
        updated = dict(parsed)
        updated.update(updates or {})
        updated["updated_at"] = now
        payload = json.dumps(updated, ensure_ascii=True)
        await r.lset(key, index, payload)
        return updated
    return None


def _parse_incident_payload(item: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(item)
    except json.JSONDecodeError as exc:
        logger.debug("Failed to decode task incident payload: %s", exc)
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
