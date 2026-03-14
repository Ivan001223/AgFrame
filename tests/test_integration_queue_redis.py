from __future__ import annotations

import os
import socket

import pytest

from app.infrastructure.queue import redis_client


def _port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _redis_available() -> bool:
    url = os.getenv("REDIS_URL", "redis://:redissecret@localhost:6379/0")
    host = "127.0.0.1"
    port = 6379
    if "://" in url:
        rest = url.split("://", 1)[1]
        if "@" in rest:
            rest = rest.split("@", 1)[1]
        host_port = rest.split("/", 1)[0]
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 6379
        elif host_port:
            host = host_port
    return _port_open(host or "127.0.0.1", port)


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis service is not available for integration test",
)


@pytest.mark.anyio
async def test_task_operation_claim_and_release_integration():
    redis_client._redis = None
    op_key = "itest:queue:claim-release"
    task_id = "task-a"

    await redis_client.release_task_operation(op_key)
    claimed = await redis_client.claim_task_operation(op_key, task_id, ttl_seconds=30)
    assert claimed == task_id

    claimed_again = await redis_client.claim_task_operation(op_key, "task-b", ttl_seconds=30)
    assert claimed_again == task_id

    await redis_client.release_task_operation(op_key, expected_task_id=task_id)
    reclaimed = await redis_client.claim_task_operation(op_key, "task-c", ttl_seconds=30)
    assert reclaimed == "task-c"

    await redis_client.release_task_operation(op_key, expected_task_id="task-c")


@pytest.mark.anyio
async def test_task_hash_roundtrip_integration():
    redis_client._redis = None
    task_id = "itest-task-roundtrip"
    await redis_client.init_task(task_id, {"status": "queued", "progress": 0})
    task = await redis_client.get_task(task_id)
    assert task["status"] == "queued"

    await redis_client.update_task(task_id, {"status": "running", "progress": 10})
    updated = await redis_client.get_task(task_id)
    assert updated["status"] == "running"
    assert updated["progress"] == "10"


@pytest.mark.anyio
async def test_task_incidents_roundtrip_integration():
    redis_client._redis = None
    created = await redis_client.append_task_incident(
        {
            "task_id": "incident-a",
            "user_id": "u1",
            "error_code": "embedding_failed",
            "timestamp": 123,
        },
        max_items=20,
    )
    incidents = await redis_client.list_task_incidents(limit=5)
    assert incidents
    assert incidents[0]["task_id"] == "incident-a"
    assert incidents[0]["error_code"] == "embedding_failed"
    assert incidents[0]["incident_id"] == created["incident_id"]
    assert incidents[0]["handled"] is False


@pytest.mark.anyio
async def test_task_incident_update_integration():
    redis_client._redis = None
    await redis_client.get_redis().delete(redis_client.task_incidents_key())
    created = await redis_client.append_task_incident(
        {
            "task_id": "incident-b",
            "user_id": "u1",
            "error_code": "vectorstore_write_failed",
            "timestamp": 456,
        },
        max_items=20,
    )
    updated = await redis_client.update_task_incident(
        created["incident_id"],
        {"handled": True, "handled_at": 1_700_000_000},
    )
    assert updated is not None
    assert updated["handled"] is True
    assert updated["handled_at"] == 1_700_000_000
    incidents = await redis_client.list_task_incidents(limit=5)
    assert incidents[0]["incident_id"] == created["incident_id"]
    assert incidents[0]["handled"] is True
