from __future__ import annotations

from copy import deepcopy

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from app.infrastructure.checkpoint.redis_store import AsyncRedisSaverWrapper


class _CheckpointTuple:
    def __init__(self, config, checkpoint, metadata):
        self.config = config
        self.checkpoint = checkpoint
        self.metadata = metadata


class _FakeSaver:
    def __init__(self):
        self.tuples: dict[tuple[str, str], _CheckpointTuple] = {}
        self.saved_calls: list[dict[str, object]] = []

    async def aget_tuple(self, config):
        configurable = config["configurable"]
        key = (
            configurable["thread_id"],
            configurable.get("checkpoint_ns", ""),
        )
        return self.tuples.get(key)

    async def aput(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        checkpoint_tuple = _CheckpointTuple(
            next_config,
            deepcopy(checkpoint),
            deepcopy(metadata),
        )
        self.tuples[(thread_id, checkpoint_ns)] = checkpoint_tuple
        self.saved_calls.append(
            {
                "config": deepcopy(config),
                "checkpoint": deepcopy(checkpoint),
                "metadata": deepcopy(metadata),
                "new_versions": deepcopy(new_versions),
            }
        )
        return next_config

    def get_next_version(self, current, _channel):
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.0000000000000000"


@pytest.mark.anyio
async def test_checkpoint_store_save_preserves_existing_checkpoint_structure():
    checkpoint_store = AsyncRedisSaverWrapper()
    fake_saver = _FakeSaver()
    checkpoint_store._saver = fake_saver

    existing_checkpoint = empty_checkpoint()
    existing_checkpoint.update(
        {
            "id": "cp-old",
            "ts": "2026-03-26T00:00:00+00:00",
            "channel_values": {
                "foo": {"nested": [1, 2]},
                "interrupted": True,
                "action_required": {"action_type": "deploy", "approved": False},
            },
            "channel_versions": {
                "foo": "00000000000000000000000000000005.0000000000000000",
                "interrupted": "00000000000000000000000000000005.0000000000000000",
                "action_required": "00000000000000000000000000000005.0000000000000000",
            },
            "versions_seen": {"node": {"foo": "00000000000000000000000000000004.0000000000000000"}},
            "updated_channels": ["foo"],
            "pending_sends": ["send-1"],
        }
    )
    fake_saver.tuples[("session-1", "default")] = _CheckpointTuple(
        {
            "configurable": {
                "thread_id": "session-1",
                "checkpoint_ns": "default",
                "checkpoint_id": "cp-old",
            }
        },
        deepcopy(existing_checkpoint),
        {"source": "loop", "step": 7, "saved_at": "2026-03-26T00:00:01+00:00"},
    )

    saved = await checkpoint_store.save(
        "session-1",
        {
            "interrupted": False,
            "action_required": {"action_type": "deploy", "approved": True, "approved_by": "u1"},
        },
    )
    loaded = await checkpoint_store.load("session-1")

    assert saved["checkpoint_id"] != "cp-old"
    assert loaded is not None
    assert loaded["checkpoint"]["checkpoint_id"] == saved["checkpoint_id"]
    assert loaded["checkpoint"]["checkpoint_ns"] == "default"
    assert loaded["checkpoint"]["channel_values"]["foo"] == {"nested": [1, 2]}
    assert loaded["checkpoint"]["interrupted"] is False
    assert "foo" not in loaded["checkpoint"]
    assert loaded["checkpoint"]["channel_values"]["interrupted"] is False
    assert loaded["checkpoint"]["action_required"]["approved"] is True
    assert loaded["checkpoint"]["channel_values"]["action_required"]["approved_by"] == "u1"
    assert loaded["checkpoint"]["versions_seen"] == existing_checkpoint["versions_seen"]
    assert loaded["checkpoint"]["pending_sends"] == ["send-1"]
    assert loaded["updated_at"]

    saved_call = fake_saver.saved_calls[-1]
    saved_checkpoint = saved_call["checkpoint"]
    assert saved_call["config"]["configurable"]["checkpoint_id"] == "cp-old"
    assert saved_checkpoint["id"] == saved["checkpoint_id"]
    assert saved_checkpoint["id"] != "cp-old"
    assert saved_checkpoint["channel_values"]["foo"] == {"nested": [1, 2]}
    assert saved_checkpoint["versions_seen"] == existing_checkpoint["versions_seen"]
    assert saved_checkpoint["pending_sends"] == ["send-1"]
    assert saved_checkpoint["channel_versions"]["foo"] == existing_checkpoint["channel_versions"]["foo"]
    assert saved_checkpoint["channel_versions"]["interrupted"] == "00000000000000000000000000000006.0000000000000000"
    assert saved_checkpoint["channel_versions"]["action_required"] == "00000000000000000000000000000006.0000000000000000"
    assert saved_checkpoint["updated_channels"] == ["interrupted", "action_required"]
    assert saved_call["new_versions"] == {
        "interrupted": "00000000000000000000000000000006.0000000000000000",
        "action_required": "00000000000000000000000000000006.0000000000000000",
    }


@pytest.mark.anyio
async def test_checkpoint_store_load_adapts_real_checkpoint_without_losing_internals():
    checkpoint_store = AsyncRedisSaverWrapper()
    fake_saver = _FakeSaver()
    checkpoint_store._saver = fake_saver

    checkpoint = empty_checkpoint()
    checkpoint.update(
        {
            "id": "cp-real",
            "ts": "2026-03-26T00:00:00+00:00",
            "channel_values": {
                "foo": "bar",
                "interrupted": True,
                "action_required": {"action_type": "review", "approved": False},
            },
            "channel_versions": {
                "foo": "00000000000000000000000000000002.0000000000000000",
                "interrupted": "00000000000000000000000000000003.0000000000000000",
                "action_required": "00000000000000000000000000000003.0000000000000000",
            },
            "versions_seen": {"worker": {"foo": "00000000000000000000000000000001.0000000000000000"}},
            "updated_channels": ["foo", "interrupted", "action_required"],
            "pending_sends": ["send-2"],
        }
    )
    fake_saver.tuples[("session-2", "")] = _CheckpointTuple(
        {
            "configurable": {
                "thread_id": "session-2",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-real",
            }
        },
        deepcopy(checkpoint),
        {"source": "loop", "step": 3, "saved_at": "2026-03-26T00:00:02+00:00"},
    )

    loaded = await checkpoint_store.load("session-2")

    assert loaded == {
        "checkpoint": {
            **checkpoint,
            "interrupted": True,
            "action_required": {"action_type": "review", "approved": False},
            "checkpoint_id": "cp-real",
            "checkpoint_ns": "",
        },
        "updated_at": "2026-03-26T00:00:02+00:00",
    }
    assert loaded["checkpoint"]["channel_values"]["foo"] == "bar"
    assert "foo" not in loaded["checkpoint"]


def test_checkpoint_store_get_next_version_supports_string_versions():
    checkpoint_store = AsyncRedisSaverWrapper()

    assert checkpoint_store.get_next_version(None, None) == "00000000000000000000000000000001.0000000000000000"
    assert checkpoint_store.get_next_version(4, None) == "00000000000000000000000000000005.0000000000000000"
    assert (
        checkpoint_store.get_next_version(
            "00000000000000000000000000000009.0000000000000000",
            None,
        )
        == "00000000000000000000000000000010.0000000000000000"
    )
