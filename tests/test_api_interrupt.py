from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import empty_checkpoint

from app.infrastructure.checkpoint.redis_store import AsyncRedisSaverWrapper
from app.server.api import interrupt as interrupt_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


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
        self.tuples[(thread_id, checkpoint_ns)] = _CheckpointTuple(
            next_config,
            deepcopy(checkpoint),
            deepcopy(metadata),
        )
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


def _seed_checkpoint(
    fake_saver: _FakeSaver,
    *,
    session_id: str,
    checkpoint_id: str,
    checkpoint_ns: str = "",
    with_action_required: bool = True,
    user_id: str = "u1",
):
    checkpoint = empty_checkpoint()
    channel_values = {
        "foo": {"keep": True},
        "interrupted": True,
        "user_id": user_id,
    }
    channel_versions = {
        "foo": "00000000000000000000000000000002.0000000000000000",
        "interrupted": "00000000000000000000000000000003.0000000000000000",
    }
    updated_channels = ["foo", "interrupted"]
    if with_action_required:
        channel_values["action_required"] = {
            "action_type": "deploy",
            "approved": False,
            "payload": {"target": "prod"},
        }
        channel_versions["action_required"] = "00000000000000000000000000000003.0000000000000000"
        updated_channels.append("action_required")

    checkpoint.update(
        {
            "id": checkpoint_id,
            "ts": "2026-03-26T00:00:00+00:00",
            "channel_values": channel_values,
            "channel_versions": channel_versions,
            "versions_seen": {
                "node": {"foo": "00000000000000000000000000000001.0000000000000000"}
            },
            "updated_channels": updated_channels,
            "pending_sends": ["send-1"],
        }
    )
    fake_saver.tuples[(session_id, checkpoint_ns)] = _CheckpointTuple(
        {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        },
        checkpoint,
        {"source": "loop", "step": 3, "saved_at": "t0"},
    )


def test_interrupt_endpoints(monkeypatch: pytest.MonkeyPatch):
    checkpoint_store = AsyncRedisSaverWrapper()
    fake_saver = _FakeSaver()
    checkpoint_store._saver = fake_saver
    recorded_events: list[dict[str, object]] = []

    class _EventService:
        def record(self, **kwargs):
            recorded_events.append(kwargs)
            return {"event_id": f"he-{len(recorded_events)}", **kwargs}

        def list_for_session(self, *, session_id: str, user_id: str | None = None, limit: int = 100):
            events = [
                {"event_id": f"he-{index}", **event}
                for index, event in enumerate(recorded_events, start=1)
                if event.get("session_id") == session_id
                and (user_id is None or event.get("user_id") == user_id)
            ]
            return events[:limit]

    monkeypatch.setattr(interrupt_api, "checkpoint_store", checkpoint_store)
    monkeypatch.setattr(interrupt_api, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(interrupt_api, "get_interrupt_event_service", lambda: _EventService())
    monkeypatch.setattr(
        interrupt_api.history_manager,
        "save_session",
        lambda user_id, session_id, messages, title=None: {
            "id": session_id,
            "title": title or "history",
            "messages": messages,
        },
    )
    app = FastAPI()
    app.include_router(interrupt_api.router)
    app.dependency_overrides[interrupt_api.get_current_active_user] = lambda: _U(username="u1")
    c = TestClient(app)

    r404 = c.get("/interrupt/s1")
    assert r404.status_code == 404

    _seed_checkpoint(fake_saver, session_id="s1", checkpoint_id="cp-1", checkpoint_ns="")
    r = c.get("/interrupt/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["interrupted"] is True
    assert body["action_required"]["payload"] == {"target": "prod"}

    resume_ready = c.get("/interrupt/s1/resume")
    assert resume_ready.status_code == 400

    approve = c.post("/interrupt/s1/approve", json={"approved": True, "comment": "ship with guardrails"})
    assert approve.status_code == 200
    assert approve.json()["approved"] is True
    assert approve.json()["approved_by"] == "u1"

    resume_ready = c.get("/interrupt/s1/resume")
    assert resume_ready.status_code == 200
    resume_payload = resume_ready.json()["resume_payload"]
    assert resume_payload["configurable"]["thread_id"] == "s1"
    assert resume_payload["configurable"]["checkpoint_ns"] == ""
    assert resume_payload["configurable"]["checkpoint_id"] == fake_saver.saved_calls[-1]["checkpoint"]["id"]

    class _ResumeService:
        async def resume_approved_session(self, *, session_id: str, checkpoint: dict[str, object]):
            assert session_id == "s1"
            assert checkpoint["checkpoint"]["channel_values"]["user_id"] == "u1"
            return {
                "ok": True,
                "interrupted": False,
                "reply": "approved reply",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "approved reply"},
                ],
                "context": {"context_pruning": {"method": "heuristic"}},
            }

    monkeypatch.setattr(interrupt_api, "get_graph_resume_service", lambda: _ResumeService())
    resumed = c.post("/interrupt/s1/resume")
    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert resumed_body["resumed"] is True
    assert resumed_body["reply"] == "approved reply"
    assert resumed_body["messages"][-1]["role"] == "assistant"
    assert resumed_body["context"]["context_pruning"]["method"] == "heuristic"

    saved_call = fake_saver.saved_calls[-1]
    assert saved_call["config"]["configurable"]["checkpoint_id"] == "cp-1"
    assert saved_call["checkpoint"]["id"] != "cp-1"
    assert saved_call["checkpoint"]["channel_values"]["foo"] == {"keep": True}
    assert saved_call["checkpoint"]["versions_seen"] == {
        "node": {"foo": "00000000000000000000000000000001.0000000000000000"}
    }
    assert saved_call["checkpoint"]["pending_sends"] == ["send-1"]

    refreshed = c.get("/interrupt/s1")
    assert refreshed.status_code == 200
    refreshed_body = refreshed.json()
    assert refreshed_body["action_required"]["approved"] is True
    assert refreshed_body["action_required"]["approved_by"] == "u1"

    s1_events = c.get("/interrupt/s1/events")
    assert s1_events.status_code == 200
    s1_event_payload = s1_events.json()["events"]
    assert [event["event_type"] for event in s1_event_payload] == [
        "interrupt.resume_blocked",
        "interrupt.approved",
        "interrupt.resume_requested",
        "interrupt.resumed",
    ]
    assert s1_event_payload[1]["details"]["comment"] == "ship with guardrails"

    _seed_checkpoint(fake_saver, session_id="s2", checkpoint_id="cp-2", checkpoint_ns="default")
    reject = c.post("/interrupt/s2/approve", json={"approved": False, "comment": "hold deployment"})
    assert reject.status_code == 200
    assert reject.json()["approved"] is False
    assert fake_saver.saved_calls[-1]["config"]["configurable"]["checkpoint_id"] == "cp-2"
    assert fake_saver.saved_calls[-1]["checkpoint"]["id"] != "cp-2"

    rejected_status = c.get("/interrupt/s2")
    assert rejected_status.status_code == 200
    assert rejected_status.json()["interrupted"] is True
    assert rejected_status.json()["action_required"]["approved"] is False
    assert rejected_status.json()["action_required"]["approved_by"] == "u1"

    resume_blocked = c.get("/interrupt/s2/resume")
    assert resume_blocked.status_code == 400

    s2_events = c.get("/interrupt/s2/events")
    assert s2_events.status_code == 200
    s2_event_payload = s2_events.json()["events"]
    assert [event["event_type"] for event in s2_event_payload] == [
        "interrupt.rejected",
        "interrupt.resume_blocked",
    ]
    assert s2_event_payload[0]["details"]["comment"] == "hold deployment"

    _seed_checkpoint(
        fake_saver,
        session_id="s3",
        checkpoint_id="cp-3",
        checkpoint_ns="",
        with_action_required=False,
    )
    r400 = c.post("/interrupt/s3/approve", json={"approved": True})
    assert r400.status_code == 400


def test_interrupt_endpoints_enforce_session_isolation(monkeypatch: pytest.MonkeyPatch):
    checkpoint_store = AsyncRedisSaverWrapper()
    fake_saver = _FakeSaver()
    checkpoint_store._saver = fake_saver

    monkeypatch.setattr(interrupt_api, "checkpoint_store", checkpoint_store)
    app = FastAPI()
    app.include_router(interrupt_api.router)
    app.dependency_overrides[interrupt_api.get_current_active_user] = lambda: _U(username="u1")
    c = TestClient(app)

    _seed_checkpoint(fake_saver, session_id="foreign", checkpoint_id="cp-9", user_id="u2")

    r_get = c.get("/interrupt/foreign")
    assert r_get.status_code == 403

    r_approve = c.post("/interrupt/foreign/approve", json={"approved": True})
    assert r_approve.status_code == 403

    r_resume = c.post("/interrupt/foreign/resume")
    assert r_resume.status_code == 403


def test_interrupt_resume_persists_history(monkeypatch: pytest.MonkeyPatch):
    checkpoint_store = AsyncRedisSaverWrapper()
    fake_saver = _FakeSaver()
    checkpoint_store._saver = fake_saver

    class _EventService:
        def record(self, **kwargs):
            return {"event_id": "he-1", **kwargs}

        def list_for_session(self, *, session_id: str, user_id: str | None = None, limit: int = 100):
            return []

    monkeypatch.setattr(interrupt_api, "checkpoint_store", checkpoint_store)
    monkeypatch.setattr(interrupt_api, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(interrupt_api, "get_interrupt_event_service", lambda: _EventService())

    saved: dict[str, object] = {}

    def _save_session(user_id: str, session_id: str, messages: list[dict[str, str]], title: str | None = None):
        saved["user_id"] = user_id
        saved["session_id"] = session_id
        saved["messages"] = messages
        saved["title"] = title
        return {
            "id": session_id,
            "title": title or "smoke",
            "messages": messages,
        }

    monkeypatch.setattr(interrupt_api.history_manager, "save_session", _save_session)

    app = FastAPI()
    app.include_router(interrupt_api.router)
    app.dependency_overrides[interrupt_api.get_current_active_user] = lambda: _U(username="u1")
    c = TestClient(app)

    _seed_checkpoint(fake_saver, session_id="persisted", checkpoint_id="cp-10", checkpoint_ns="")

    class _ResumeService:
        async def resume_approved_session(self, *, session_id: str, checkpoint: dict[str, object]):
            assert session_id == "persisted"
            return {
                "ok": True,
                "interrupted": False,
                "reply": "approved reply",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "approved reply"},
                ],
                "context": {"context_pruning": {"method": "heuristic"}},
            }

    monkeypatch.setattr(interrupt_api, "get_graph_resume_service", lambda: _ResumeService())

    approved = c.post("/interrupt/persisted/approve", json={"approved": True})
    assert approved.status_code == 200

    resumed = c.post("/interrupt/persisted/resume")
    assert resumed.status_code == 200
    assert saved["user_id"] == "u1"
    assert saved["session_id"] == "persisted"
    assert saved["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "approved reply"},
    ]
