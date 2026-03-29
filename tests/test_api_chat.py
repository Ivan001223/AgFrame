from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import chat as chat_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_workbench_invoke_persists_history_and_normalizes_identity(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    saved: dict[str, Any] = {}

    class _State:
        values = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "context": {"context_pruning": {"method": "heuristic"}},
            "interrupted": False,
        }

    class _GraphApp:
        async def ainvoke(self, input_value: dict[str, Any], config: dict[str, Any]):
            captured["input"] = input_value
            captured["config"] = config
            return {"ok": True}

        async def aget_state(self, config: dict[str, Any]):
            captured["state_config"] = config
            return _State()

    def _persist_session_messages(
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        background_tasks: Any = None,
        title: str | None = None,
    ):
        saved["user_id"] = user_id
        saved["session_id"] = session_id
        saved["messages"] = messages
        saved["title"] = title
        return {
            "id": session_id,
            "messages": messages,
        }

    monkeypatch.setattr(chat_api, "get_chat_graph_app", lambda: _GraphApp())
    monkeypatch.setattr(chat_api, "persist_session_messages", _persist_session_messages)

    app = FastAPI()
    app.include_router(chat_api.router)
    app.dependency_overrides[chat_api.get_current_active_user] = lambda: _U(username="u1")
    client = TestClient(app)

    response = client.post(
        "/chat/workbench-invoke",
        json={
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "context": {"session_id": "s1", "context_focus_hint": "focus"},
            },
            "config": {
                "configurable": {
                    "thread_id": "spoofed-session",
                    "user_id": "spoofed-user",
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s1"
    assert body["reply"] == "hello"
    assert body["messages"][-1]["role"] == "assistant"
    assert body["context"]["context_pruning"]["method"] == "heuristic"

    assert captured["config"]["configurable"]["thread_id"] == "s1"
    assert captured["config"]["configurable"]["user_id"] == "u1"
    assert captured["input"]["context"]["session_id"] == "s1"
    assert captured["state_config"]["configurable"]["thread_id"] == "s1"

    assert saved["user_id"] == "u1"
    assert saved["session_id"] == "s1"
    assert saved["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
