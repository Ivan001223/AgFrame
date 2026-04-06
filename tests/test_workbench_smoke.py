from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import chat as chat_api
from app.server.api import documents as documents_api
from app.server.api import history as history_api
from app.server.api import interrupt as interrupt_api
from app.server.api import memory as memory_api
from app.server.api import tasks as tasks_api
from app.server.api import upload as upload_api


@dataclass(frozen=True)
class _User:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_workbench_smoke_flow(tmp_path: Any, monkeypatch: Any):
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    docs_root.mkdir()
    uploads_root.mkdir()

    state: dict[str, Any] = {
        "tasks": {},
        "documents": [],
        "history": {},
        "chat_states": {},
        "checkpoints": {},
        "interrupt_events": [],
        "memory_items": [],
        "profile": {"facts": [{"text": "prefers concise answers"}]},
        "next_doc_id": 1,
    }
    user = _User()

    async def _init_task(task_id: str, payload: dict[str, Any]) -> None:
        state["tasks"][task_id] = {k: str(v) for k, v in payload.items()}

    async def _get_task(task_id: str) -> dict[str, str]:
        return dict(state["tasks"].get(task_id) or {})

    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any) -> str:
        return task_id

    async def _release_task_operation(
        operation_key: str,
        *,
        expected_task_id: str | None = None,
    ) -> None:
        return None

    async def _enqueue(task_id: str, file_path: str, user_id: str = None, **_: Any) -> None:
        task = state["tasks"][task_id]
        task.update({"status": "succeeded", "progress": "100", "step": "done"})
        doc_id = state["next_doc_id"]
        state["next_doc_id"] += 1
        state["documents"].append(
            {
                "doc_id": doc_id,
                "user_id": user_id,
                "source_path": file_path,
                "checksum": "checksum",
                "created_at": 1,
                "parent_chunk_count": 1,
                "embedding_count": 1,
                "preview": [{"parent_chunk_id": 1, "doc_id": doc_id, "page_num": 1, "content": "hello doc"}],
            }
        )

    class _DocStore:
        def find_by_checksum(self, *, user_id: str, checksum: str):
            return None

        def search_documents(self, user_id: str, *, include_all_users: bool = False, filename_query: str | None = None):
            q = str(filename_query or "").lower()
            docs = [
                d for d in state["documents"]
                if include_all_users or d["user_id"] == user_id
            ]
            if q:
                docs = [d for d in docs if q in d["source_path"].lower()]
            return [{k: v for k, v in d.items() if k != "preview"} for d in docs]

        def get_document(self, doc_id: int):
            for d in state["documents"]:
                if d["doc_id"] == doc_id:
                    return {k: v for k, v in d.items() if k != "preview"}
            return None

        def get_document_preview(self, doc_id: int, *, limit: int = 5):
            for d in state["documents"]:
                if d["doc_id"] == doc_id:
                    return list(d["preview"])[:limit]
            return []

        def delete_document(self, doc_id: int):
            before = len(state["documents"])
            state["documents"] = [d for d in state["documents"] if d["doc_id"] != doc_id]
            return len(state["documents"]) != before

    class _HistoryStore:
        def search_sessions(self, user_id: str, query: str):
            sessions = list(state["history"].get(user_id, {}).values())
            q = str(query or "").lower()
            if q:
                sessions = [
                    s for s in sessions
                    if q in s["title"].lower()
                    or any(q in str(message.get("content") or "").lower() for message in s.get("messages", []))
                ]
            return sessions

        def save_session(self, user_id: str, session_id: str, messages: list[dict[str, Any]], title: str | None = None):
            payload = {
                "id": session_id,
                "title": title or "hello session",
                "created_at": 1,
                "updated_at": 1,
                "messages": messages,
            }
            state["history"].setdefault(user_id, {})[session_id] = payload
            return payload

        def get_session_detail(self, user_id: str, session_id: str):
            return state["history"].get(user_id, {}).get(session_id)

        def rename_session(self, user_id: str, session_id: str, title: str):
            session = state["history"].get(user_id, {}).get(session_id)
            if not session:
                return None
            session["title"] = title
            return session

        def delete_session(self, user_id: str, session_id: str):
            state["history"].get(user_id, {}).pop(session_id, None)
            return True

    class _ProfileEngine:
        def get_profile(self, user_id: str):
            return state["profile"]

        def upsert_profile(self, user_id: str, profile: dict[str, Any], version: int):
            state["profile"] = profile

    class _MemoryEngine:
        def replace_profile_semantic_memory(self, *, user_id: str, profile: dict[str, Any]):
            state["profile"] = profile
            return 1

    class _MemoryStore:
        def list_items(self, **kwargs):
            return list(state["memory_items"])

        def get_item(self, *, user_id: str, item_id: int):
            for item in state["memory_items"]:
                if item["item_id"] == item_id:
                    return item
            return None

        def delete_item(self, *, user_id: str, item_id: int):
            before = len(state["memory_items"])
            state["memory_items"] = [i for i in state["memory_items"] if i["item_id"] != item_id]
            return len(state["memory_items"]) != before

        def upsert_items(self, rows: list[dict[str, Any]]):
            next_id = len(state["memory_items"]) + 1
            for row in rows:
                state["memory_items"].append(
                    {
                        "item_id": next_id,
                        "user_id": row["user_id"],
                        "kind": row["kind"],
                        "subkind": row.get("subkind"),
                        "item_hash": row["item_hash"],
                        "text": row["text"],
                    }
                )
                next_id += 1
            return len(rows)

        def get_item_by_hash(self, *, user_id: str, kind: str, item_hash: str):
            for item in state["memory_items"]:
                if (
                    item["user_id"] == user_id
                    and item["kind"] == kind
                    and item["item_hash"] == item_hash
                ):
                    return item
            return None

    class _Embeddings:
        def embed_documents(self, texts: list[str]):
            return [[0.1, 0.2] for _ in texts]

    class _ChatState:
        def __init__(self, values: dict[str, Any]):
            self.values = values

    class _CheckpointStore:
        async def load(self, session_id: str):
            return state["checkpoints"].get(session_id)

        async def save(self, session_id: str, checkpoint: dict[str, Any]):
            state["checkpoints"][session_id] = {
                "checkpoint": checkpoint,
                "updated_at": "t1",
            }
            return state["checkpoints"][session_id]

    class _EventService:
        def record(self, **kwargs):
            state["interrupt_events"].append(kwargs)
            return {"event_id": f"he-{len(state['interrupt_events'])}", **kwargs}

        def list_for_session(self, *, session_id: str, user_id: str | None = None, limit: int = 100):
            events = [
                {"event_id": f"he-{index}", **event}
                for index, event in enumerate(state["interrupt_events"], start=1)
                if event.get("session_id") == session_id
                and (user_id is None or event.get("user_id") == user_id)
            ]
            return events[:limit]

    def _persist_history(
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
    ):
        return _HistoryStore().save_session(user_id, session_id, messages, title)

    class _GraphApp:
        async def ainvoke(self, input_value: dict[str, Any], config: dict[str, Any]):
            context = dict(input_value.get("context") or {})
            session_id = str(context.get("session_id") or config.get("configurable", {}).get("thread_id") or "")
            user_id = str(config.get("configurable", {}).get("user_id") or user.username)
            user_messages = list(input_value.get("messages") or [])
            assistant_content = (
                "Approval pending smoke draft"
                if context.get("require_human_approval")
                else "Workbench smoke reply"
            )
            session_state = {
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    *user_messages,
                    {"role": "assistant", "content": assistant_content},
                ],
                "context": {
                    "context_pruning": {"method": "smoke"},
                    "session_id": session_id,
                    "user_id": user_id,
                },
                "interrupted": bool(context.get("require_human_approval")),
            }
            if context.get("require_human_approval"):
                action_required = {
                    "action_type": str(context.get("interrupt_action_type") or "deploy"),
                    "description": str(context.get("interrupt_description") or "need approval"),
                    "payload": dict(context.get("interrupt_payload") or {"next_step": "generate"}),
                    "requires_approval": True,
                    "approved": False,
                    "approved_by": None,
                    "approved_at": None,
                }
                session_state["action_required"] = action_required
                state["checkpoints"][session_id] = {
                    "checkpoint": {
                        "checkpoint_id": f"cp-{session_id}",
                        "checkpoint_ns": "",
                        "user_id": user_id,
                        "context": {
                            "user_id": user_id,
                            "session_id": session_id,
                        },
                        "interrupted": True,
                        "action_required": action_required,
                        "channel_values": {
                            "user_id": user_id,
                            "context": {
                                "user_id": user_id,
                                "session_id": session_id,
                            },
                            "interrupted": True,
                            "action_required": action_required,
                        },
                    },
                    "updated_at": "t0",
                }
            state["chat_states"][session_id] = session_state
            return {
                "messages": session_state["messages"],
                "context": session_state["context"],
                "interrupted": session_state["interrupted"],
            }

        async def aget_state(self, config: dict[str, Any]):
            session_id = str(config.get("configurable", {}).get("thread_id") or "")
            current = state["chat_states"].get(session_id) or {}
            if current:
                return _ChatState(current)
            return _ChatState({"messages": [], "context": {}, "interrupted": False})

    class _ResumeService:
        async def resume_approved_session(self, *, session_id: str, checkpoint: dict[str, Any]):
            checkpoint_data = dict(checkpoint.get("checkpoint") or {})
            action_required = dict(checkpoint_data.get("action_required") or {})
            if not bool(action_required.get("approved")):
                return {
                    "ok": False,
                    "interrupted": True,
                    "error_message": "approval missing",
                }
            current = dict(state["chat_states"].get(session_id) or {})
            messages = list(current.get("messages") or [])
            approved_reply = "Approved smoke reply"
            messages.append({"role": "assistant", "content": approved_reply})
            context = dict(current.get("context") or {})
            context["context_pruning"] = {"method": "smoke_resume"}
            current.update(
                {
                    "messages": messages,
                    "context": context,
                    "interrupted": False,
                    "action_required": None,
                }
            )
            state["chat_states"][session_id] = current

            checkpoint_data["interrupted"] = False
            checkpoint_data["action_required"] = None
            channel_values = checkpoint_data.get("channel_values")
            if isinstance(channel_values, dict):
                channel_values["interrupted"] = False
                channel_values["action_required"] = None
            state["checkpoints"][session_id] = {
                "checkpoint": checkpoint_data,
                "updated_at": "t2",
            }
            return {
                "ok": True,
                "interrupted": False,
                "reply": approved_reply,
                "messages": messages,
                "context": context,
            }

    monkeypatch.setattr(upload_api.settings.storage_local, "documents_dir", str(docs_root))
    monkeypatch.setattr(upload_api.settings.storage_local, "uploads_dir", str(uploads_root))
    monkeypatch.setattr(upload_api, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(upload_api, "get_task", _get_task)
    monkeypatch.setattr(upload_api, "release_task_operation", _release_task_operation)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    monkeypatch.setattr(documents_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(documents_api, "MySQLDocStore", lambda: _DocStore())

    monkeypatch.setattr(history_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(history_api, "MySQLConversationStore", lambda: _HistoryStore())
    monkeypatch.setattr(
        history_api,
        "persist_session_messages",
        lambda *, user_id, session_id, messages, background_tasks=None, title=None: _persist_history(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            title=title,
        ),
    )
    monkeypatch.setattr(chat_api, "get_chat_graph_app", lambda: _GraphApp())
    monkeypatch.setattr(
        chat_api,
        "persist_session_messages",
        lambda *, user_id, session_id, messages, background_tasks=None, title=None: _persist_history(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            title=title,
        ),
    )
    monkeypatch.setattr(interrupt_api, "checkpoint_store", _CheckpointStore())
    monkeypatch.setattr(interrupt_api, "get_graph_resume_service", lambda: _ResumeService())
    monkeypatch.setattr(interrupt_api, "get_interrupt_event_service", lambda: _EventService())
    monkeypatch.setattr(
        interrupt_api,
        "persist_session_messages",
        lambda *, user_id, session_id, messages, background_tasks=None, title=None: _persist_history(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            title=title,
        ),
    )

    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(memory_api, "PgUserMemoryStore", lambda: _MemoryStore())
    monkeypatch.setattr(memory_api, "UserProfileEngine", lambda: _ProfileEngine())
    monkeypatch.setattr(memory_api, "UserMemoryEngine", lambda: _MemoryEngine())
    monkeypatch.setattr(memory_api, "get_embeddings", lambda: _Embeddings())

    app = FastAPI()
    app.include_router(chat_api.router)
    app.include_router(interrupt_api.router)
    app.include_router(upload_api.router)
    app.include_router(tasks_api.router)
    app.include_router(documents_api.router)
    app.include_router(history_api.router)
    app.include_router(memory_api.router)
    for module in [chat_api, interrupt_api, upload_api, tasks_api, documents_api, history_api, memory_api]:
        app.dependency_overrides[module.get_current_active_user] = lambda u=user: u

    client = TestClient(app)

    upload = client.post(
        "/upload",
        files=[("files", ("guide.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert upload.status_code == 200
    task_id = upload.json()["results"][0]["task_id"]

    task = client.get(f"/tasks/{task_id}")
    assert task.status_code == 200
    assert task.json()["status"] == "succeeded"

    docs = client.get("/documents", params={"q": "guide"})
    assert docs.status_code == 200
    assert len(docs.json()["documents"]) == 1
    doc_id = docs.json()["documents"][0]["doc_id"]

    doc = client.get(f"/documents/{doc_id}")
    assert doc.status_code == 200
    assert doc.json()["download_url"] == f"/documents/{doc_id}/download"
    assert doc.json()["preview"][0]["content"] == "hello doc"

    download = client.get(f"/documents/{doc_id}/download")
    assert download.status_code == 200
    assert download.content == b"%PDF-1.4"

    chat = client.post(
        "/chat/workbench-invoke",
        json={
            "input": {
                "messages": [{"role": "user", "content": "question about guide"}],
                "context": {"session_id": "s1", "context_focus_hint": "focus on upload smoke"},
            },
            "config": {"configurable": {"thread_id": "s1"}},
        },
    )
    assert chat.status_code == 200
    assert chat.json()["reply"] == "Workbench smoke reply"
    assert chat.json()["messages"][-1]["role"] == "assistant"

    history = client.get(f"/history/{user.username}", params={"q": "guide"})
    assert history.status_code == 200
    assert history.json()["history"][0]["id"] == "s1"
    assert history.json()["history"][0]["messages"][-1]["content"] == "Workbench smoke reply"

    interrupt_chat = client.post(
        "/chat/workbench-invoke",
        json={
            "input": {
                "messages": [{"role": "user", "content": "deploy the guide changes"}],
                "context": {
                    "session_id": "approve-1",
                    "require_human_approval": True,
                    "interrupt_action_type": "deploy",
                    "interrupt_description": "approve guide deploy",
                    "interrupt_payload": {"next_step": "generate"},
                },
            },
            "config": {"configurable": {"thread_id": "approve-1"}},
        },
    )
    assert interrupt_chat.status_code == 200
    assert interrupt_chat.json()["interrupted"] is True
    assert interrupt_chat.json()["reply"] == "Approval pending smoke draft"
    assert interrupt_chat.json()["messages"][-1]["content"] == "Approval pending smoke draft"
    assert interrupt_chat.json()["context"]["context_pruning"]["method"] == "smoke"

    interrupt_status = client.get("/interrupt/approve-1")
    assert interrupt_status.status_code == 200
    assert interrupt_status.json()["interrupted"] is True
    assert interrupt_status.json()["action_required"]["approved"] is False

    approve_interrupt = client.post("/interrupt/approve-1/approve", json={"approved": True})
    assert approve_interrupt.status_code == 200
    assert approve_interrupt.json()["approved"] is True

    resume_interrupt = client.post("/interrupt/approve-1/resume")
    assert resume_interrupt.status_code == 200
    assert resume_interrupt.json()["resumed"] is True
    assert resume_interrupt.json()["reply"] == "Approved smoke reply"
    assert resume_interrupt.json()["messages"][-1]["content"] == "Approved smoke reply"
    assert resume_interrupt.json()["context"]["context_pruning"]["method"] == "smoke_resume"

    interrupt_status = client.get("/interrupt/approve-1")
    assert interrupt_status.status_code == 200
    assert interrupt_status.json()["interrupted"] is False
    assert interrupt_status.json()["action_required"] is None

    approve_events = client.get("/interrupt/approve-1/events")
    assert approve_events.status_code == 200
    assert [event["event_type"] for event in approve_events.json()["events"]] == [
        "interrupt.approved",
        "interrupt.resume_requested",
        "interrupt.resumed",
    ]

    resumed_history = client.get(f"/history/{user.username}/approve-1")
    assert resumed_history.status_code == 200
    assert resumed_history.json()["messages"][-1]["content"] == "Approved smoke reply"

    profile = client.get("/memory/profile")
    assert profile.status_code == 200
    assert profile.json()["profile"]["facts"][0]["text"] == "prefers concise answers"

    create_memory = client.post(
        "/memory/items",
        json={"kind": "semantic", "subkind": "manual_note", "text": "remember this"},
    )
    assert create_memory.status_code == 200
    assert create_memory.json()["item"]["item_id"] == 1

    memory_items = client.get("/memory/items")
    assert memory_items.status_code == 200
    assert len(memory_items.json()["items"]) == 1
