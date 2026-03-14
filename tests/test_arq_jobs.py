from __future__ import annotations

import pytest

from app.infrastructure.queue import arq_jobs


@pytest.mark.anyio
async def test_ingest_pdf_persists_structured_failure(monkeypatch: pytest.MonkeyPatch):
    updates: list[dict[str, str | int]] = []

    async def _get_task(task_id: str) -> dict[str, str]:
        return {"task_id": task_id, "operation_key": "op-1"}

    async def _update_task(task_id: str, payload: dict[str, str | int]) -> None:
        updates.append(payload)

    async def _release_task_operation(operation_key: str, expected_task_id: str | None = None) -> None:
        return None

    async def _append_task_incident(payload: dict[str, str | int], **kwargs):
        return None

    class _Rag:
        def add_knowledge_base(self, file_path: str, user_id: str | None = None):
            return {
                "ok": False,
                "error_code": "embedding_failed",
                "error_message": "embedding timeout",
                "stage": "embedding",
            }

    monkeypatch.setattr(arq_jobs, "get_task", _get_task)
    monkeypatch.setattr(arq_jobs, "update_task", _update_task)
    monkeypatch.setattr(arq_jobs, "release_task_operation", _release_task_operation)
    monkeypatch.setattr(arq_jobs, "append_task_incident", _append_task_incident)
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())

    ok = await arq_jobs.ingest_pdf({}, "task-1", "/tmp/a.pdf", user_id="u1")

    assert ok is False
    assert updates[-1]["status"] == "failed"
    assert updates[-1]["error"] == "embedding timeout"
    assert updates[-1]["error_code"] == "embedding_failed"
    assert updates[-1]["result_stage"] == "embedding"
