import pytest

from app.harness.runtime.approval_service import ApprovalService


@pytest.mark.anyio
async def test_approval_service_approve_updates_checkpoint_and_enqueues(monkeypatch):
    calls = {"saved": None, "enqueued": None, "approval": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return {
                "checkpoint": {
                    "interrupted": True,
                    "action_required": {
                        "action_type": "resume",
                        "approved": False,
                    },
                }
            }

        async def save(self, session_id: str, checkpoint: dict[str, object]):
            calls["saved"] = (session_id, checkpoint)

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {"approval_id": "ha-1", "run_id": run_id, "status": "pending"}

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            calls["approval"] = (approval_id, status, resolved_by, comment)
            return {"approval_id": approval_id, "status": status}

        def get_run(self, run_id: str):
            return {"run_id": run_id, "session_id": "s1"}

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

    async def _enqueue(run_id: str):
        calls["enqueued"] = run_id
        return "job-1"

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", True, "u1", "ok")

    assert result["status"] == "approved"
    assert calls["approval"] == ("ha-1", "approved", "u1", "ok")
    assert calls["saved"][0] == "s1"
    assert calls["saved"][1]["action_required"]["approved"] is True
    assert calls["enqueued"] == "hr-1"
    assert ("approved", "hr-1") in calls["marks"]
    assert ("persist", "hr-1", True) in calls["marks"]


@pytest.mark.anyio
async def test_approval_service_reject_does_not_enqueue(monkeypatch):
    calls = {"saved": None, "enqueued": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return {
                "checkpoint": {
                    "interrupted": True,
                    "action_required": {"approved": False},
                }
            }

        async def save(self, session_id: str, checkpoint: dict[str, object]):
            calls["saved"] = (session_id, checkpoint)

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {"approval_id": "ha-1", "run_id": run_id, "status": "pending"}

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            return {"approval_id": approval_id, "status": status}

        def get_run(self, run_id: str):
            return {"run_id": run_id, "session_id": "s1"}

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

    async def _enqueue(run_id: str):
        calls["enqueued"] = run_id
        return "job-1"

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", False, "u1", "stop")

    assert result["status"] == "rejected"
    assert calls["saved"][1]["action_required"]["approved"] is False
    assert calls["enqueued"] is None
    assert ("rejected", "hr-1") in calls["marks"]
    assert ("persist", "hr-1", False) in calls["marks"]


@pytest.mark.anyio
async def test_approval_service_approve_orchestration_review_updates_resume_state(monkeypatch):
    calls = {"updated_input": None, "enqueued": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return None

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {"approval_id": "ha-1", "run_id": run_id, "status": "pending", "action_type": "orchestration_review"}

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            return {"approval_id": approval_id, "status": status, "action_type": "orchestration_review"}

        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "session_id": None,
                "input_json": {"orchestration_resume": {"next_step_index": 2, "state": {"task": "t"}}},
            }

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            calls["updated_input"] = input_json

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

    async def _enqueue(run_id: str):
        calls["enqueued"] = run_id
        return "job-1"

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", True, "u1", "continue")

    assert result["status"] == "approved"
    assert calls["updated_input"]["orchestration_resume"]["review_decision"] == "approved"
    assert calls["enqueued"] == "hr-1"


@pytest.mark.anyio
async def test_approval_service_approve_cluster_research_sets_continue_with_research(monkeypatch):
    calls = {"updated_input": None, "updated_metadata": None, "enqueued": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return None

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "pending",
                "action_type": "orchestration_review",
                "payload_json": {"review_stage": "cluster_research"},
            }

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            return {"approval_id": approval_id, "status": status, "action_type": "orchestration_review"}

        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "session_id": None,
                "metadata_json": {"source": "studio"},
                "input_json": {"orchestration_resume": {"next_step_index": 2, "state": {"task": "t"}}},
            }

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            calls["updated_input"] = input_json

        def update_run_metadata_json(self, run_id: str, metadata_json: dict[str, object]):
            calls["updated_metadata"] = metadata_json

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

    async def _enqueue(run_id: str):
        calls["enqueued"] = run_id
        return "job-1"

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", True, "u1", "accept evidence")

    assert result["status"] == "approved"
    assert calls["updated_input"]["orchestration_resume"]["continue_mode"] == "accept_research_evidence"
    assert calls["updated_metadata"]["review_recovery_mode"] == "continue_with_research"
    assert calls["enqueued"] == "hr-1"


@pytest.mark.anyio
async def test_approval_service_approve_stream_review_sets_accept_partial_stream_output(monkeypatch):
    calls = {"updated_input": None, "updated_metadata": None, "enqueued": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return None

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "pending",
                "action_type": "orchestration_review",
                "payload_json": {"review_stage": "agent_output_stream", "partial_output": "safe partial"},
            }

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            return {"approval_id": approval_id, "status": status, "action_type": "orchestration_review"}

        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "session_id": None,
                "metadata_json": {"source": "studio"},
                "input_json": {"orchestration_resume": {"next_step_index": 2, "state": {"task": "t"}}},
            }

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            calls["updated_input"] = input_json

        def update_run_metadata_json(self, run_id: str, metadata_json: dict[str, object]):
            calls["updated_metadata"] = metadata_json

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

    async def _enqueue(run_id: str):
        calls["enqueued"] = run_id
        return "job-1"

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", True, "u1", "accept partial")

    assert result["status"] == "approved"
    assert calls["updated_input"]["orchestration_resume"]["continue_mode"] == "accept_partial_stream_output"
    assert calls["updated_metadata"]["review_recovery_mode"] == "continue_with_partial_stream_output"
    assert calls["enqueued"] == "hr-1"


@pytest.mark.anyio
async def test_approval_service_reject_cluster_research_sets_discard_mode(monkeypatch):
    calls = {"updated_input": None, "updated_metadata": None, "marks": []}

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return None

    class _RunService:
        def get_pending_approval_for_run(self, run_id: str):
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "pending",
                "action_type": "orchestration_review",
                "payload_json": {"review_stage": "cluster_research"},
            }

        def update_approval(self, approval_id: str, *, status: str, resolved_by: str, comment: str | None):
            return {"approval_id": approval_id, "status": status, "action_type": "orchestration_review"}

        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "session_id": None,
                "input_json": {
                    "orchestration_resume": {
                        "next_step_index": 3,
                        "state": {"task": "t"},
                        "rollback_state": {"task": "t", "agent_outputs": {"cluster_a": "safe"}},
                    }
                },
            }

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            calls["updated_input"] = input_json

        def update_run_metadata_json(self, run_id: str, metadata_json: dict[str, object]):
            calls["updated_metadata"] = metadata_json

        def mark_rejected(self, run_id: str):
            calls["marks"].append(("rejected", run_id))

        def persist_approval_resolution(self, run_id: str, *, approved: bool):
            calls["marks"].append(("persist", run_id, approved))

        def mark_approved(self, run_id: str):
            calls["marks"].append(("approved", run_id))

    async def _enqueue(run_id: str):
        raise AssertionError("should not enqueue on reject")

    service = ApprovalService(checkpoint_adapter=_CheckpointAdapter())
    service.run_service = _RunService()
    monkeypatch.setattr("app.harness.runtime.approval_service.enqueue_harness_resume", _enqueue)

    result = await service.resolve("hr-1", False, "u1", "discard research")

    assert result["status"] == "rejected"
    assert calls["updated_input"]["orchestration_resume"]["continue_mode"] == "discard_research_evidence"
    assert calls["updated_input"]["orchestration_resume"]["next_step_index"] == 3
    assert calls["updated_metadata"]["review_recovery_mode"] == "continue_without_research"
