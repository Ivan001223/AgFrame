from app.harness.runtime.run_service import HarnessRunService


def test_create_document_ingest_run():
    created = {}

    class _Store:
        def create_run(self, **kwargs):
            created.update(kwargs)
            return {**kwargs, "created_at": 1, "updated_at": 1}

    service = HarnessRunService(run_store=_Store())
    run = service.create_run(
        user_id="u1",
        task_type="document_ingest",
        input_json={"file_path": "/tmp/a.pdf"},
        session_id=None,
        metadata_json=None,
    )

    assert run["task_type"] == "document_ingest"
    assert run["policy_id"] == "document_ingest:v1"
    assert run["status"] == "created"
    assert created["approval_required"] is False


def test_create_approval_required_run_creates_pending_approval():
    created = {}
    approvals = []

    class _RunStore:
        def create_run(self, **kwargs):
            created.update(kwargs)
            return {**kwargs, "created_at": 1, "updated_at": 1}

    class _ApprovalStore:
        def create_approval(self, **kwargs):
            approvals.append(kwargs)
            return {**kwargs, "created_at": 1}

    service = HarnessRunService(run_store=_RunStore(), approval_store=_ApprovalStore())
    run = service.create_run(
        user_id="u1",
        task_type="session_resume_approval",
        input_json={"session_id": "s1"},
        session_id="s1",
        metadata_json=None,
    )

    assert run["task_type"] == "session_resume_approval"
    assert run["status"] == "waiting_approval"
    assert created["approval_required"] is True
    assert approvals[0]["run_id"] == run["run_id"]
    assert approvals[0]["status"] == "pending"
    assert approvals[0]["payload_json"]["session_id"] == "s1"
