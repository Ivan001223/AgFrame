from app.harness.persistence.stores import HarnessAgentProjectStore, HarnessApprovalStore, HarnessRunStore, HarnessVerificationStore


def test_run_store_builds_run_record(monkeypatch):
    created = {}

    class _Session:
        def add(self, obj):
            created["obj"] = obj

        def flush(self):
            return None

    class _Ctx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.harness.persistence.stores.get_session", lambda: _Ctx())

    store = HarnessRunStore()
    run = store.create_run(
        run_id="hr-1",
        user_id="u1",
        session_id=None,
        task_type="document_ingest",
        status="queued",
        policy_id="document_ingest:v1",
        input_json={"file_path": "/tmp/a.pdf"},
        metadata_json=None,
        approval_required=False,
    )

    assert run["run_id"] == "hr-1"
    assert run["task_type"] == "document_ingest"
    assert created["obj"].run_id == "hr-1"


def test_approval_store_builds_record(monkeypatch):
    created = {}

    class _Session:
        def add(self, obj):
            created["obj"] = obj

        def flush(self):
            return None

    class _Ctx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.harness.persistence.stores.get_session", lambda: _Ctx())

    store = HarnessApprovalStore()
    approval = store.create_approval(
        approval_id="ha-1",
        run_id="hr-1",
        action_type="resume",
        reason="needs approval",
        payload_json={"session_id": "s1"},
        status="pending",
        requested_by="u1",
    )

    assert approval["approval_id"] == "ha-1"
    assert approval["run_id"] == "hr-1"
    assert created["obj"].status == "pending"


def test_agent_project_store_builds_record(monkeypatch):
    created = {}

    class _Session:
        def add(self, obj):
            created["obj"] = obj

        def flush(self):
            return None

    class _Ctx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.harness.persistence.stores.get_session", lambda: _Ctx())

    store = HarnessAgentProjectStore()
    project = store.create_project(
        project_id="hp-1",
        user_id="u1",
        name="Studio",
        description="Canvas config",
        graph_json={"agents": [], "edges": []},
    )

    assert project["project_id"] == "hp-1"
    assert project["user_id"] == "u1"
    assert created["obj"].name == "Studio"


def test_verification_store_builds_record(monkeypatch):
    created = {}

    class _Session:
        def add(self, obj):
            created["obj"] = obj

        def flush(self):
            return None

    class _Ctx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.harness.persistence.stores.get_session", lambda: _Ctx())

    store = HarnessVerificationStore()
    verification = store.create_verification(
        verification_id="hv-1",
        run_id="hr-1",
        status="pass",
        checks_json={"checks_run": ["document_ingest_result"]},
        artifacts_json={"stage": "done"},
        summary="ok",
    )

    assert verification["verification_id"] == "hv-1"
    assert verification["run_id"] == "hr-1"
    assert created["obj"].status == "pass"
