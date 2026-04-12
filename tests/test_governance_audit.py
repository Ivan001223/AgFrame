from app.platform.governance.audit import build_lifecycle_event_details


def test_lifecycle_event_details_capture_audit_metadata():
    details = build_lifecycle_event_details(
        run_id="hr-1",
        actor="u1",
        contract_version="run.v1",
        from_status="queued",
        to_status="running",
        triggered_by="api",
    )

    assert details["run_id"] == "hr-1"
    assert details["contract_version"] == "run.v1"
    assert details["triggered_by"] == "api"
