from app.platform.contracts.translators import (
    approval_record_to_legacy_payload,
    legacy_run_to_run_envelope,
    legacy_verification_to_record,
)


def test_legacy_run_to_run_envelope_maps_harness_shape():
    envelope = legacy_run_to_run_envelope(
        {
            "run_id": "hr-1",
            "task_type": "document_ingest",
            "status": "queued",
            "input_json": {"file_path": "/tmp/a.pdf"},
            "metadata_json": {"source": "manual"},
        }
    )

    assert envelope.version == "run.v1"
    assert envelope.lifecycle_status.value == "queued"
    assert envelope.input["file_path"] == "/tmp/a.pdf"


def test_approval_record_to_legacy_payload_preserves_current_api_shape():
    payload = approval_record_to_legacy_payload(
        {
            "approval_id": "ha-1",
            "target_run_id": "hr-1",
            "decision_state": "pending",
            "requested_decision": "approve",
        }
    )

    assert payload["approval_id"] == "ha-1"
    assert payload["run_id"] == "hr-1"
    assert payload["status"] == "pending"


def test_legacy_verification_to_record_maps_artifacts_and_summary():
    record = legacy_verification_to_record(
        {
            "verification_id": "hv-1",
            "run_id": "hr-1",
            "status": "pass",
            "artifacts_json": {"session_id": "s1"},
            "summary": "resume ok",
        }
    )

    assert record.version == "verification.v1"
    assert record.subject_run_id == "hr-1"
    assert record.evidence == {"session_id": "s1"}
    assert record.summary == "resume ok"
