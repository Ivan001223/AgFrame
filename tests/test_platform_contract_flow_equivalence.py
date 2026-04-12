import json
from pathlib import Path

from app.platform.contracts.translators import legacy_run_to_run_envelope


def test_create_and_query_run_cases_map_to_canonical_contracts():
    fixture_path = Path("tests/fixtures/platform_contract_cases.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    created_case = cases["create_run"]
    run_envelope = legacy_run_to_run_envelope(created_case["legacy_run"])

    assert run_envelope.lifecycle_status.value == created_case["expected_status"]
    assert run_envelope.task_type == created_case["legacy_run"]["task_type"]
