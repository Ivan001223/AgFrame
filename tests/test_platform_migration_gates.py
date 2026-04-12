from pathlib import Path


def test_stage_0_inventory_documents_authoritative_write_paths():
    content = Path("docs/architecture/platform-domain-stage-0-inventory.md").read_text(encoding="utf-8")

    assert "app/harness/runtime/run_service.py" in content
    assert "app/infrastructure/queue/arq_jobs.py" in content
    assert "authoritative write path" in content.lower()
