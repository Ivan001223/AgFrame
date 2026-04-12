from pathlib import Path


def test_governance_control_plane_doc_names_single_authoritative_write_path():
    content = Path("docs/architecture/platform-governance-control-plane.md").read_text(encoding="utf-8")

    assert "single authoritative write path" in content.lower()
    assert "app/platform/governance/service.py" in content
