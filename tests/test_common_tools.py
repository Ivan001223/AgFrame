from __future__ import annotations

from pathlib import Path

from app.skills.common import tools


def test_tools_module_registers_write_file():
    tool_names = [getattr(tool, "name", "") for tool in tools.ALL_TOOLS]

    assert "write_file" in tool_names


def test_write_file_tool_respects_flag_and_writes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tools.settings.feature_flags, "enable_tools_write_file", True)
    monkeypatch.setattr(tools.settings.storage_local, "data_dir", str(tmp_path))
    monkeypatch.setattr(tools.settings.storage_local, "documents_dir", str(tmp_path / "documents"))
    monkeypatch.setattr(tools.settings.storage_local, "uploads_dir", str(tmp_path / "uploads"))
    target = "notes/result.txt"

    result = tools.write_file.invoke({"file_path": str(target), "content": "hello"})

    expected = tmp_path / "notes" / "result.txt"
    assert expected.read_text(encoding="utf-8") == "hello"
    assert str(expected) in result


def test_write_file_tool_blocks_paths_outside_allowed_roots(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tools.settings.feature_flags, "enable_tools_write_file", True)
    monkeypatch.setattr(tools.settings.storage_local, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(tools.settings.storage_local, "documents_dir", str(tmp_path / "documents"))
    monkeypatch.setattr(tools.settings.storage_local, "uploads_dir", str(tmp_path / "uploads"))

    result = tools.write_file.invoke(
        {"file_path": str(tmp_path.parent / "escape.txt"), "content": "hello"}
    )

    assert result.startswith("File write denied:")


def test_read_document_uses_shared_document_extraction(monkeypatch, tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(tools, "extract_text_from_file", lambda path: "normalized text")

    result = tools.read_document.invoke({"file_path": str(target)})

    assert result == "normalized text"
