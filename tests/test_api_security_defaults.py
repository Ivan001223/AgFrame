from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.server.cors_policy import build_cors_options


def test_server_config_source_has_secure_cors_defaults():
    settings_path = Path(__file__).resolve().parent.parent / "app" / "infrastructure" / "config" / "settings.py"
    content = settings_path.read_text(encoding="utf-8")
    assert "cors_origins: list[str] = Field(default_factory=list, alias=\"CORS_ORIGINS\")" in content
    assert "cors_allow_credentials: bool = False" in content


def test_cors_rejects_wildcard_with_credentials():
    with pytest.raises(ValueError):
        build_cors_options(cors_origins=["*"], cors_allow_credentials=True)


def test_cors_allows_explicit_origins():
    options = build_cors_options(
        cors_origins=[" https://a.example.com ", "https://a.example.com", "https://b.example.com"],
        cors_allow_credentials=True,
    )
    assert options["allow_origins"] == ["https://a.example.com", "https://b.example.com"]
    assert options["allow_credentials"] is True


def test_error_handler_source_uses_sanitized_messages():
    handler_path = Path(__file__).resolve().parent.parent / "app" / "server" / "error_handlers.py"
    content = handler_path.read_text(encoding="utf-8")
    assert "Internal server error" in content
    assert "Request validation failed" in content
    assert "str(exc)" not in content


def test_main_source_registers_unified_exception_handlers():
    main_path = Path(__file__).resolve().parent.parent / "app" / "server" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_exception_handlers"
    ]
    assert len(calls) == 1
