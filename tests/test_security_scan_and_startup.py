from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_security_scan_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "security_scan.py"
    module_name = "security_scan_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load security_scan module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_security_scan_missing_tool_fails_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    security_scan = _load_security_scan_module()
    out = tmp_path / "security.json"

    monkeypatch.setattr(
        security_scan.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(out=str(out)),
    )
    monkeypatch.setattr(
        security_scan,
        "_bandit",
        lambda: security_scan.ToolResult("bandit", "missing", None, {"reason": "bandit not installed"}, None),
    )
    monkeypatch.setattr(
        security_scan,
        "_pip_audit",
        lambda: security_scan.ToolResult("pip-audit", "ok", 0, {"total": 0}, []),
    )

    code = security_scan.main()
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert code == 2
    assert payload["gate"]["pass"] is False
    assert payload["gate"]["tools_ready"] is False
    assert payload["gate"]["missing_tools"] == ["bandit"]


def test_lifespan_contains_validate_security_call():
    main_path = Path(__file__).resolve().parent.parent / "app" / "server" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))

    lifespan_node = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    validate_calls = [
        node
        for node in ast.walk(lifespan_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "settings"
        and node.func.attr == "validate_security"
    ]

    assert len(validate_calls) > 0
