from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: str
    exit_code: int | None
    summary: dict[str, Any]
    raw: Any | None


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def _audit_timeout_seconds(default: int = 90) -> int:
    try:
        value = int(os.getenv("SECURITY_SCAN_TIMEOUT", str(default)))
    except ValueError:
        return default
    return max(30, value)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_tool(tool_name: str) -> list[str] | None:
    direct = shutil.which(tool_name)
    if direct:
        return [direct]

    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [scripts_dir / f"{tool_name}.exe", scripts_dir / tool_name]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]

    if tool_name == "pip-audit":
        return [sys.executable, "-m", "pip_audit"]
    if tool_name == "bandit":
        return [sys.executable, "-m", "bandit"]
    return None


def _count_pip_audit_vulns(raw: Any) -> int:
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, dict):
        dependencies = raw.get("dependencies", [])
        if isinstance(dependencies, list):
            total = 0
            for dep in dependencies:
                if not isinstance(dep, dict):
                    continue
                vulns = dep.get("vulns", [])
                if isinstance(vulns, list):
                    total += len(vulns)
            return total
    return 0


def _bandit() -> ToolResult:
    bandit_cmd = _resolve_tool("bandit")
    if bandit_cmd is None:
        return ToolResult("bandit", "missing", None, {"reason": "bandit not installed"}, None)
    proc = _run([*bandit_cmd, "-r", "app", "-f", "json", "-q"])
    raw: Any | None = None
    try:
        raw = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError:
        raw = None
    issues = (raw or {}).get("results", []) if isinstance(raw, dict) else []
    counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for it in issues:
        sev = str(it.get("issue_severity") or "").upper()
        if sev in counts:
            counts[sev] += 1
    return ToolResult(
        "bandit",
        "ok" if proc.returncode == 0 else "issues",
        proc.returncode,
        {"total": len(issues), "by_severity": counts},
        raw,
    )


def _pip_audit() -> ToolResult:
    pip_audit_cmd = _resolve_tool("pip-audit")
    if pip_audit_cmd is None:
        return ToolResult("pip-audit", "missing", None, {"reason": "pip-audit not installed"}, None)
    cache_dir = _project_root() / ".pip-audit-cache"
    cache_dir.mkdir(exist_ok=True)
    timeout_seconds = _audit_timeout_seconds()
    attempts: list[tuple[str, list[str], int]] = [
        (
            "local",
            [*pip_audit_cmd, "--local", "--progress-spinner", "off", "--cache-dir", str(cache_dir), "-f", "json"],
            timeout_seconds,
        )
    ]
    export_path: str | None = None

    if shutil.which("uv") is not None and os.path.exists("pyproject.toml"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            export_path = tmp.name
        export_proc = _run(
            ["uv", "export", "--format", "requirements-txt", "--no-hashes", "-o", export_path],
            timeout=60,
        )
        if export_proc.returncode == 0 and os.path.exists(export_path):
            attempts.insert(
                0,
                (
                    "uv-export",
                    [*pip_audit_cmd, "-r", export_path, "--progress-spinner", "off", "--cache-dir", str(cache_dir), "-f", "json"],
                    timeout_seconds,
                ),
            )

    last_proc: subprocess.CompletedProcess[str] | None = None
    try:
        for mode, cmd, timeout in attempts:
            try:
                proc = _run(cmd, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                last_proc = subprocess.CompletedProcess(
                    cmd,
                    returncode=124,
                    stdout="",
                    stderr=f"timeout after {timeout}s\n{stderr}".strip(),
                )
                continue
            last_proc = proc
            raw: Any | None = None
            try:
                raw = json.loads(proc.stdout) if proc.stdout.strip() else None
            except json.JSONDecodeError:
                raw = None

            if raw is not None:
                vuln_total = _count_pip_audit_vulns(raw)
                return ToolResult(
                    "pip-audit",
                    "ok" if proc.returncode == 0 and vuln_total == 0 else "issues",
                    proc.returncode,
                    {"total": vuln_total, "mode": mode},
                    raw,
                )
    finally:
        if export_path and os.path.exists(export_path):
            os.unlink(export_path)

    return ToolResult(
        "pip-audit",
        "error",
        last_proc.returncode if last_proc is not None else None,
        {
            "reason": "pip-audit produced no parseable JSON output",
            "mode": "requirements_then_local",
            "stderr": (last_proc.stderr or "")[:2000] if last_proc is not None else "",
        },
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bandit_res = _bandit()
    pip_audit_res = _pip_audit()

    bandit_high = int(bandit_res.summary.get("by_severity", {}).get("HIGH", 0))
    pip_audit_total = int(pip_audit_res.summary.get("total", 0))
    missing_tools = [res.tool for res in (bandit_res, pip_audit_res) if res.status == "missing"]
    tools_ready = len(missing_tools) == 0
    bandit_ready = bandit_res.status == "ok"
    pip_audit_ready = pip_audit_res.status == "ok"
    gate_pass = tools_ready and bandit_ready and pip_audit_ready and bandit_high == 0 and pip_audit_total == 0

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "bandit": {
            "status": bandit_res.status,
            "exit_code": bandit_res.exit_code,
            "summary": bandit_res.summary,
            "raw": bandit_res.raw,
        },
        "pip_audit": {
            "status": pip_audit_res.status,
            "exit_code": pip_audit_res.exit_code,
            "summary": pip_audit_res.summary,
            "raw": pip_audit_res.raw,
        },
        "gate": {
            "pass": gate_pass,
            "tools_ready": tools_ready,
            "bandit_ready": bandit_ready,
            "pip_audit_ready": pip_audit_ready,
            "missing_tools": missing_tools,
            "bandit_high": bandit_high,
            "pip_audit_total": pip_audit_total,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0 if payload["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
