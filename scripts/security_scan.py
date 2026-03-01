from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: str
    exit_code: int | None
    summary: dict[str, Any]
    raw: Any | None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def _bandit() -> ToolResult:
    if shutil.which("bandit") is None:
        return ToolResult("bandit", "skipped", None, {"reason": "bandit not installed"}, None)
    proc = _run(["bandit", "-r", "app", "-f", "json", "-q"])
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
    if shutil.which("pip-audit") is None:
        return ToolResult("pip-audit", "skipped", None, {"reason": "pip-audit not installed"}, None)
    proc = _run(["pip-audit", "-r", "requirements.txt", "-f", "json"])
    raw: Any | None = None
    try:
        raw = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError:
        raw = None
    vulns = raw if isinstance(raw, list) else []
    return ToolResult(
        "pip-audit",
        "ok" if proc.returncode == 0 and len(vulns) == 0 else "issues",
        proc.returncode,
        {"total": len(vulns)},
        raw,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bandit_res = _bandit()
    pip_audit_res = _pip_audit()

    bandit_high = int(bandit_res.summary.get("by_severity", {}).get("HIGH", 0))
    pip_audit_total = int(pip_audit_res.summary.get("total", 0))

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
            "pass": bandit_high == 0 and pip_audit_total == 0,
            "bandit_high": bandit_high,
            "pip_audit_total": pip_audit_total,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0 if payload["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

