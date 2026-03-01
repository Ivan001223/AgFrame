from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CoverageSummary:
    line_rate: float
    branch_rate: float
    files: dict[str, float]


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_coverage_xml(path: str) -> CoverageSummary:
    tree = ET.parse(path)
    root = tree.getroot()
    line_rate = float(root.attrib.get("line-rate") or 0.0)
    branch_rate = float(root.attrib.get("branch-rate") or 0.0)
    files: dict[str, float] = {}
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename")
        if not filename:
            continue
        rate = float(cls.attrib.get("line-rate") or 0.0)
        files[filename] = rate
    return CoverageSummary(line_rate=line_rate, branch_rate=branch_rate, files=files)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _key_path_rates(cov: CoverageSummary) -> dict[str, float | None]:
    targets = {
        "app/runtime/prompts/prompt_builder.py": None,
        "app/infrastructure/utils/security.py": None,
        "app/server/api/auth.py": None,
        "app/server/api/upload.py": None,
    }
    for k in list(targets.keys()):
        for filename, rate in cov.files.items():
            if filename.endswith(k):
                targets[k] = rate
                break
    return targets


def _pytest_summary(pytest_json: dict[str, Any]) -> dict[str, int]:
    s = pytest_json.get("summary", {}) if isinstance(pytest_json, dict) else {}
    return {
        "total": int(s.get("total", 0)),
        "passed": int(s.get("passed", 0)),
        "failed": int(s.get("failed", 0)),
        "skipped": int(s.get("skipped", 0)),
        "xfailed": int(s.get("xfailed", 0)),
        "xpassed": int(s.get("xpassed", 0)),
        "errors": int(s.get("errors", 0)),
    }


def _pytest_failures(pytest_json: dict[str, Any]) -> list[dict[str, str]]:
    tests = pytest_json.get("tests", []) if isinstance(pytest_json, dict) else []
    out: list[dict[str, str]] = []
    for t in tests:
        if t.get("outcome") not in {"failed", "error"}:
            continue
        nodeid = str(t.get("nodeid") or "")
        longrepr = ""
        call = t.get("call") or {}
        if isinstance(call, dict):
            longrepr = str(call.get("longrepr") or "")
        out.append({"nodeid": nodeid, "longrepr": longrepr})
    return out


def _write_defects(
    *,
    path: str,
    failures: list[dict[str, str]],
    security: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# 缺陷清单")
    lines.append("")
    idx = 1
    for f in failures:
        lines.append(f"## DEF-{idx:03d} 测试失败：{f['nodeid']}")
        lines.append("")
        lines.append("- 严重级别：P1")
        lines.append("- 范围：测试")
        lines.append("- 复现：运行对应用例")
        lines.append("- 期望：用例通过")
        lines.append("- 实际：用例失败")
        if f["longrepr"].strip():
            lines.append("")
            lines.append("```")
            lines.append(f["longrepr"][:4000])
            lines.append("```")
        lines.append("")
        idx += 1

    gate = (security or {}).get("gate", {}) if isinstance(security, dict) else {}
    if not gate.get("pass", True):
        lines.append(f"## DEF-{idx:03d} 安全门禁未通过")
        lines.append("")
        lines.append("- 严重级别：P0")
        lines.append("- 范围：安全扫描")
        lines.append("- 复现：运行 scripts/security_scan.py")
        lines.append("- 期望：无高危问题与已知依赖漏洞")
        lines.append("- 实际：存在安全问题")
        lines.append("")
        idx += 1

    if idx == 1:
        lines.append("- 无")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-json", required=True)
    parser.add_argument("--coverage-xml", required=True)
    parser.add_argument("--perf-json", required=True)
    parser.add_argument("--security-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--defects", required=True)
    args = parser.parse_args()

    pytest_json = _load_json(args.pytest_json)
    cov = _load_coverage_xml(args.coverage_xml)
    perf = _load_json(args.perf_json)
    security = _load_json(args.security_json)

    summary = _pytest_summary(pytest_json)
    failures = _pytest_failures(pytest_json)
    key_rates = _key_path_rates(cov)

    cov_gate = cov.line_rate >= 0.80
    key_gate = all(v == 1.0 for v in key_rates.values() if v is not None)
    test_gate = summary["failed"] == 0 and summary["errors"] == 0
    sec_gate = bool((security or {}).get("gate", {}).get("pass", True))
    overall_pass = cov_gate and key_gate and test_gate and sec_gate

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.defects), exist_ok=True)
    _write_defects(path=args.defects, failures=failures, security=security)

    lines: list[str] = []
    lines.append("# 测试报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 通过判定：{'PASS' if overall_pass else 'FAIL'}")
    lines.append("")

    lines.append("## 执行摘要")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 用例总数 | {summary['total']} |")
    lines.append(f"| 通过 | {summary['passed']} |")
    lines.append(f"| 失败 | {summary['failed']} |")
    lines.append(f"| 错误 | {summary['errors']} |")
    lines.append(f"| 跳过 | {summary['skipped']} |")
    lines.append("")

    lines.append("## 覆盖率")
    lines.append("")
    lines.append("| 指标 | 值 | 门禁 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 总体行覆盖率 | {_pct(cov.line_rate)} | {'PASS' if cov_gate else 'FAIL'} |")
    lines.append(f"| 总体分支覆盖率 | {_pct(cov.branch_rate)} | - |")
    lines.append("")
    lines.append("### 关键路径（100% 行覆盖）")
    lines.append("")
    lines.append("| 文件 | 行覆盖率 | 判定 |")
    lines.append("|---|---:|---:|")
    for path, rate in key_rates.items():
        if rate is None:
            lines.append(f"| {path} | N/A | FAIL |")
        else:
            lines.append(f"| {path} | {_pct(rate)} | {'PASS' if rate == 1.0 else 'FAIL'} |")
    lines.append("")

    lines.append("## 性能基准")
    lines.append("")
    lines.append("| 场景 | p50(ms) | p95(ms) | mean(ms) | runs |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in (perf or {}).get("results", []):
        lines.append(
            f"| {r.get('name')} | {r.get('p50_ms'):.3f} | {r.get('p95_ms'):.3f} | {r.get('mean_ms'):.3f} | {r.get('runs')} |"
        )
    lines.append("")

    lines.append("## 安全测试")
    lines.append("")
    gate = (security or {}).get("gate", {})
    lines.append(f"- 门禁：{'PASS' if gate.get('pass', True) else 'FAIL'}")
    lines.append(f"- bandit HIGH：{gate.get('bandit_high')}")
    lines.append(f"- pip-audit 漏洞数：{gate.get('pip_audit_total')}")
    lines.append("")

    lines.append("## 失败用例")
    lines.append("")
    if not failures:
        lines.append("- 无")
        lines.append("")
    else:
        for f in failures[:50]:
            lines.append(f"- {f['nodeid']}")
        lines.append("")

    lines.append("## 缺陷清单")
    lines.append("")
    lines.append(f"- {os.path.basename(args.defects)}")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

