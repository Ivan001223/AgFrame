from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from scripts.pruning_report_schema import normalize_quality_summary


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
        "app/runtime/llm/model_manager.py": None,
        "app/infrastructure/utils/security.py": None,
        "app/infrastructure/config/env.py": None,
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


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _pruning_summaries(perf_json: dict[str, Any]) -> list[dict[str, Any]]:
    items = (perf_json or {}).get("context_pruning", [])
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(item)
    return out


def _pruning_quality_summaries(eval_json: dict[str, Any]) -> list[dict[str, Any]]:
    items = (eval_json or {}).get("summary", [])
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(normalize_quality_summary(item))
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
    parser.add_argument("--context-pruning-eval-json", required=True)
    parser.add_argument("--security-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--defects", required=True)
    args = parser.parse_args()

    pytest_json = _load_json(args.pytest_json)
    cov = _load_coverage_xml(args.coverage_xml)
    perf = _load_json(args.perf_json)
    pruning_eval = _load_json(args.context_pruning_eval_json)
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

    pruning = _pruning_summaries(perf)
    lines.append("## Context Pruning Benchmark")
    lines.append("")
    if not pruning:
        lines.append("- 无")
        lines.append("")
    else:
        lines.append("| 方法 | source | chars before | chars after | chars saved | saved % | lines saved |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for item in pruning:
            lines.append(
                f"| {item.get('method')} | "
                f"{item.get('scoring_source') or '-'} | "
                f"{_fmt_int(item.get('char_before'))} | "
                f"{_fmt_int(item.get('char_after'))} | "
                f"{_fmt_int(item.get('char_saved'))} | "
                f"{_pct(float(item.get('char_saved_ratio') or 0.0))} | "
                f"{_fmt_int(item.get('line_saved'))} |"
            )
        best = max(pruning, key=lambda item: float(item.get("char_saved_ratio") or 0.0))
        lines.append("")
        lines.append(
            f"- 最佳节省率：`{best.get('method')}`，节省 {_pct(float(best.get('char_saved_ratio') or 0.0))} chars，"
            f"共节省 {_fmt_int(best.get('char_saved'))}。"
        )
        lines.append("")

    pruning_quality = _pruning_quality_summaries(pruning_eval)
    lines.append("## Context Pruning Quality")
    lines.append("")
    if not pruning_quality:
        lines.append("- 无")
        lines.append("")
    else:
        lines.append("| 方法 | effective methods | scoring sources | case count | hard cases | all required rate | hard required rate | avg required recall | hard avg recall | avg char saved % | unique outputs | divergence cases | win count | tie count |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in pruning_quality:
            lines.append(
                f"| {item.get('method')} | "
                f"{', '.join(item.get('effective_methods') or []) or '-'} | "
                f"{', '.join(item.get('scoring_sources') or []) or '-'} | "
                f"{_fmt_int(item.get('case_count'))} | "
                f"{_fmt_int(item.get('hard_case_count'))} | "
                f"{_pct(float(item.get('all_required_rate') or 0.0))} | "
                f"{_pct(float(item.get('hard_all_required_rate') or 0.0))} | "
                f"{_pct(float(item.get('avg_required_recall') or 0.0))} | "
                f"{_pct(float(item.get('hard_avg_required_recall') or 0.0))} | "
                f"{_pct(float(item.get('avg_char_saved_ratio') or 0.0))} | "
                f"{_fmt_int(item.get('unique_output_count'))} | "
                f"{_fmt_int(item.get('divergence_case_count'))} | "
                f"{_fmt_int(item.get('win_count'))} | "
                f"{_fmt_int(item.get('tie_count'))} |"
            )
        best_quality = max(
            pruning_quality,
            key=lambda item: (
                float(item.get("all_required_rate") or 0.0),
                float(item.get("avg_required_recall") or 0.0),
            ),
        )
        lines.append("")
        lines.append(
            f"- 最佳保留质量：`{best_quality.get('method')}`，关键行全保留率 "
            f"{_pct(float(best_quality.get('all_required_rate') or 0.0))}，平均召回 "
            f"{_pct(float(best_quality.get('avg_required_recall') or 0.0))}。"
        )
        fallback_methods = [
            item.get("method")
            for item in pruning_quality
            if "local_phrase_fallback" in (item.get("scoring_sources") or [])
        ]
        model_backed_methods = [
            item.get("method")
            for item in pruning_quality
            if "reranker_model" in (item.get("scoring_sources") or [])
        ]
        if fallback_methods and not model_backed_methods:
            lines.append(
                f"- 当前环境未检测到模型型 reranker source；`{', '.join(str(method) for method in fallback_methods)}` "
                "使用 `local_phrase_fallback`。"
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
