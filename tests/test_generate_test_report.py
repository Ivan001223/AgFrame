from __future__ import annotations

import json
from pathlib import Path

from scripts import generate_test_report


def test_generate_test_report_includes_context_pruning_section(tmp_path: Path, monkeypatch):
    pytest_json = {
        "summary": {"total": 2, "passed": 2, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0},
        "tests": [],
    }
    perf_json = {
        "results": [
            {"name": "context_pruning.heuristic", "p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.5, "runs": 2}
        ],
        "context_pruning": [
            {
                "method": "heuristic",
                "scoring_source": "heuristic",
                "char_before": 1000,
                "char_after": 300,
                "char_saved": 700,
                "char_saved_ratio": 0.7,
                "line_before": 100,
                "line_after": 40,
                "line_saved": 60,
                "line_saved_ratio": 0.6,
            }
        ],
    }
    pruning_eval_json = {
        "summary": [
            {
                "method": "heuristic",
                "effective_methods": ["heuristic"],
                "scoring_sources": ["heuristic"],
                "case_count": 3,
                "hard_case_count": 1,
                "all_required_rate": 1.0,
                "hard_all_required_rate": 1.0,
                "avg_required_recall": 1.0,
                "hard_avg_required_recall": 1.0,
                "avg_char_saved_ratio": 0.7,
                "unique_output_count": 2,
                "divergence_case_count": 1,
                "win_count": 2,
                "tie_count": 1,
            }
        ]
    }
    security_json = {"gate": {"pass": True, "bandit_high": 0, "pip_audit_total": 0}}
    coverage_xml = """<?xml version="1.0" ?>
<coverage line-rate="1.0" branch-rate="1.0">
  <packages>
    <package name="app">
      <classes>
        <class filename="app/runtime/prompts/prompt_builder.py" line-rate="1.0" />
        <class filename="app/runtime/llm/model_manager.py" line-rate="1.0" />
        <class filename="app/infrastructure/utils/security.py" line-rate="1.0" />
        <class filename="app/infrastructure/config/env.py" line-rate="1.0" />
      </classes>
    </package>
  </packages>
</coverage>
"""

    pytest_path = tmp_path / "pytest.json"
    perf_path = tmp_path / "perf.json"
    security_path = tmp_path / "security.json"
    pruning_eval_path = tmp_path / "context_pruning_eval.json"
    coverage_path = tmp_path / "coverage.xml"
    out_path = tmp_path / "report.md"
    defects_path = tmp_path / "defects.md"

    pytest_path.write_text(json.dumps(pytest_json), encoding="utf-8")
    perf_path.write_text(json.dumps(perf_json), encoding="utf-8")
    security_path.write_text(json.dumps(security_json), encoding="utf-8")
    pruning_eval_path.write_text(json.dumps(pruning_eval_json), encoding="utf-8")
    coverage_path.write_text(coverage_xml, encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
        "generate_test_report.py",
        "--pytest-json",
        str(pytest_path),
        "--coverage-xml",
        str(coverage_path),
        "--perf-json",
        str(perf_path),
        "--context-pruning-eval-json",
        str(pruning_eval_path),
        "--security-json",
        str(security_path),
        "--out",
        str(out_path),
        "--defects",
        str(defects_path),
        ],
    )

    assert generate_test_report.main() == 0

    report = out_path.read_text(encoding="utf-8")
    assert "## Context Pruning Benchmark" in report
    assert "## Context Pruning Quality" in report
    assert "| heuristic | heuristic | 1000 | 300 | 700 | 70.00% | 60 |" in report
    assert "| heuristic | heuristic | heuristic | 3 | 1 | 100.00% | 100.00% | 100.00% | 100.00% | 70.00% | 2 | 1 | 2 | 1 |" in report
