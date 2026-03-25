from __future__ import annotations

import json
from pathlib import Path

from scripts import eval_context_pruning
from app.runtime.prompts.context_pruner import ContextPruningConfig as RealContextPruningConfig


def test_eval_context_pruning_generates_summary(tmp_path: Path, monkeypatch):
    cases = {
        "cases": [
            {
                "id": "case-1",
                "query": "Where is retry logic?",
                "focus_hint": "retry backoff",
                "content": "setup\nretry controller\nbackoff line\ncleanup",
                "must_keep_lines": ["retry controller", "backoff line"],
            }
        ]
    }
    cases_path = tmp_path / "cases.json"
    out_path = tmp_path / "out.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    monkeypatch.setattr(eval_context_pruning, "prune_document_content", eval_context_pruning.prune_document_content)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_context_pruning.py",
            "--cases",
            str(cases_path),
            "--out",
            str(out_path),
        ],
    )

    assert eval_context_pruning.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 3
    assert {item["method"] for item in payload["summary"]} == {"heuristic", "auto", "reranker"}
    assert all(item["avg_required_recall"] >= 0 for item in payload["summary"])
    assert all("effective_methods" in item for item in payload["summary"])
    assert all("scoring_sources" in item for item in payload["summary"])
    assert all("hard_case_count" in item for item in payload["summary"])
    assert all("hard_all_required_rate" in item for item in payload["summary"])
    assert all("hard_avg_required_recall" in item for item in payload["summary"])
    assert all("unique_output_count" in item for item in payload["summary"])
    assert all("divergence_case_count" in item for item in payload["summary"])
    assert all("win_count" in item for item in payload["summary"])
    assert all("tie_count" in item for item in payload["summary"])


def test_eval_context_pruning_can_show_reranker_winning_semantic_case(tmp_path: Path, monkeypatch):
    semantic_content = "\n".join(
        [f"filler line {i}" for i in range(42)]
        + [
            "oauth callback received",
            "issue browser grant from callback state",
            "prolong identity artifact in cache",
            "ui success banner",
        ]
    )
    cases = {
        "cases": [
            {
                "id": "semantic-case",
                "query": "Which code renews login credentials after callback?",
                "focus_hint": "refresh token rotate after oauth callback",
                "content": semantic_content,
                "must_keep_lines": [
                    "issue browser grant from callback state",
                    "prolong identity artifact in cache",
                ],
            }
        ]
    }
    cases_path = tmp_path / "semantic_cases.json"
    out_path = tmp_path / "semantic_out.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    monkeypatch.setattr(
        eval_context_pruning,
        "ContextPruningConfig",
        lambda method: RealContextPruningConfig(
            method=method,
            min_keep_lines=1,
            max_keep_ratio=0.02,
            neighbor_window=0,
            max_lines_per_item=2,
            auto_reranker_min_lines=10,
            auto_reranker_min_chars=100,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_context_pruning.py",
            "--cases",
            str(cases_path),
            "--out",
            str(out_path),
        ],
    )

    assert eval_context_pruning.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    by_method = {item["method"]: item for item in payload["summary"]}

    assert by_method["reranker"]["avg_required_recall"] >= by_method["heuristic"]["avg_required_recall"]
    assert by_method["auto"]["effective_methods"] == ["reranker"]
