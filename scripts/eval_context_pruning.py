from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.runtime.prompts.context_pruner import ContextPruningConfig, prune_document_content
from scripts.pruning_report_schema import empty_quality_summary


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    method: str
    effective_method: str
    scoring_source: str
    difficulty: str
    kept_all_required: bool
    required_recall: float
    char_saved_ratio: float
    output_fingerprint: str


def _load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("cases") or [])


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _evaluate_case(case: dict[str, Any], method: str) -> CaseResult:
    content = str(case.get("content") or "")
    query = str(case.get("query") or "")
    focus_hint = str(case.get("focus_hint") or "")
    must_keep_lines = [str(line) for line in list(case.get("must_keep_lines") or [])]

    pruned, stats = prune_document_content(
        content,
        query=query,
        focus_hint=focus_hint,
        config=ContextPruningConfig(method=method),
    )
    kept = [line for line in must_keep_lines if line in pruned]
    recall = len(kept) / len(must_keep_lines) if must_keep_lines else 1.0
    return CaseResult(
        case_id=str(case.get("id") or ""),
        method=method,
        effective_method=str(stats.get("method") or method),
        scoring_source=str(stats.get("scoring_source") or "heuristic"),
        difficulty=str(case.get("difficulty") or "standard"),
        kept_all_required=len(kept) == len(must_keep_lines),
        required_recall=round(recall, 4),
        char_saved_ratio=float((stats.get("char_savings") or {}).get("saved_ratio") or 0.0),
        output_fingerprint=_fingerprint(pruned),
    )


def _group_by_case(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)
    return grouped


def _method_outcomes(results: list[CaseResult], method: str) -> tuple[int, int, int]:
    grouped = _group_by_case(results)
    divergence_case_count = 0
    win_count = 0
    tie_count = 0
    for rows in grouped.values():
        fingerprints = {row.output_fingerprint for row in rows}
        if len(fingerprints) > 1 and any(row.method == method for row in rows):
            divergence_case_count += 1
        best_recall = max(row.required_recall for row in rows)
        contenders = [row for row in rows if row.required_recall == best_recall]
        best_saved = max(row.char_saved_ratio for row in contenders)
        winners = [
            row.method for row in contenders if row.char_saved_ratio == best_saved
        ]
        if len(winners) > 1 and method in winners:
            tie_count += 1
        elif winners == [method]:
            win_count += 1
    return divergence_case_count, win_count, tie_count


def _summarize(results: list[CaseResult], method: str) -> dict[str, Any]:
    rows = [result for result in results if result.method == method]
    hard_rows = [row for row in rows if row.difficulty in {"hard", "semantic"}]
    if not rows:
        return empty_quality_summary(method)
    divergence_case_count, win_count, tie_count = _method_outcomes(results, method)
    return {
        "method": method,
        "case_count": len(rows),
        "effective_methods": sorted({row.effective_method for row in rows}),
        "scoring_sources": sorted({row.scoring_source for row in rows}),
        "all_required_rate": round(sum(1 for row in rows if row.kept_all_required) / len(rows), 4),
        "avg_required_recall": round(sum(row.required_recall for row in rows) / len(rows), 4),
        "avg_char_saved_ratio": round(sum(row.char_saved_ratio for row in rows) / len(rows), 4),
        "hard_case_count": len(hard_rows),
        "hard_all_required_rate": round(
            sum(1 for row in hard_rows if row.kept_all_required) / len(hard_rows),
            4,
        ) if hard_rows else 0.0,
        "hard_avg_required_recall": round(
            sum(row.required_recall for row in hard_rows) / len(hard_rows),
            4,
        ) if hard_rows else 0.0,
        "unique_output_count": len({row.output_fingerprint for row in rows}),
        "divergence_case_count": divergence_case_count,
        "win_count": win_count,
        "tie_count": tie_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(Path("tests/fixtures/context_pruning_cases.json")),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    methods = ("heuristic", "auto", "reranker")
    per_case: list[CaseResult] = []
    for case in cases:
        for method in methods:
            per_case.append(_evaluate_case(case, method))

    payload = {
        "cases": [asdict(item) for item in per_case],
        "summary": [_summarize(per_case, method) for method in methods],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
