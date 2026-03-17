from __future__ import annotations

from typing import Any


QUALITY_SUMMARY_FIELDS = (
    "method",
    "case_count",
    "effective_methods",
    "scoring_sources",
    "all_required_rate",
    "avg_required_recall",
    "avg_char_saved_ratio",
    "hard_case_count",
    "hard_all_required_rate",
    "hard_avg_required_recall",
    "unique_output_count",
    "divergence_case_count",
    "win_count",
    "tie_count",
)


def empty_quality_summary(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "case_count": 0,
        "effective_methods": [],
        "scoring_sources": [],
        "all_required_rate": 0.0,
        "avg_required_recall": 0.0,
        "avg_char_saved_ratio": 0.0,
        "hard_case_count": 0,
        "hard_all_required_rate": 0.0,
        "hard_avg_required_recall": 0.0,
        "unique_output_count": 0,
        "divergence_case_count": 0,
        "win_count": 0,
        "tie_count": 0,
    }


def normalize_quality_summary(item: dict[str, Any]) -> dict[str, Any]:
    normalized = empty_quality_summary(str(item.get("method") or ""))
    for field in QUALITY_SUMMARY_FIELDS:
        if field in item:
            normalized[field] = item[field]
    return normalized
