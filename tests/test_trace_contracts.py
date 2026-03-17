from app.runtime.contracts.trace import build_agent_trace_payload


def test_build_agent_trace_payload_preserves_existing_and_overrides_target_fields():
    existing = {
        "trace_id": "trace-1",
        "self_correction_attempts": 1,
        "candidate_pruning": {
            "focus_hint": "old",
            "method": "heuristic",
            "scoring_source": "heuristic",
            "items": 2,
            "items_pruned": 1,
            "char_savings": {"saved": 20, "saved_ratio": 0.2},
            "line_savings": {"saved": 2, "saved_ratio": 0.2},
        },
        "prompt_pruning": {
            "focus_hint": "old",
            "method": "heuristic",
            "scoring_source": "heuristic",
            "docs": {
                "method": "heuristic",
                "scoring_source": "heuristic",
                "char_savings": {"saved": 10, "saved_ratio": 0.1},
                "line_savings": {"saved": 1, "saved_ratio": 0.1},
                "items": 1,
                "items_pruned": 0,
            },
            "memories": {
                "method": "heuristic",
                "scoring_source": "heuristic",
                "char_savings": {"saved": 5, "saved_ratio": 0.05},
                "line_savings": {"saved": 1, "saved_ratio": 0.05},
                "items": 1,
                "items_pruned": 0,
            },
        },
    }
    next_candidate = {
        "focus_hint": "new",
        "method": "auto",
        "scoring_source": "local_phrase_fallback",
        "items": 3,
        "items_pruned": 2,
        "char_savings": {"saved": 60, "saved_ratio": 0.6},
        "line_savings": {"saved": 6, "saved_ratio": 0.6},
    }

    payload = build_agent_trace_payload(
        current=existing,
        self_correction_attempts=2,
        candidate_pruning=next_candidate,
    )

    assert payload["trace_id"] == "trace-1"
    assert payload["self_correction_attempts"] == 2
    assert payload["candidate_pruning"]["method"] == "auto"
    assert payload["candidate_pruning"]["char_savings"]["saved"] == 60
    assert payload["prompt_pruning"]["docs"]["char_savings"]["saved"] == 10
