from app.runtime.contracts.pruning import (
    build_chat_context_pruning_payload,
    build_retrieval_debug_payload,
)


def test_build_chat_context_pruning_payload_preserves_existing_and_overrides_pruning_fields():
    existing = {
        "session_id": "sess-1",
        "context_focus_hint": "old hint",
        "web_search": {"query": "q1", "result": "r1"},
        "self_correction": "rewrite carefully",
        "retrieved_profile_items": [{"kind": "fact", "text": "prefers concise answers"}],
        "context_pruning": {"focus_hint": "old hint", "method": "heuristic", "scoring_source": "heuristic"},
        "retrieval_debug": {
            "candidate_pruning": {
                "items": 1,
                "items_pruned": 0,
                "char_count_before": 10,
                "char_count_after": 10,
                "line_count_before": 1,
                "line_count_after": 1,
                "ratio": 1.0,
                "focus_hint": "old hint",
                "enabled": True,
                "method": "heuristic",
                "scoring_source": "heuristic",
                "char_savings": {"saved": 0, "saved_ratio": 0.0},
                "line_savings": {"saved": 0, "saved_ratio": 0.0},
            }
        },
    }
    next_candidate = {
        "items": 3,
        "items_pruned": 2,
        "char_count_before": 90,
        "char_count_after": 30,
        "line_count_before": 9,
        "line_count_after": 3,
        "ratio": 0.3333,
        "focus_hint": "new hint",
        "enabled": True,
        "method": "auto",
        "scoring_source": "local_phrase_fallback",
        "char_savings": {"saved": 60, "saved_ratio": 0.6667},
        "line_savings": {"saved": 6, "saved_ratio": 0.6667},
    }
    next_prompt = {
        "focus_hint": "new hint",
        "method": "mixed",
        "scoring_source": "mixed",
        "docs": next_candidate,
        "memories": next_candidate,
    }

    retrieval_debug = build_retrieval_debug_payload(
        current=existing["retrieval_debug"],
        candidate_pruning=next_candidate,
    )
    payload = build_chat_context_pruning_payload(
        current=existing,
        focus_hint="new hint",
        prompt_pruning=next_prompt,
        retrieval_debug=retrieval_debug,
    )

    assert payload["session_id"] == "sess-1"
    assert payload["context_focus_hint"] == "new hint"
    assert payload["web_search"] == {"query": "q1", "result": "r1"}
    assert payload["self_correction"] == "rewrite carefully"
    assert payload["retrieved_profile_items"] == [{"kind": "fact", "text": "prefers concise answers"}]
    assert payload["context_pruning"]["focus_hint"] == "new hint"
    assert payload["retrieval_debug"]["candidate_pruning"]["method"] == "auto"
    assert payload["retrieval_debug"]["candidate_pruning"]["char_savings"]["saved"] == 60
