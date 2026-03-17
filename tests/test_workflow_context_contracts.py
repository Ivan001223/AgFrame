from app.runtime.contracts.workflow_context import build_workflow_context_payload


def test_build_workflow_context_payload_preserves_and_overrides_workflow_fields():
    existing = {
        "session_id": "sess-1",
        "context_focus_hint": "focus",
        "grade": {
            "verdict": "rewrite",
            "reasoning": "old",
            "issues": ["missing_info"],
            "rewrite_instructions": "old rewrite",
            "search_query": None,
        },
        "search_query": "old query",
        "web_search": {"query": "old query", "result": "old result"},
        "self_correction": "old correction",
        "require_human_approval": True,
        "interrupt_action_type": "confirm",
        "interrupt_description": "need approval",
        "interrupt_payload": {"next_step": "generate"},
    }

    payload = build_workflow_context_payload(
        current=existing,
        grade={
            "verdict": "search",
            "reasoning": "need latest info",
            "issues": ["missing_info"],
            "rewrite_instructions": "search then rewrite",
            "search_query": "fresh query",
        },
        search_query="fresh query",
        web_search={"query": "fresh query", "result": "fresh result"},
        self_correction="search then rewrite",
    )

    assert payload["session_id"] == "sess-1"
    assert payload["grade"]["verdict"] == "search"
    assert payload["search_query"] == "fresh query"
    assert payload["web_search"]["result"] == "fresh result"
    assert payload["self_correction"] == "search then rewrite"
    assert payload["require_human_approval"] is True
    assert payload["interrupt_action_type"] == "confirm"
    assert payload["interrupt_payload"] == {"next_step": "generate"}


def test_build_workflow_context_payload_can_clear_stale_search_state():
    existing = {
        "search_query": "old query",
        "web_search": {"query": "old query", "result": "old result"},
        "self_correction": "old correction",
    }

    payload = build_workflow_context_payload(
        current=existing,
        clear_search_query=True,
        clear_web_search=True,
        clear_self_correction=True,
    )

    assert "search_query" not in payload
    assert "web_search" not in payload
    assert "self_correction" not in payload


def test_build_workflow_context_payload_can_clear_human_approval_state():
    existing = {
        "require_human_approval": True,
        "interrupt_action_type": "confirm",
        "interrupt_description": "need approval",
        "interrupt_payload": {"next_step": "generate"},
    }

    payload = build_workflow_context_payload(current=existing, clear_human_approval=True)

    assert "require_human_approval" not in payload
    assert "interrupt_action_type" not in payload
    assert "interrupt_description" not in payload
    assert "interrupt_payload" not in payload
