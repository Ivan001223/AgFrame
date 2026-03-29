from __future__ import annotations

import pytest

from app.runtime.graph.nodes.bootstrap_request import bootstrap_request_node


@pytest.mark.anyio
async def test_bootstrap_request_node_prefers_server_configured_identity():
    result = await bootstrap_request_node(
        {
            "user_id": "spoofed-user",
            "session_id": "spoofed-session",
            "context": {
                "user_id": "spoofed-user",
                "session_id": "spoofed-session",
                "context_focus_hint": "focus on approvals",
            },
        },
        {
            "configurable": {
                "user_id": "u1",
                "thread_id": "server-session",
            }
        },
    )

    assert result["user_id"] == "u1"
    assert result["session_id"] == "server-session"
    assert result["context"]["user_id"] == "u1"
    assert result["context"]["session_id"] == "server-session"
    assert result["context"]["context_focus_hint"] == "focus on approvals"


@pytest.mark.anyio
async def test_bootstrap_request_node_preserves_existing_values_without_config():
    result = await bootstrap_request_node(
        {
            "user_id": "u1",
            "session_id": "s1",
            "context": {
                "user_id": "u1",
                "session_id": "s1",
                "require_human_approval": True,
            },
        }
    )

    assert result["user_id"] == "u1"
    assert result["session_id"] == "s1"
    assert result["context"]["require_human_approval"] is True
