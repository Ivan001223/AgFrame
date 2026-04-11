from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import TypedDict

from app.runtime.contracts.pruning import (
    ChatContextPruningPayload,
    build_chat_context_pruning_payload,
)


class WebSearchPayload(TypedDict):
    query: str
    result: str


class GradePayload(TypedDict):
    verdict: str
    reasoning: str
    issues: list[str]
    rewrite_instructions: str | None
    search_query: str | None


class WorkflowContextPayload(ChatContextPruningPayload, total=False):
    user_id: str
    grade: GradePayload
    search_query: str
    web_search: WebSearchPayload
    self_correction: str
    require_human_approval: bool
    interrupt_action_type: str
    interrupt_description: str
    interrupt_payload: dict[str, Any]


def build_workflow_context_payload(
    *,
    current: Mapping[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    grade: GradePayload | None = None,
    search_query: str | None = None,
    web_search: WebSearchPayload | None = None,
    self_correction: str | None = None,
    clear_search_query: bool = False,
    clear_web_search: bool = False,
    clear_self_correction: bool = False,
    clear_human_approval: bool = False,
    require_human_approval: bool | None = None,
    interrupt_action_type: str | None = None,
    interrupt_description: str | None = None,
    interrupt_payload: dict[str, Any] | None = None,
) -> WorkflowContextPayload:
    payload = cast(
        WorkflowContextPayload,
        build_chat_context_pruning_payload(
            current=current,
            session_id=session_id,
        ),
    )

    existing_user_id = payload.get("user_id")
    if isinstance(existing_user_id, str) and existing_user_id:
        payload["user_id"] = existing_user_id

    existing_grade = payload.get("grade")
    if isinstance(existing_grade, dict):
        payload["grade"] = existing_grade

    existing_web_search = payload.get("web_search")
    if not clear_web_search and isinstance(existing_web_search, dict):
        payload["web_search"] = existing_web_search

    existing_search_query = payload.get("search_query")
    if not clear_search_query and isinstance(existing_search_query, str):
        payload["search_query"] = existing_search_query

    existing_self_correction = payload.get("self_correction")
    if not clear_self_correction and isinstance(existing_self_correction, str):
        payload["self_correction"] = existing_self_correction

    existing_interrupt_action_type = payload.get("interrupt_action_type")
    if not clear_human_approval and isinstance(existing_interrupt_action_type, str):
        payload["interrupt_action_type"] = existing_interrupt_action_type

    existing_interrupt_description = payload.get("interrupt_description")
    if not clear_human_approval and isinstance(existing_interrupt_description, str):
        payload["interrupt_description"] = existing_interrupt_description

    existing_interrupt_payload = payload.get("interrupt_payload")
    if not clear_human_approval and isinstance(existing_interrupt_payload, dict):
        payload["interrupt_payload"] = existing_interrupt_payload

    existing_require_human_approval = payload.get("require_human_approval")
    if not clear_human_approval and isinstance(existing_require_human_approval, bool):
        payload["require_human_approval"] = existing_require_human_approval

    if clear_search_query:
        payload.pop("search_query", None)
    if clear_web_search:
        payload.pop("web_search", None)
    if clear_self_correction:
        payload.pop("self_correction", None)
    if clear_human_approval:
        payload.pop("require_human_approval", None)
        payload.pop("interrupt_action_type", None)
        payload.pop("interrupt_description", None)
        payload.pop("interrupt_payload", None)

    if grade is not None:
        payload["grade"] = grade
    if search_query:
        payload["search_query"] = search_query
    if web_search is not None:
        payload["web_search"] = web_search
    if self_correction is not None:
        payload["self_correction"] = self_correction
    if user_id is not None:
        payload["user_id"] = user_id
    if require_human_approval is not None:
        payload["require_human_approval"] = require_human_approval
    if interrupt_action_type is not None:
        payload["interrupt_action_type"] = interrupt_action_type
    if interrupt_description is not None:
        payload["interrupt_description"] = interrupt_description
    if interrupt_payload is not None:
        payload["interrupt_payload"] = interrupt_payload
    return payload
