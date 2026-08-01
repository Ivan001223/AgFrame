from __future__ import annotations

import re
import time
import uuid
from typing import Any, cast

import anyio

from app.harness.contracts.run import HarnessTaskType
from app.harness.persistence.stores import HarnessModelProviderStore
from app.harness.runtime.checkpoint_adapter import CheckpointAdapter
from app.harness.runtime.run_service import build_run_service
from app.harness.runtime.verification_service import VerificationService
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.queue.redis_client import (
    append_task_incident,
    claim_task_operation,
    get_task,
    release_task_operation,
    update_task,
)
from app.infrastructure.storage.object_store import get_object_store
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.platform.contracts.runtime_protocol import (
    RuntimeCommandV1,
    runtime_resume_point_from_payload,
    runtime_resume_point_to_payload,
)
from app.platform.runtime.bootstrap import (
    build_runtime_command_for_run,
    build_runtime_execution_plan,
)
from app.platform.runtime.events import (
    build_runtime_completed_event,
    build_runtime_step_completed_event,
)
from app.platform.runtime.service import RuntimeApplicationService
from app.runtime.graph.orchestration_graph import (
    OrchestrationState,
    OutputGuardrailTrip,
    build_orchestration_execution_plan,
    invoke_orchestration_step,
    review_orchestration_output,
)
from app.runtime.graph.resume_service import GraphResumeService
from app.runtime.llm.provider_registry import ModelProviderRegistry
from app.skills.rag.rag_engine import get_rag_engine
from app.skills.registry import build_fallback_skill_descriptor, get_skill_descriptor
from app.skills.research.enhanced_search import (
    enhanced_search_response,
    enhanced_web_search,
    fetch_browser_previews,
)

_log = get_logger("task_queue.arq_jobs")


def _maybe_call(service: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(service, method_name, None)
    if callable(method):
        return method(*args, **kwargs)
    return None


def _accept_runtime_command_for_run(
    service: Any,
    *,
    run_id: str,
    task_type: str,
) -> RuntimeCommandV1:
    """Route the harness run through the platform runtime command gateway.

    This ensures every task type (document_ingest, agent_orchestration,
    session_resume_approval) enters execution via RuntimeApplicationService
    rather than bypassing the platform layer.
    """
    execution_plan = build_runtime_execution_plan(run_id=run_id, task_type=task_type)
    _maybe_call(
        service,
        "record_event",
        run_id,
        event_type="runtime.execution_planned",
        details=execution_plan,
    )
    runtime_command = build_runtime_command_for_run(run_id=run_id, task_type=task_type)
    runtime_result = RuntimeApplicationService(run_service=service).accept(runtime_command)
    _maybe_call(
        service,
        "record_event",
        run_id,
        event_type="runtime.command_accepted",
        details=dict(runtime_result.payload),
    )
    return runtime_result


def _persist_session_messages(*args: Any, **kwargs: Any) -> Any:
    from app.memory.session_history import persist_session_messages

    return persist_session_messages(*args, **kwargs)


persist_session_messages = _persist_session_messages


def _normalize_ingest_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if result is True:
        return {"ok": True}
    return {
        "ok": False,
        "error_code": "ingest_returned_false",
        "error_message": "add_knowledge_base 返回 False",
    }


def _add_knowledge_base_with_optional_library(
    file_path: str,
    source_uri: str | None,
    user_id: str | None,
    knowledge_base_id: str | None,
) -> Any:
    rag_engine = get_rag_engine()
    attempts = [
        {
            "source_uri": source_uri,
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
        },
        {
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
        },
        {
            "source_uri": source_uri,
            "user_id": user_id,
        },
        {
            "user_id": user_id,
        },
    ]
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return rag_engine.add_knowledge_base(file_path, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return rag_engine.add_knowledge_base(file_path)


def _normalize_skill_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_skill_catalog_lookup(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in graph.get("skill_catalog") or []:
        if not isinstance(item, dict):
            continue
        skill_id = _normalize_skill_key(str(item.get("skill_id") or ""))
        if not skill_id:
            continue
        lookup[skill_id] = dict(item)
    return lookup


def _build_skill_requirement_detail(skill_id: str, *, skill_catalog_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_skill_key(skill_id)
    catalog_item = skill_catalog_lookup.get(normalized)
    if catalog_item:
        return {
            "skill_id": normalized,
            "title": str(catalog_item.get("title") or normalized).strip(),
            "description": str(catalog_item.get("description") or "").strip() or None,
            "prompt_hint": str(catalog_item.get("prompt_hint") or "").strip() or None,
            "source": str(catalog_item.get("source") or "").strip() or None,
            "suggested_tool_ids": [
                str(tool_id).strip()
                for tool_id in catalog_item.get("suggested_tool_ids") or []
                if str(tool_id).strip()
            ],
            "suggested_mcp_server_ids": [
                str(server_id).strip()
                for server_id in catalog_item.get("suggested_mcp_server_ids") or []
                if str(server_id).strip()
            ],
        }

    descriptor = get_skill_descriptor(normalized) or build_fallback_skill_descriptor(normalized)
    return {
        "skill_id": normalized,
        "title": descriptor.title,
        "description": descriptor.description or None,
        "prompt_hint": descriptor.prompt_hint or None,
        "source": f"app/skills/{descriptor.skill_id}",
        "suggested_tool_ids": list(descriptor.suggested_tool_ids),
        "suggested_mcp_server_ids": list(descriptor.suggested_mcp_server_ids),
    }


def _normalize_task_checklist(values: Any) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    normalized: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        content = str(value.get("content") or "").strip()
        if not content:
            continue
        status = str(value.get("status") or "pending").strip() or "pending"
        if status not in {"pending", "in_progress", "completed"}:
            status = "pending"
        active_form = str(value.get("active_form") or "").strip() or content
        normalized.append(
            {
                "item_id": str(value.get("item_id") or f"check_{index + 1}"),
                "content": content,
                "status": status,
                "active_form": active_form,
            }
        )
    return normalized


def _format_task_checklist_context(values: Any) -> str:
    checklist = _normalize_task_checklist(values)
    if not checklist:
        return ""
    status_label = {
        "pending": "[pending]",
        "in_progress": "[in progress]",
        "completed": "[completed]",
    }
    lines = ["Execution checklist:"]
    for item in checklist:
        status = str(item.get("status") or "pending")
        content = str(
            item.get("active_form")
            if status == "in_progress"
            else item.get("content")
            or ""
        ).strip()
        if not content:
            continue
        lines.append(f"- {status_label.get(status, '[pending]')} {content}")
    if len(lines) == 1:
        return ""
    lines.append("Keep the checklist in mind when planning handoffs, ordering work, and deciding what remains unfinished.")
    return "\n".join(lines)


def _task_checklist_preview(values: Any, *, limit: int = 3) -> list[str]:
    checklist = _normalize_task_checklist(values)
    preview: list[str] = []
    for item in checklist:
        if str(item.get("status") or "pending") == "completed":
            continue
        content = str(item.get("active_form") or item.get("content") or "").strip()
        if not content:
            continue
        preview.append(content)
        if len(preview) >= limit:
            break
    return preview


def _snapshot_orchestration_state(state: dict[str, Any], *, fallback_task: str = "") -> dict[str, Any]:
    payload = {
        "task": str(state.get("task") or fallback_task),
        "agent_outputs": dict(state.get("agent_outputs") or {}),
        "output_artifacts": dict(state.get("output_artifacts") or {}),
        "current_agent": str(state.get("current_agent") or ""),
        "loop_index": int(state.get("loop_index") or 0),
        "errors": list(state.get("errors") or []),
    }
    knowledge_base_ids = _normalize_string_list(state.get("knowledge_base_ids"))
    if knowledge_base_ids:
        payload["knowledge_base_ids"] = knowledge_base_ids
    knowledge_context = str(state.get("knowledge_context") or "").strip()
    if knowledge_context:
        payload["knowledge_context"] = knowledge_context
    return payload


def _trim_prompt_context(text: str, *, max_chars: int) -> str:
    content = str(text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[: max(max_chars - 3, 1)].rstrip() + "..."


def _dedupe_trimmed_string_list(values: Any, *, limit: int = 4, max_chars: int = 180) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value or "").strip().split())
        if not normalized:
            continue
        normalized = _trim_prompt_context(normalized, max_chars=max_chars)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
        if len(output) >= limit:
            break
    return output


def _extract_handoff_summary_for_approval(content: str, *, max_chars: int = 280) -> str:
    paragraphs = [
        " ".join(part.strip().split())
        for part in re.split(r"\n\s*\n", str(content or "").strip())
        if str(part).strip()
    ]
    if not paragraphs:
        return ""
    return _trim_prompt_context(paragraphs[0], max_chars=max_chars)


def _extract_action_items_for_approval(content: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if re.match(r"^([-*]|\d+[.)])\s+", line):
            candidates.append(re.sub(r"^([-*]|\d+[.)])\s+", "", line).strip())
            continue
        if lower.startswith(("next:", "next step:", "action:", "actions:", "todo:", "follow-up:", "follow up:")):
            candidates.append(line.split(":", 1)[1].strip() if ":" in line else line)
    return _dedupe_trimmed_string_list(candidates, limit=4, max_chars=180)


def _extract_open_questions_for_approval(content: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("question:"):
            candidates.append(line.split(":", 1)[1].strip())
            continue
        if line.endswith("?"):
            candidates.append(line)
    return _dedupe_trimmed_string_list(candidates, limit=4, max_chars=180)


def _extract_risk_flags_for_approval(content: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if (
            lower.startswith(("risk:", "risks:", "blocker:", "blockers:", "unknown:", "uncertain:", "constraint:", "constraints:"))
            or "blocked by" in lower
            or "at risk" in lower
        ):
            candidates.append(line)
    return _dedupe_trimmed_string_list(candidates, limit=4, max_chars=180)


def _build_downstream_handoff_snapshot(agent_conf: dict[str, Any], *, limit: int = 4) -> list[dict[str, str]]:
    downstream_handoffs: list[dict[str, str]] = []
    agent_directory = dict(agent_conf.get("agent_directory") or {})
    for edge in list(agent_conf.get("outgoing_edges") or []):
        if not isinstance(edge, dict):
            continue
        target_agent_id = str(edge.get("target_agent_id") or "").strip()
        if not target_agent_id:
            continue
        target_agent = dict(agent_directory.get(target_agent_id) or {})
        downstream_handoffs.append(
            {
                "target_agent_id": target_agent_id,
                "target_agent_name": str(target_agent.get("name") or target_agent_id).strip(),
                "interaction": str(edge.get("interaction") or "handoff").strip() or "handoff",
            }
        )
        if len(downstream_handoffs) >= limit:
            break
    return downstream_handoffs


def _build_consumed_handoff_snapshot(
    agent_conf: dict[str, Any],
    *,
    state: dict[str, Any] | None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []

    consumed_handoffs: list[dict[str, Any]] = []
    agent_directory = dict(agent_conf.get("agent_directory") or {})
    outputs = dict(state.get("agent_outputs") or {})
    artifacts = dict(state.get("output_artifacts") or {})

    for edge in list(agent_conf.get("incoming_edges") or []):
        if not isinstance(edge, dict):
            continue
        source_agent_id = str(edge.get("source_agent_id") or "").strip()
        if not source_agent_id:
            continue
        source_agent = dict(agent_directory.get(source_agent_id) or {})
        source_output_key = str(source_agent.get("cluster_agent_id") or source_agent.get("agent_id") or source_agent_id).strip()
        source_name = str(source_agent.get("name") or source_agent_id).strip()
        interaction = str(edge.get("interaction") or "handoff").strip() or "handoff"
        source_output = str(outputs.get(source_output_key) or "").strip()
        source_artifact = dict(artifacts.get(source_output_key) or {})

        handoff: dict[str, Any] = {
            "source_agent_id": source_agent_id,
            "source_agent_name": source_name,
            "interaction": interaction,
        }
        artifact_summary = _trim_prompt_context(str(source_artifact.get("handoff_summary") or "").strip(), max_chars=220)
        if artifact_summary:
            handoff["artifact_summary"] = artifact_summary
        elif str(source_artifact.get("node_kind") or "") == "cluster":
            winning_strategy = _trim_prompt_context(str(source_artifact.get("winning_strategy") or "").strip(), max_chars=220)
            if winning_strategy:
                handoff["artifact_summary"] = winning_strategy
        if source_output:
            handoff["output_preview"] = _trim_prompt_context(source_output, max_chars=400)
            handoff["output_char_count"] = len(source_output)

        if len(handoff) <= 3:
            continue
        consumed_handoffs.append(handoff)
        if len(consumed_handoffs) >= limit:
            break

    return consumed_handoffs


def _coerce_non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _build_partial_output_artifact_snapshot(
    agent_conf: dict[str, Any],
    partial_output: str,
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = str(partial_output or "").strip()
    if not content:
        return {}
    downstream_handoffs = _build_downstream_handoff_snapshot(agent_conf)
    consumed_handoffs = _build_consumed_handoff_snapshot(agent_conf, state=state)
    snapshot: dict[str, Any] = {
        "node_kind": "agent",
        "agent_id": str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or "").strip(),
        "agent_name": str(agent_conf.get("cluster_name") or agent_conf.get("name") or "agent").strip(),
        "role": str(agent_conf.get("role") or "specialist").strip() or "specialist",
        "handoff_summary": _extract_handoff_summary_for_approval(content),
        "action_items": _extract_action_items_for_approval(content),
        "open_questions": _extract_open_questions_for_approval(content),
        "risk_flags": _extract_risk_flags_for_approval(content),
        "output_preview": _trim_prompt_context(content, max_chars=600),
        "output_char_count": len(content),
        "final_output": len(downstream_handoffs) == 0,
    }
    mcp_server_ids = _normalize_string_list(agent_conf.get("mcp_server_ids"))
    missing_mcp_server_ids = _normalize_string_list(agent_conf.get("missing_mcp_server_ids"))
    allowed_tool_ids = _normalize_string_list(agent_conf.get("allowed_tool_ids"))
    denied_tool_ids = _normalize_string_list(agent_conf.get("denied_tool_ids"))
    allowed_mcp_server_ids = _normalize_string_list(agent_conf.get("allowed_mcp_server_ids"))
    denied_mcp_server_ids = _normalize_string_list(agent_conf.get("denied_mcp_server_ids"))
    readiness_status = str(agent_conf.get("readiness_status") or "").strip()
    readiness_blockers = _normalize_string_list(agent_conf.get("readiness_blockers"))
    readiness_warnings = _normalize_string_list(agent_conf.get("readiness_warnings"))
    if mcp_server_ids:
        snapshot["mcp_server_ids"] = mcp_server_ids[:6]
    if missing_mcp_server_ids:
        snapshot["missing_mcp_server_ids"] = missing_mcp_server_ids[:6]
    if allowed_tool_ids:
        snapshot["allowed_tool_ids"] = allowed_tool_ids[:6]
    if denied_tool_ids:
        snapshot["denied_tool_ids"] = denied_tool_ids[:6]
    if allowed_mcp_server_ids:
        snapshot["allowed_mcp_server_ids"] = allowed_mcp_server_ids[:6]
    if denied_mcp_server_ids:
        snapshot["denied_mcp_server_ids"] = denied_mcp_server_ids[:6]
    if readiness_status:
        snapshot["readiness_status"] = readiness_status
    if readiness_blockers:
        snapshot["readiness_blockers"] = readiness_blockers[:4]
    if readiness_warnings:
        snapshot["readiness_warnings"] = readiness_warnings[:4]
    if downstream_handoffs:
        snapshot["downstream_handoffs"] = downstream_handoffs
    if consumed_handoffs:
        snapshot["consumed_handoffs"] = consumed_handoffs
    return snapshot


def _snapshot_research_payload_for_approval(research_payload: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    queries = _dedupe_trimmed_string_list(research_payload.get("queries"), limit=5, max_chars=160)
    if queries:
        snapshot["queries"] = queries

    digest = _trim_prompt_context(str(research_payload.get("digest") or "").strip(), max_chars=600)
    if digest:
        snapshot["digest"] = digest

    blocked = bool(research_payload.get("blocked"))
    if blocked:
        snapshot["blocked"] = True

    review_output = _trim_prompt_context(str(research_payload.get("review_output") or "").strip(), max_chars=400)
    if review_output:
        snapshot["review_output"] = review_output

    for field_name, fallback_field in (
        ("result_count", None),
        ("paper_count", "papers"),
        ("browser_preview_count", "browser_previews"),
        ("source_count", "sources"),
    ):
        if fallback_field is None:
            count_value = _coerce_non_negative_int(research_payload.get(field_name))
        else:
            raw_list = research_payload.get(fallback_field)
            count_value = len(raw_list) if isinstance(raw_list, list) else _coerce_non_negative_int(research_payload.get(field_name))
        if count_value is not None:
            snapshot[field_name] = count_value

    memory_payload = research_payload.get("memory")
    if isinstance(memory_payload, dict):
        memory_snapshot: dict[str, Any] = {
            "stored": bool(memory_payload.get("stored")),
        }
        reason = _trim_prompt_context(str(memory_payload.get("reason") or "").strip(), max_chars=200)
        if reason:
            memory_snapshot["reason"] = reason
        snapshot["memory"] = memory_snapshot

    error_message = _trim_prompt_context(str(research_payload.get("error") or "").strip(), max_chars=300)
    if error_message:
        snapshot["error"] = error_message

    return snapshot


def _snapshot_output_artifact_for_approval(artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, dict) or not artifact:
        return {}

    node_kind = str(artifact.get("node_kind") or "agent").strip() or "agent"
    if node_kind == "cluster":
        snapshot: dict[str, Any] = {
            "node_kind": "cluster",
            "cluster_agent_id": str(artifact.get("cluster_agent_id") or "").strip(),
            "cluster_name": str(artifact.get("cluster_name") or artifact.get("cluster_agent_id") or "cluster").strip(),
            "cluster_strategy": str(artifact.get("cluster_strategy") or "").strip() or None,
            "member_count": _coerce_non_negative_int(artifact.get("member_count")),
            "winning_vote": str(artifact.get("winning_vote") or "").strip() or None,
            "winning_strategy": _trim_prompt_context(str(artifact.get("winning_strategy") or "").strip(), max_chars=280) or None,
            "next_step": _trim_prompt_context(str(artifact.get("next_step") or "").strip(), max_chars=280) or None,
            "dominant_risks": _trim_prompt_context(str(artifact.get("dominant_risks") or "").strip(), max_chars=280) or None,
        }
        research_payload = artifact.get("research")
        if isinstance(research_payload, dict):
            research_snapshot = _snapshot_research_payload_for_approval(research_payload)
            if research_snapshot:
                snapshot["research"] = research_snapshot
        return {key: value for key, value in snapshot.items() if value not in (None, "", [], {})}

    snapshot = {
        "node_kind": "agent",
        "agent_id": str(artifact.get("agent_id") or "").strip(),
        "agent_name": str(artifact.get("agent_name") or artifact.get("agent_id") or "agent").strip(),
        "role": str(artifact.get("role") or "specialist").strip() or "specialist",
        "handoff_summary": _trim_prompt_context(str(artifact.get("handoff_summary") or "").strip(), max_chars=280) or None,
        "action_items": _dedupe_trimmed_string_list(artifact.get("action_items"), limit=4, max_chars=180),
        "open_questions": _dedupe_trimmed_string_list(artifact.get("open_questions"), limit=4, max_chars=180),
        "risk_flags": _dedupe_trimmed_string_list(artifact.get("risk_flags"), limit=4, max_chars=180),
        "output_preview": _trim_prompt_context(str(artifact.get("output_preview") or "").strip(), max_chars=600) or None,
        "output_char_count": _coerce_non_negative_int(artifact.get("output_char_count")),
        "final_output": bool(artifact.get("final_output")),
    }
    mcp_server_ids = _normalize_string_list(artifact.get("mcp_server_ids"))
    missing_mcp_server_ids = _normalize_string_list(artifact.get("missing_mcp_server_ids"))
    allowed_tool_ids = _normalize_string_list(artifact.get("allowed_tool_ids"))
    denied_tool_ids = _normalize_string_list(artifact.get("denied_tool_ids"))
    allowed_mcp_server_ids = _normalize_string_list(artifact.get("allowed_mcp_server_ids"))
    denied_mcp_server_ids = _normalize_string_list(artifact.get("denied_mcp_server_ids"))
    readiness_status = str(artifact.get("readiness_status") or "").strip()
    readiness_blockers = _dedupe_trimmed_string_list(artifact.get("readiness_blockers"), limit=4, max_chars=180)
    readiness_warnings = _dedupe_trimmed_string_list(artifact.get("readiness_warnings"), limit=4, max_chars=180)
    if mcp_server_ids:
        snapshot["mcp_server_ids"] = mcp_server_ids[:6]
    if missing_mcp_server_ids:
        snapshot["missing_mcp_server_ids"] = missing_mcp_server_ids[:6]
    if allowed_tool_ids:
        snapshot["allowed_tool_ids"] = allowed_tool_ids[:6]
    if denied_tool_ids:
        snapshot["denied_tool_ids"] = denied_tool_ids[:6]
    if allowed_mcp_server_ids:
        snapshot["allowed_mcp_server_ids"] = allowed_mcp_server_ids[:6]
    if denied_mcp_server_ids:
        snapshot["denied_mcp_server_ids"] = denied_mcp_server_ids[:6]
    if readiness_status:
        snapshot["readiness_status"] = readiness_status
    if readiness_blockers:
        snapshot["readiness_blockers"] = readiness_blockers
    if readiness_warnings:
        snapshot["readiness_warnings"] = readiness_warnings

    downstream_handoffs: list[dict[str, str]] = []
    for handoff in list(artifact.get("downstream_handoffs") or []):
        if not isinstance(handoff, dict):
            continue
        target_agent_id = str(handoff.get("target_agent_id") or "").strip()
        target_agent_name = str(handoff.get("target_agent_name") or target_agent_id).strip()
        interaction = str(handoff.get("interaction") or "handoff").strip() or "handoff"
        if not target_agent_id and not target_agent_name:
            continue
        downstream_handoffs.append(
            {
                "target_agent_id": target_agent_id,
                "target_agent_name": target_agent_name,
                "interaction": interaction,
            }
        )
        if len(downstream_handoffs) >= 4:
            break
    if downstream_handoffs:
        snapshot["downstream_handoffs"] = downstream_handoffs

    consumed_handoffs: list[dict[str, Any]] = []
    for handoff in list(artifact.get("consumed_handoffs") or []):
        if not isinstance(handoff, dict):
            continue
        source_agent_id = str(handoff.get("source_agent_id") or "").strip()
        source_agent_name = str(handoff.get("source_agent_name") or source_agent_id).strip()
        interaction = str(handoff.get("interaction") or "handoff").strip() or "handoff"
        if not source_agent_id and not source_agent_name:
            continue
        normalized_handoff: dict[str, Any] = {
            "source_agent_id": source_agent_id,
            "source_agent_name": source_agent_name,
            "interaction": interaction,
        }
        artifact_summary = _trim_prompt_context(str(handoff.get("artifact_summary") or "").strip(), max_chars=220)
        if artifact_summary:
            normalized_handoff["artifact_summary"] = artifact_summary
        output_preview = _trim_prompt_context(str(handoff.get("output_preview") or "").strip(), max_chars=400)
        if output_preview:
            normalized_handoff["output_preview"] = output_preview
        output_char_count = _coerce_non_negative_int(handoff.get("output_char_count"))
        if output_char_count is not None:
            normalized_handoff["output_char_count"] = output_char_count
        consumed_handoffs.append(normalized_handoff)
        if len(consumed_handoffs) >= 4:
            break
    if consumed_handoffs:
        snapshot["consumed_handoffs"] = consumed_handoffs

    tool_runs: list[dict[str, Any]] = []
    for tool_run in list(artifact.get("tool_runs") or []):
        if not isinstance(tool_run, dict):
            continue
        tool_id = str(tool_run.get("tool_id") or "").strip()
        if not tool_id:
            continue
        normalized_tool_run: dict[str, Any] = {
            "tool_id": tool_id,
            "status": str(tool_run.get("status") or "success").strip() or "success",
        }
        call_id = str(tool_run.get("call_id") or "").strip()
        if call_id:
            normalized_tool_run["call_id"] = call_id
        args_preview = _trim_prompt_context(str(tool_run.get("args_preview") or "").strip(), max_chars=220)
        if args_preview:
            normalized_tool_run["args_preview"] = args_preview
        result_preview = _trim_prompt_context(str(tool_run.get("result_preview") or "").strip(), max_chars=300)
        if result_preview:
            normalized_tool_run["result_preview"] = result_preview
        result_char_count = _coerce_non_negative_int(tool_run.get("result_char_count"))
        if result_char_count is not None:
            normalized_tool_run["result_char_count"] = result_char_count
        tool_runs.append(normalized_tool_run)
        if len(tool_runs) >= 4:
            break
    if tool_runs:
        snapshot["tool_runs"] = tool_runs

    return {key: value for key, value in snapshot.items() if value not in (None, "", [], {})}


def _attach_approval_artifact_snapshot(
    review_payload: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
    agent_conf: dict[str, Any] | None = None,
    partial_output: str | None = None,
    state: dict[str, Any] | None = None,
    artifact_source: str,
) -> dict[str, Any]:
    artifact_snapshot = _snapshot_output_artifact_for_approval(dict(artifact or {}))
    if not artifact_snapshot and agent_conf is not None and str(partial_output or "").strip():
        artifact_snapshot = _build_partial_output_artifact_snapshot(
            agent_conf,
            str(partial_output or ""),
            state=state,
        )
    if artifact_snapshot:
        review_payload["artifact_snapshot"] = artifact_snapshot
        review_payload["artifact_source"] = artifact_source
    return review_payload


def _format_orchestration_knowledge_context(docs: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    sections: list[str] = []
    sources: list[dict[str, Any]] = []
    for index, doc in enumerate(list(docs or [])[:3], start=1):
        content = str(getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue
        metadata = dict(getattr(doc, "metadata", {}) or {})
        knowledge_base_name = str(metadata.get("knowledge_base_name") or "").strip()
        knowledge_base_id = str(metadata.get("knowledge_base_id") or "").strip()
        source = str(metadata.get("source") or metadata.get("source_path") or "").strip()
        page_num = metadata.get("page_num")
        label_parts: list[str] = []
        if knowledge_base_name:
            label_parts.append(f"Knowledge base: {knowledge_base_name}")
        elif knowledge_base_id:
            label_parts.append(f"Knowledge base id: {knowledge_base_id}")
        if source:
            label_parts.append(f"Source: {source}")
        if page_num not in (None, ""):
            label_parts.append(f"Page: {page_num}")
        label = "; ".join(label_parts) or f"Retrieved context {index}"
        sections.append(f"[Knowledge source {index}] {label}\n{_trim_prompt_context(content, max_chars=1200)}")
        sources.append(
            {
                "knowledge_base_id": knowledge_base_id or None,
                "knowledge_base_name": knowledge_base_name or None,
                "source": source or None,
                "page_num": page_num,
            }
        )
    return "\n\n".join(sections).strip(), sources


def _filter_graph_for_execution(graph: dict[str, Any], selected_agent_ids: list[str]) -> dict[str, Any]:
    if not selected_agent_ids:
        return graph
    selected = {str(agent_id) for agent_id in selected_agent_ids}
    filtered = dict(graph)
    filtered["agents"] = [
        agent
        for agent in graph.get("agents") or []
        if isinstance(agent, dict) and str(agent.get("agent_id") or "") in selected
    ]
    filtered["edges"] = [
        edge
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("source_agent_id") or "") in selected
        and str(edge.get("target_agent_id") or "") in selected
    ]
    selected_scope_orchestration_summary = graph.get("selected_scope_orchestration_summary")
    if isinstance(selected_scope_orchestration_summary, dict):
        filtered["orchestration_summary"] = dict(selected_scope_orchestration_summary)
    return filtered


def _build_capability_snapshot(
    graph: dict[str, Any],
    *,
    active_agent_ids: list[str],
    handoff_scope: str = "all_agents",
) -> dict[str, Any]:
    normalized_active_agent_ids = [
        str(agent_id).strip() for agent_id in active_agent_ids if str(agent_id).strip()
    ]
    resolved_handoff_scope = "selected_agents" if str(handoff_scope).strip() == "selected_agents" else "all_agents"
    selected = set(normalized_active_agent_ids) if resolved_handoff_scope == "selected_agents" else set()
    agent_by_id = {
        str(agent.get("agent_id") or "").strip(): dict(agent)
        for agent in graph.get("agents") or []
        if isinstance(agent, dict) and str(agent.get("agent_id") or "").strip()
    }
    summary_by_id = {
        str(summary.get("agent_id") or "").strip(): dict(summary)
        for summary in graph.get("agent_capability_summaries") or []
        if isinstance(summary, dict) and str(summary.get("agent_id") or "").strip()
    }

    def _normalize_target_fit_list(value: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("agent_id") or "").strip()
            agent_name = str(item.get("agent_name") or agent_id).strip()
            if not agent_id or not agent_name:
                continue
            if selected and agent_id not in selected:
                continue
            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "score": _coerce_non_negative_int(item.get("score")) or 0,
                "fit": str(item.get("fit") or "weak").strip() or "weak",
            }
            rationale = str(item.get("rationale") or "").strip()
            if rationale:
                payload["rationale"] = rationale
            overlap_lane_ids = _normalize_string_list(item.get("overlap_lane_ids"))
            complementary_lane_ids = _normalize_string_list(item.get("complementary_lane_ids"))
            if overlap_lane_ids:
                payload["overlap_lane_ids"] = overlap_lane_ids
            if complementary_lane_ids:
                payload["complementary_lane_ids"] = complementary_lane_ids
            if item.get("edge_present") is not None:
                payload["edge_present"] = bool(item.get("edge_present"))
            interaction = str(item.get("interaction") or "").strip()
            if interaction:
                payload["interaction"] = interaction
            normalized.append(payload)
        return normalized

    def _normalize_coordination_preview_list(value: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value or []:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("agent_id") or "").strip()
            agent_name = str(item.get("agent_name") or agent_id).strip()
            if not agent_id or not agent_name or agent_id in seen:
                continue
            if selected and agent_id not in selected:
                continue
            seen.add(agent_id)
            normalized.append(
                {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                }
            )
        return normalized

    def _normalize_skill_detail_list(value: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value or []:
            if not isinstance(item, dict):
                continue
            skill_id = _normalize_skill_key(str(item.get("skill_id") or ""))
            title = str(item.get("title") or skill_id).strip()
            source = str(item.get("source") or f"app/skills/{skill_id}").strip()
            if not skill_id or not title or not source or skill_id in seen:
                continue
            seen.add(skill_id)
            normalized.append(
                {
                    "skill_id": skill_id,
                    "title": title,
                    "description": str(item.get("description") or "").strip() or None,
                    "source": source,
                    "status": str(item.get("status") or "available").strip() or "available",
                    "prompt_hint": str(item.get("prompt_hint") or "").strip() or None,
                    "suggested_tool_ids": _normalize_string_list(item.get("suggested_tool_ids")),
                    "suggested_mcp_server_ids": _normalize_string_list(item.get("suggested_mcp_server_ids")),
                }
            )
        return normalized

    def _normalize_mcp_server_detail_list(value: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value or []:
            if not isinstance(item, dict):
                continue
            server_id = _normalize_skill_key(str(item.get("server_id") or ""))
            title = str(item.get("title") or server_id).strip()
            if not server_id or not title or server_id in seen:
                continue
            seen.add(server_id)
            normalized.append(
                {
                    "server_id": server_id,
                    "title": title,
                    "description": str(item.get("description") or "").strip() or None,
                    "status": str(item.get("status") or "disabled").strip() or "disabled",
                    "command_preview": str(item.get("command_preview") or "").strip() or None,
                }
            )
        return normalized

    def _normalize_role_profile_suggestion(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        profile_id = str(value.get("profile_id") or "generalist").strip() or "generalist"
        normalized = {
            "profile_id": profile_id,
            "suggested_skill_ids": _normalize_string_list(value.get("suggested_skill_ids")),
            "available_skill_ids": _normalize_string_list(value.get("available_skill_ids")),
            "missing_skill_ids": _normalize_string_list(value.get("missing_skill_ids")),
            "suggested_tool_ids": _normalize_string_list(value.get("suggested_tool_ids")),
            "suggested_mcp_server_ids": _normalize_string_list(value.get("suggested_mcp_server_ids")),
            "restrictive_tool_ids": _normalize_string_list(value.get("restrictive_tool_ids")),
            "restrictive_mcp_server_ids": _normalize_string_list(value.get("restrictive_mcp_server_ids")),
        }
        return normalized

    def _derive_readiness_payload(summary: dict[str, Any]) -> dict[str, Any]:
        loaded_skill_ids = _normalize_string_list(summary.get("loaded_skill_ids"))
        missing_skill_ids = _normalize_string_list(summary.get("missing_skill_ids"))
        configured_allowed_tool_ids = _normalize_string_list(summary.get("configured_allowed_tool_ids"))
        disabled_tool_ids = _normalize_string_list(summary.get("disabled_tool_ids"))
        provider_limited_tool_ids = _normalize_string_list(summary.get("provider_limited_tool_ids"))
        configured_allowed_mcp_server_ids = _normalize_string_list(summary.get("configured_allowed_mcp_server_ids"))
        missing_mcp_server_ids = _normalize_string_list(summary.get("missing_mcp_server_ids"))
        unknown_allowed_tool_ids = _normalize_string_list(summary.get("unknown_allowed_tool_ids"))
        unknown_allowed_mcp_server_ids = _normalize_string_list(summary.get("unknown_allowed_mcp_server_ids"))
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()

        blockers: list[str] = []
        warnings: list[str] = []
        if missing_skill_ids:
            blockers.append(
                "Missing approved skills before this node can run: "
                + ", ".join(missing_skill_ids)
            )

        has_explicit_capability_requirements = bool(
            loaded_skill_ids
            or missing_skill_ids
            or configured_allowed_tool_ids
            or configured_allowed_mcp_server_ids
        )
        if has_explicit_capability_requirements and missing_mcp_server_ids:
            warnings.append(
                "Relevant MCP servers are not enabled in project inventory: "
                + ", ".join(missing_mcp_server_ids)
            )
        if has_explicit_capability_requirements and provider_limited_tool_ids:
            line = (
                "Current provider route cannot execute these tools directly: "
                + ", ".join(provider_limited_tool_ids)
            )
            if tool_execution_support_reason:
                line += f" ({tool_execution_support_reason})"
            warnings.append(line)
        if has_explicit_capability_requirements and disabled_tool_ids:
            warnings.append(
                "Some relevant tools stay disabled until feature flags change: "
                + ", ".join(disabled_tool_ids)
            )
        if unknown_allowed_tool_ids:
            warnings.append("Node policy references unknown tool ids: " + ", ".join(unknown_allowed_tool_ids))
        if unknown_allowed_mcp_server_ids:
            warnings.append("Node policy references unknown MCP ids: " + ", ".join(unknown_allowed_mcp_server_ids))

        status = str(summary.get("readiness_status") or "").strip()
        if status not in {"ready", "limited", "blocked"}:
            status = "blocked" if blockers else "limited" if warnings else "ready"

        return {
            "readiness_status": status,
            "readiness_blockers": _normalize_string_list(summary.get("readiness_blockers")) or blockers,
            "readiness_warnings": _normalize_string_list(summary.get("readiness_warnings")) or warnings,
        }

    def _derive_availability_payload(summary: dict[str, Any]) -> dict[str, Any]:
        missing_required_skill_ids = _normalize_string_list(summary.get("missing_required_skill_ids"))
        missing_required_tool_ids = _normalize_string_list(summary.get("missing_required_tool_ids"))
        missing_required_mcp_server_ids = _normalize_string_list(summary.get("missing_required_mcp_server_ids"))
        requires_tool_calling = bool(summary.get("requires_tool_calling"))
        tool_execution_support = str(summary.get("tool_execution_support") or "").strip()
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()

        blockers: list[str] = []
        warnings: list[str] = []
        if missing_required_skill_ids:
            blockers.append(
                "Definition requires approved skills that are not yet in the project pool: "
                + ", ".join(missing_required_skill_ids)
            )
        if missing_required_tool_ids:
            blockers.append(
                "Definition requires tools that are not currently enabled for this node: "
                + ", ".join(missing_required_tool_ids)
            )
        if missing_required_mcp_server_ids:
            blockers.append(
                "Definition requires enabled MCP servers that are not currently available: "
                + ", ".join(missing_required_mcp_server_ids)
            )
        if requires_tool_calling:
            if tool_execution_support == "unsupported":
                line = "Definition requires a provider route with direct tool-calling support"
                if tool_execution_support_reason:
                    line += f" ({tool_execution_support_reason})"
                blockers.append(line)
            elif tool_execution_support != "supported":
                line = "Definition expects direct tool-calling support, but the current provider route is not verified"
                if tool_execution_support_reason:
                    line += f" ({tool_execution_support_reason})"
                warnings.append(line)

        status = str(summary.get("availability_status") or "").strip()
        if status not in {"available", "limited", "unavailable"}:
            status = "unavailable" if blockers else "limited" if warnings else "available"

        return {
            "availability_status": status,
            "availability_blockers": _normalize_string_list(summary.get("availability_blockers")) or blockers,
            "availability_warnings": _normalize_string_list(summary.get("availability_warnings")) or warnings,
        }

    def _derive_execution_contract_payload(summary: dict[str, Any]) -> dict[str, Any]:
        approved_skill_ids = _normalize_string_list(summary.get("loaded_skill_ids"))
        suggested_skill_ids = _normalize_string_list(summary.get("suggested_skill_ids"))
        executable_tool_ids = _normalize_string_list(summary.get("enabled_tool_ids"))
        planning_only_tool_ids = _normalize_string_list(summary.get("provider_limited_tool_ids"))
        disabled_tool_ids = _normalize_string_list(summary.get("disabled_tool_ids"))
        planning_only_mcp_server_ids = _normalize_string_list(summary.get("mcp_server_ids"))
        missing_mcp_server_ids = _normalize_string_list(summary.get("missing_mcp_server_ids"))
        tool_execution_support = str(summary.get("tool_execution_support") or "").strip()
        if tool_execution_support == "unsupported" and executable_tool_ids:
            planning_only_tool_ids = list(dict.fromkeys([*planning_only_tool_ids, *executable_tool_ids]))
            executable_tool_ids = []
        if executable_tool_ids and planning_only_tool_ids:
            tool_access_mode = "mixed"
        elif executable_tool_ids:
            tool_access_mode = "direct_execution"
        elif planning_only_tool_ids:
            tool_access_mode = "planning_only"
        else:
            tool_access_mode = "none"
        mcp_access_mode = "planning_only" if planning_only_mcp_server_ids else "none"
        return {
            "skill_execution_mode": "guidance_only",
            "approved_skill_ids": approved_skill_ids,
            "suggested_skill_ids": suggested_skill_ids,
            "tool_access_mode": tool_access_mode,
            "executable_tool_ids": executable_tool_ids,
            "planning_only_tool_ids": planning_only_tool_ids,
            "disabled_tool_ids": disabled_tool_ids,
            "mcp_access_mode": mcp_access_mode,
            "planning_only_mcp_server_ids": planning_only_mcp_server_ids,
            "missing_mcp_server_ids": missing_mcp_server_ids,
        }

    def _derive_delegation_contract_payload(summary: dict[str, Any]) -> dict[str, Any]:
        contract = dict(summary.get("delegation_contract") or {}) if isinstance(summary.get("delegation_contract"), dict) else {}
        primary_role_mode = str(contract.get("primary_role_mode") or "generalist").strip() or "generalist"
        work_strategy = str(contract.get("work_strategy") or "flexible").strip() or "flexible"
        return {
            "primary_role_mode": primary_role_mode,
            "supporting_role_modes": _normalize_string_list(contract.get("supporting_role_modes")),
            "work_strategy": work_strategy,
            "should_coordinate_parallel_work": bool(contract.get("should_coordinate_parallel_work")),
            "should_produce_final_output": bool(contract.get("should_produce_final_output")),
            "primary_focus": str(contract.get("primary_focus") or "").strip() or None,
            "upstream_agents": _normalize_coordination_preview_list(contract.get("upstream_agents")),
            "downstream_agents": _normalize_coordination_preview_list(contract.get("downstream_agents")),
            "preferred_collaborators": _normalize_coordination_preview_list(contract.get("preferred_collaborators")),
            "weak_handoff_targets": _normalize_coordination_preview_list(contract.get("weak_handoff_targets")),
            "watchouts": _normalize_string_list(contract.get("watchouts")),
        }

    agent_capabilities: list[dict[str, Any]] = []
    for agent_id in normalized_active_agent_ids:
        summary = dict(summary_by_id.get(agent_id) or {})
        agent = agent_by_id.get(agent_id, {})
        readiness_payload = _derive_readiness_payload(summary)
        availability_payload = _derive_availability_payload(summary)
        execution_contract_payload = _derive_execution_contract_payload(summary)
        delegation_contract_payload = _derive_delegation_contract_payload(summary)
        agent_capabilities.append(
            {
                "agent_id": agent_id,
                "agent_name": str(agent.get("name") or agent_id).strip(),
                "role": str(agent.get("role") or "specialist").strip() or "specialist",
                "delegation_focus": str(summary.get("delegation_focus") or "").strip() or None,
                "delegation_lane_ids": _normalize_string_list(summary.get("delegation_lane_ids")),
                "loaded_skill_ids": _normalize_string_list(summary.get("loaded_skill_ids")),
                "missing_skill_ids": _normalize_string_list(summary.get("missing_skill_ids")),
                "missing_skill_details": _normalize_skill_detail_list(summary.get("missing_skill_details")),
                "suggested_skill_ids": _normalize_string_list(summary.get("suggested_skill_ids")),
                "loaded_skill_hints": _normalize_string_list(summary.get("loaded_skill_hints")),
                "required_skill_ids": _normalize_string_list(summary.get("required_skill_ids")),
                "missing_required_skill_ids": _normalize_string_list(summary.get("missing_required_skill_ids")),
                "required_tool_ids": _normalize_string_list(summary.get("required_tool_ids")),
                "missing_required_tool_ids": _normalize_string_list(summary.get("missing_required_tool_ids")),
                "configured_allowed_tool_ids": _normalize_string_list(summary.get("configured_allowed_tool_ids")),
                "configured_denied_tool_ids": _normalize_string_list(summary.get("configured_denied_tool_ids")),
                "enabled_tool_ids": _normalize_string_list(summary.get("enabled_tool_ids")),
                "disabled_tool_ids": _normalize_string_list(summary.get("disabled_tool_ids")),
                "policy_added_tool_ids": _normalize_string_list(summary.get("policy_added_tool_ids")),
                "policy_blocked_tool_ids": _normalize_string_list(summary.get("policy_blocked_tool_ids")),
                "unknown_allowed_tool_ids": _normalize_string_list(summary.get("unknown_allowed_tool_ids")),
                "requires_tool_calling": bool(summary.get("requires_tool_calling")),
                "provider_limited_tool_ids": _normalize_string_list(summary.get("provider_limited_tool_ids")),
                "tool_execution_support": str(summary.get("tool_execution_support") or "").strip() or None,
                "tool_execution_support_reason": str(summary.get("tool_execution_support_reason") or "").strip() or None,
                "required_mcp_server_ids": _normalize_string_list(summary.get("required_mcp_server_ids")),
                "missing_required_mcp_server_ids": _normalize_string_list(summary.get("missing_required_mcp_server_ids")),
                "configured_allowed_mcp_server_ids": _normalize_string_list(
                    summary.get("configured_allowed_mcp_server_ids")
                ),
                "configured_denied_mcp_server_ids": _normalize_string_list(
                    summary.get("configured_denied_mcp_server_ids")
                ),
                "mcp_server_ids": _normalize_string_list(summary.get("mcp_server_ids")),
                "missing_mcp_server_ids": _normalize_string_list(summary.get("missing_mcp_server_ids")),
                "missing_mcp_server_details": _normalize_mcp_server_detail_list(
                    summary.get("missing_mcp_server_details")
                ),
                "policy_added_mcp_server_ids": _normalize_string_list(summary.get("policy_added_mcp_server_ids")),
                "policy_blocked_mcp_server_ids": _normalize_string_list(
                    summary.get("policy_blocked_mcp_server_ids")
                ),
                "unknown_allowed_mcp_server_ids": _normalize_string_list(
                    summary.get("unknown_allowed_mcp_server_ids")
                ),
                "recommended_collaborators": _normalize_target_fit_list(summary.get("recommended_collaborators")),
                "downstream_handoff_scores": _normalize_target_fit_list(summary.get("downstream_handoff_scores")),
                "availability_status": availability_payload["availability_status"],
                "availability_blockers": availability_payload["availability_blockers"],
                "availability_warnings": availability_payload["availability_warnings"],
                "readiness_status": readiness_payload["readiness_status"],
                "readiness_blockers": readiness_payload["readiness_blockers"],
                "readiness_warnings": readiness_payload["readiness_warnings"],
                "provider_route": str(summary.get("provider_route") or "").strip() or None,
                "review_mode": str(summary.get("review_mode") or "").strip() or None,
                "capability_brief": str(summary.get("capability_brief") or "").strip() or None,
                "execution_contract": execution_contract_payload,
                "delegation_contract": delegation_contract_payload,
                "role_profile_suggestion": _normalize_role_profile_suggestion(
                    summary.get("role_profile_suggestion")
                ),
            }
        )

    mcp_server_catalog: list[dict[str, Any]] = []
    for item in graph.get("mcp_server_catalog") or []:
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server_id") or "").strip()
        if not server_id:
            continue
        mcp_server_catalog.append(
            {
                "server_id": server_id,
                "title": str(item.get("title") or server_id).strip(),
                "status": str(item.get("status") or "enabled").strip() or "enabled",
            }
        )

    capability_snapshot: dict[str, Any] = {
        "active_agent_ids": normalized_active_agent_ids,
        "agent_capabilities": agent_capabilities,
        "handoff_diagnostic_scope": resolved_handoff_scope,
    }
    if mcp_server_catalog:
        capability_snapshot["mcp_server_catalog"] = mcp_server_catalog
    if resolved_handoff_scope == "selected_agents":
        capability_snapshot["captured_from_selected_agents"] = True
    return capability_snapshot


def _build_provider_registry(*, user_id: str | None) -> ModelProviderRegistry:
    registry = ModelProviderRegistry()
    rows = HarnessModelProviderStore().list_providers(user_id=user_id)
    registry.load_from_store_rows(rows)
    return registry


def _record_auto_review_decision(
    service: Any,
    *,
    run_id: str,
    reason: str | None,
    payload_json: dict[str, Any],
    requested_by: str | None,
    review_agent_name: str,
    comment: str | None,
) -> Any:
    resolved = _maybe_call(
        service,
        "create_resolved_approval",
        run_id=run_id,
        action_type="orchestration_review",
        reason=reason,
        payload_json=payload_json,
        requested_by=requested_by,
        status="rejected",
        resolved_by=review_agent_name,
        comment=comment,
    )
    if resolved is not None:
        return resolved

    created = _maybe_call(
        service,
        "create_approval_request",
        run_id=run_id,
        action_type="orchestration_review",
        reason=reason,
        payload_json=payload_json,
        requested_by=requested_by,
    )
    approval_id = str((created or {}).get("approval_id") or "").strip() if isinstance(created, dict) else ""
    if approval_id:
        updated = _maybe_call(
            service,
            "update_approval",
            approval_id,
            status="rejected",
            resolved_by=review_agent_name,
            comment=comment,
        )
        if updated is not None:
            return updated
    return created


def _record_review_notification(
    service: Any,
    *,
    run_id: str,
    verdict: str,
    title: str,
    message: str | None,
    reviewer: str | None,
    review_stage: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> Any:
    return _maybe_call(
        service,
        "record_event",
        run_id,
        event_type="run.notification_ready",
        details={
            "notification_type": "review_verdict",
            "delivery_status": "ready",
            "verdict": str(verdict or "").strip() or "unknown",
            "title": str(title or "").strip() or "Run update ready",
            "message": str(message or "").strip() or None,
            "reviewer": str(reviewer or "").strip() or None,
            "review_stage": str(review_stage or "").strip() or None,
            "agent_id": str(agent_id or "").strip() or None,
            "agent_name": str(agent_name or "").strip() or None,
        },
    )


def _finalize_auto_review_rejection(
    service: Any,
    verification_service: VerificationService,
    *,
    run_id: str,
    active_agent_ids: list[str],
    loop_count: int,
    review_agent_enabled: bool,
    error_code: str,
    error_message: str,
    review_details: dict[str, Any],
    agent_outputs: dict[str, str] | None = None,
    output_artifacts: dict[str, dict[str, Any]] | None = None,
    recovery_mode: str | None = None,
    capability_snapshot: dict[str, Any] | None = None,
    selected_agent_ids: list[str] | None = None,
) -> bool:
    verification = verification_service.build_agent_orchestration_result(
        ok=False,
        active_agent_ids=active_agent_ids,
        blocked_agents=[],
        loop_count=loop_count,
        review_agent_enabled=review_agent_enabled,
        error_code=error_code,
        error_message=error_message,
        agent_outputs=agent_outputs,
        output_artifacts=output_artifacts,
        recovery_mode=recovery_mode,
        review_details=review_details,
        capability_snapshot=capability_snapshot,
        handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
        selected_agent_ids=selected_agent_ids,
    )
    if callable(getattr(service, "create_verification", None)) and callable(getattr(service, "mark_rejected", None)):
        _maybe_call(service, "mark_verifying", run_id)
        _maybe_call(service, "create_verification", run_id, verification)
        _maybe_call(service, "mark_rejected", run_id)
        return False
    _maybe_call(service, "complete_with_verification", run_id, verification)
    return False


def _serialize_orchestration_resume_state(state: dict[str, Any], next_step_index: int) -> dict[str, Any]:
    payload = runtime_resume_point_to_payload({"next_step_index": next_step_index})
    payload["state"] = _snapshot_orchestration_state(state)
    return payload


def _build_review_blocked_resume_payload(
    *,
    state: dict[str, Any],
    next_step_index: int,
    rollback_state: dict[str, Any] | None = None,
    continuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _serialize_orchestration_resume_state(state, next_step_index)
    if rollback_state is not None:
        payload["rollback_state"] = rollback_state
    if continuation is not None:
        payload["continuation"] = continuation
    return payload


def _window_text_for_stream_review(
    text: str,
    *,
    window_chars: int,
    end_char: int | None = None,
) -> tuple[str, int, int]:
    content = str(text or "")
    if not content:
        return "", 0, 0
    normalized_window = max(int(window_chars or 0), 1)
    end = min(len(content), max(int(end_char if end_char is not None else len(content)), 0))
    start = max(0, end - normalized_window)
    return content[start:end], start, end


async def _review_incremental_stream_output(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_conf: dict[str, Any],
    stream_review_state: dict[str, int],
    chunk_text: str,
    accumulated_text: str,
    chunk_index: int,
) -> dict[str, Any] | None:
    review_cursor = int(stream_review_state.get("last_reviewed_chars") or 0)
    trigger_chars = max(int(review_config.get("stream_review_trigger_chars") or 24), 1)
    window_chars = int(review_config.get("stream_review_window_chars") or 1200)
    check_count = int(stream_review_state.get("check_count") or 0)
    should_force_trailing_check = bool(str(chunk_text or "").strip())

    while review_cursor < len(accumulated_text):
        remaining = len(accumulated_text) - review_cursor
        if remaining < trigger_chars and review_cursor > 0 and not should_force_trailing_check:
            break
        review_end = min(len(accumulated_text), review_cursor + trigger_chars)
        window_text, start_char, end_char = _window_text_for_stream_review(
            accumulated_text,
            window_chars=window_chars,
            end_char=review_end,
        )
        if not window_text.strip():
            stream_review_state["last_reviewed_chars"] = review_end
            review_cursor = review_end
            should_force_trailing_check = False
            continue
        check_count += 1
        review_result = await review_orchestration_output(
            review_config=review_config,
            provider_config=provider_config,
            default_timeout=default_timeout,
            provider_registry=provider_registry,
            task=task,
            agent_id=f"{str(agent_conf.get('cluster_agent_id') or agent_conf.get('agent_id') or '')}::stream::{check_count}",
            agent_name=f"{str(agent_conf.get('cluster_name') or agent_conf.get('name') or 'agent')} live stream guard",
            output=window_text,
        )
        if not bool(review_result.get("approved")):
            return {
                "blocked": True,
                "review_stage": "agent_output_stream",
                "review_output": str(review_result.get("review_output") or ""),
                "chunk_index": int(chunk_index),
                "check_count": int(check_count),
                "segment_index": int(check_count) - 1,
                "segment_count": int(check_count),
                "segment_start_char": int(start_char),
                "segment_end_char": int(end_char),
                "last_reviewed_char": int(end_char),
                "review_cursor_char": int(end_char),
                "segment_preview": window_text[-400:],
                "partial_output": accumulated_text[:end_char],
            }
        stream_review_state["last_reviewed_chars"] = review_end
        stream_review_state["check_count"] = check_count
        review_cursor = review_end
        should_force_trailing_check = False

    return None


def _restore_orchestration_state(task: str, resume_payload: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
    payload = dict(resume_payload or {})
    resume_point = runtime_resume_point_from_payload(cast(dict[str, object], payload))
    state_payload = dict(payload.get("state") or {})
    next_step_index = int(resume_point.next_step_index)
    restored = {
        "messages": [],
        "task": str(state_payload.get("task") or task),
        "agent_outputs": dict(state_payload.get("agent_outputs") or {}),
        "output_artifacts": dict(state_payload.get("output_artifacts") or {}),
        "current_agent": str(state_payload.get("current_agent") or ""),
        "loop_index": int(state_payload.get("loop_index") or 0),
        "errors": list(state_payload.get("errors") or []),
    }
    knowledge_base_ids = _normalize_string_list(state_payload.get("knowledge_base_ids"))
    if knowledge_base_ids:
        restored["knowledge_base_ids"] = knowledge_base_ids
    knowledge_context = str(state_payload.get("knowledge_context") or "").strip()
    if knowledge_context:
        restored["knowledge_context"] = knowledge_context
    if isinstance(resume_point.continuation, dict) and resume_point.continuation:
        restored["continuation"] = dict(resume_point.continuation)
    return restored, next_step_index


def _build_cluster_research_plan(*, task: str, artifact: dict[str, Any]) -> dict[str, list[str]]:
    cluster_name = str(artifact.get("cluster_name") or artifact.get("cluster_agent_id") or "cluster").strip()
    winning_strategy = str(artifact.get("winning_strategy") or artifact.get("winning_vote") or "").strip()
    next_step = str(artifact.get("next_step") or "").strip()
    base_task = str(task or "").strip()
    paper_prompts = [
        f"{base_task} {winning_strategy} benchmark evaluation research paper".strip(),
        f"{base_task} {next_step or cluster_name} recent advances arxiv".strip(),
    ]
    web_prompts = [
        f"{base_task} {winning_strategy} latest progress".strip(),
        f"{base_task} {next_step or cluster_name} latest industry developments".strip(),
    ]

    def _dedupe(values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            normalized = " ".join(str(value or "").split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                items.append(normalized)
        return items[:limit]

    paper_queries = _dedupe(paper_prompts, 2)
    web_queries = _dedupe(web_prompts, 2)
    return {
        "paper_queries": paper_queries,
        "web_queries": web_queries,
        "queries": paper_queries + web_queries,
    }


def _build_cluster_research_queries(*, task: str, artifact: dict[str, Any]) -> list[str]:
    return _build_cluster_research_plan(task=task, artifact=artifact)["queries"]


def _parse_search_result_items(result: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    pattern = re.compile(r"^\d+\.\s+\[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)\s*$")
    lines = [line.rstrip() for line in str(result or "").splitlines()]
    current: dict[str, str] | None = None
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            if current:
                items.append(current)
            current = {
                "title": match.group("title").strip(),
                "url": match.group("url").strip(),
                "snippet": "",
            }
            continue
        if current and line.strip():
            snippet = line.strip().removeprefix("-").strip()
            current["snippet"] = f"{current['snippet']} {snippet}".strip()
    if current:
        items.append(current)
    return items


def _search_response_to_items(response: Any) -> list[dict[str, str]]:
    results = getattr(response, "results", None)
    if not isinstance(results, list):
        return []
    items: list[dict[str, str]] = []
    for result in results:
        title = str(getattr(result, "title", "") or "").strip()
        url = str(getattr(result, "url", "") or "").strip()
        snippet = str(getattr(result, "snippet", "") or "").strip()
        provider = str(getattr(result, "provider", "") or "").strip()
        if not url:
            continue
        items.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "provider": provider,
            }
        )
    return items


def _chunk_output_for_review(
    output: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    text = str(output or "")
    normalized_chunk = max(int(chunk_chars or 0), 1)
    normalized_overlap = max(0, min(int(overlap_chars or 0), normalized_chunk - 1))
    if not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    total_length = len(text)
    while start < total_length:
        end = min(total_length, start + normalized_chunk)
        chunks.append(
            {
                "segment_index": len(chunks),
                "start_char": start,
                "end_char": end,
                "content": text[start:end],
            }
        )
        if end >= total_length:
            break
        start = max(end - normalized_overlap, start + 1)
    return chunks


async def _run_segmented_output_review(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    output: str,
) -> dict[str, Any]:
    review_enabled = bool(review_config.get("enabled", True))
    pipeline_enabled = bool(review_config.get("pipeline_review_enabled", True))
    if not review_enabled or not pipeline_enabled:
        return {
            "approved": True,
            "mode": "disabled",
            "segments_reviewed": 0,
            "segment_count": 0,
            "blocked_segment": None,
            "results": [],
        }

    chunks = _chunk_output_for_review(
        output,
        chunk_chars=int(review_config.get("pipeline_chunk_chars") or 1200),
        overlap_chars=int(review_config.get("pipeline_chunk_overlap_chars") or 150),
    )
    if len(chunks) <= 1:
        return {
            "approved": True,
            "mode": "single_segment_bypass",
            "segments_reviewed": len(chunks),
            "segment_count": len(chunks),
            "blocked_segment": None,
            "results": [],
        }

    results: list[dict[str, Any]] = []
    total = len(chunks)
    for chunk in chunks:
        segment_index = int(chunk["segment_index"])
        review_result = await review_orchestration_output(
            review_config=review_config,
            provider_config=provider_config,
            default_timeout=default_timeout,
            provider_registry=provider_registry,
            task=task,
            agent_id=f"{agent_id}::segment::{segment_index + 1}",
            agent_name=f"{agent_name} segment {segment_index + 1}/{total}",
            output=str(chunk["content"] or ""),
        )
        chunk_result = {
            **chunk,
            "approved": bool(review_result.get("approved")),
            "review_output": str(review_result.get("review_output") or ""),
            "segment_count": total,
        }
        results.append(chunk_result)
        if not chunk_result["approved"]:
            return {
                "approved": False,
                "mode": "segmented",
                "segments_reviewed": len(results),
                "segment_count": total,
                "blocked_segment": chunk_result,
                "results": results,
            }

    return {
        "approved": True,
        "mode": "segmented",
        "segments_reviewed": len(results),
        "segment_count": total,
        "blocked_segment": None,
        "results": results,
    }


def _classify_research_source(item: dict[str, str]) -> str:
    haystack = f"{item.get('title', '')} {item.get('url', '')} {item.get('snippet', '')}".lower()
    if any(token in haystack for token in ["arxiv", "paper", "doi", "ieee", "acm", "springer"]):
        return "paper"
    return "web"


def _summarize_research_payload(searches: list[dict[str, Any]], *, browser_previews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sources: list[dict[str, str]] = []
    citations: list[dict[str, str]] = []
    latest_progress: list[str] = []
    papers: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for search in searches:
        query = str(search.get("query") or "").strip()
        for item in search.get("items") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            source_kind = _classify_research_source(item)
            normalized = {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": str(item.get("snippet") or "").strip(),
                "query": query,
                "kind": source_kind,
            }
            sources.append(normalized)
            citations.append(
                {
                    "label": normalized["title"] or normalized["url"],
                    "url": normalized["url"],
                    "kind": source_kind,
                }
            )
            if normalized["snippet"]:
                latest_progress.append(normalized["snippet"])
            if source_kind == "paper":
                papers.append(normalized)

    latest_progress = latest_progress[:5]
    papers = papers[:5]
    normalized_previews = [
        {
            "url": str(item.get("url") or ""),
            "final_url": str(item.get("final_url") or ""),
            "title": str(item.get("title") or ""),
            "description": str(item.get("description") or ""),
            "status_code": item.get("status_code"),
            "content_type": str(item.get("content_type") or "") or None,
        }
        for item in (browser_previews or [])
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    ][:3]
    digest_sections: list[str] = []
    if papers:
        digest_sections.append(
            "Paper-First Findings:\n"
            + "\n".join(f"- {item['title']} ({item['url']})" for item in papers)
        )
    if latest_progress:
        digest_sections.append("Latest Progress:\n" + "\n".join(f"- {item}" for item in latest_progress))
    if normalized_previews:
        digest_sections.append(
            "Browser Previews:\n"
            + "\n".join(
                f"- {item['title'] or item['url']}: {item['description'] or 'No preview description.'}"
                for item in normalized_previews
            )
        )
    if not digest_sections and sources:
        digest_sections.append(
            "Sources:\n"
            + "\n".join(f"- {item['title'] or item['url']} ({item['url']})" for item in sources[:5])
        )
    return {
        "sources": sources[:8],
        "citations": citations[:8],
        "latest_progress": latest_progress,
        "papers": papers,
        "browser_previews": normalized_previews,
        "digest": "\n\n".join(digest_sections).strip(),
    }


async def _run_cluster_auto_research(
    *,
    task: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    plan = _build_cluster_research_plan(task=task, artifact=artifact)
    searches: list[dict[str, Any]] = []
    provider_runs: list[dict[str, Any]] = []

    for query in plan["paper_queries"]:
        response = await enhanced_search_response(
            query=query,
            provider="arxiv",
            use_cache=True,
            max_results=4,
        )
        items = _search_response_to_items(response)
        searches.append(
            {
                "query": query,
                "provider": "arxiv",
                "items": items,
                "result_count": len(items),
            }
        )
        provider_runs.append(
            {
                "provider": "arxiv",
                "query": query,
                "result_count": len(items),
                "cached": bool(getattr(response, "cached", False)),
            }
        )

    for query in plan["web_queries"]:
        result = await enhanced_web_search(
            query=query,
            provider=None,
            use_cache=True,
            max_results=5,
        )
        items = _parse_search_result_items(result)
        searches.append(
            {
                "query": query,
                "provider": "web",
                "result": result,
                "items": items,
                "result_count": len(items),
            }
        )
        provider_runs.append(
            {
                "provider": "web",
                "query": query,
                "result_count": len(items),
                "cached": True,
            }
        )

    preview_urls: list[str] = []
    for search in searches:
        prioritized_items = search.get("items") or []
        if not isinstance(prioritized_items, list):
            continue
        for item in prioritized_items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in preview_urls:
                continue
            preview_urls.append(url)
            if len(preview_urls) >= 3:
                break
        if len(preview_urls) >= 3:
            break

    browser_previews = await anyio.to_thread.run_sync(
        lambda: [
            preview.__dict__
            for preview in fetch_browser_previews(preview_urls, max_previews=3, timeout_seconds=8)
        ]
    ) if preview_urls else []

    summary = _summarize_research_payload(searches, browser_previews=browser_previews)
    return {
        "research_mode": "paper_first_browser_preview",
        "queries": plan["queries"],
        "paper_queries": plan["paper_queries"],
        "web_queries": plan["web_queries"],
        "result_count": len(searches),
        "searches": searches,
        "provider_runs": provider_runs,
        "sources": summary["sources"],
        "citations": summary["citations"],
        "latest_progress": summary["latest_progress"],
        "papers": summary["papers"],
        "browser_previews": summary["browser_previews"],
        "digest": summary["digest"],
    }


async def _review_cluster_research_evidence(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    research_payload: dict[str, Any],
) -> dict[str, object]:
    digest = str(research_payload.get("digest") or "").strip()
    latest_progress = list(research_payload.get("latest_progress") or [])
    papers = list(research_payload.get("papers") or [])
    browser_previews = list(research_payload.get("browser_previews") or [])
    evidence_lines = [
        f"Task: {task}",
        f"Cluster node id: {agent_id}",
        f"Cluster node name: {agent_name}",
        "",
        "Structured research evidence:",
        f"- Queries: {', '.join(str(query) for query in research_payload.get('queries') or []) or 'none'}",
        f"- Latest progress count: {len(latest_progress)}",
        f"- Paper count: {len(papers)}",
        f"- Browser preview count: {len(browser_previews)}",
    ]
    if latest_progress:
        evidence_lines.append("Latest progress:")
        evidence_lines.extend(f"- {str(item)}" for item in latest_progress[:5])
    if papers:
        evidence_lines.append("Papers:")
        evidence_lines.extend(
            f"- {str(item.get('title') or 'Untitled')} ({str(item.get('url') or 'no-url')})"
            for item in papers[:5]
            if isinstance(item, dict)
        )
    if browser_previews:
        evidence_lines.append("Browser previews:")
        evidence_lines.extend(
            f"- {str(item.get('title') or item.get('url') or 'Untitled')}: {str(item.get('description') or 'no description')}"
            for item in browser_previews[:3]
            if isinstance(item, dict)
        )
    if digest:
        evidence_lines.extend(["", "Digest:", digest])
    evidence_text = "\n".join(evidence_lines).strip()
    return await review_orchestration_output(
        review_config=review_config,
        provider_config=provider_config,
        default_timeout=default_timeout,
        provider_registry=provider_registry,
        task=task,
        agent_id=f"{agent_id}::research",
        agent_name=f"{agent_name} research evidence",
        output=evidence_text,
    )


def _persist_cluster_research_memory(
    *,
    user_id: str | None,
    run_id: str,
    cluster_id: str,
    research_digest: str,
) -> dict[str, Any]:
    if not user_id or not research_digest.strip():
        return {"stored": False, "reason": "missing_user_or_digest"}
    if not ensure_schema_if_possible():
        return {"stored": False, "reason": "schema_unavailable"}
    try:
        UserMemoryEngine().add_chat_summary(
            user_id=user_id,
            session_id=f"harness_run:{run_id}:{cluster_id}",
            summary_text=research_digest,
        )
        return {"stored": True, "reason": None}
    except Exception as exc:
        return {"stored": False, "reason": str(exc)}


async def ingest_pdf(
    ctx: dict[str, Any],
    task_id: str,
    storage_uri: str,
    user_id: str | None = None,
    knowledge_base_id: str | None = None,
) -> bool:
    logger = bind_logger(_log, session_id=task_id, node="ingest_pdf")
    started_at = int(time.time())
    existing_task = await get_task(task_id)
    operation_key = str(existing_task.get("operation_key") or "")
    await update_task(
        task_id,
        {
            "status": "running",
            "progress": 1,
            "step": "start",
            "started_at": started_at,
            "message": "开始处理",
            "error": "",
            "user_id": user_id or "unknown",
        },
    )

    try:
        await update_task(
            task_id, {"progress": 10, "step": "validating", "message": "校验文件"}
        )
        await update_task(
            task_id, {"progress": 25, "step": "ingest", "message": "开始摄取"}
        )
        await update_task(
            task_id, {"progress": 60, "step": "indexing", "message": "构建索引"}
        )
        await update_task(
            task_id, {"progress": 85, "step": "finalizing", "message": "写入结果"}
        )
        # 传递 user_id 给 RAG 引擎
        with get_object_store().materialize_to_local_path(storage_uri) as local_path:
            result = await anyio.to_thread.run_sync(
                lambda: _normalize_ingest_result(
                    _add_knowledge_base_with_optional_library(
                        file_path=local_path,
                        source_uri=storage_uri,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id,
                    )
                )
            )
        finished_at = int(time.time())
        if result.get("ok"):
            await update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "step": "done",
                    "finished_at": finished_at,
                    "message": "处理完成",
                    "retryable": "false",
                    "result_stage": str(result.get("stage") or ""),
                },
            )
            logger.info("task succeeded storage_uri=%s", storage_uri)
            await release_task_operation(operation_key, expected_task_id=task_id)
            return True

        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "failed",
                "finished_at": finished_at,
                "error": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "result_stage": str(result.get("stage") or ""),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "error_message": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "stage": str(result.get("stage") or ""),
                "file_path": storage_uri,
                "timestamp": finished_at,
            }
        )
        logger.info(
            "task failed storage_uri=%s error_code=%s stage=%s",
            storage_uri,
            result.get("error_code"),
            result.get("stage"),
        )
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False
    except Exception as e:
        finished_at = int(time.time())
        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "exception",
                "finished_at": finished_at,
                "error": str(e),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": "task_exception",
                "error_message": str(e),
                "stage": "exception",
                "file_path": storage_uri,
                "timestamp": finished_at,
            }
        )
        logger.exception("task exception storage_uri=%s", storage_uri)
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False


async def run_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    service = build_run_service()
    run = service.get_run(run_id)
    if not run:
        return False

    verification_service = VerificationService()
    task_type = str(run.get("task_type") or "")
    run_status = str(run.get("status") or "").strip()
    if run_status in {"completed", "failed"}:
        return run_status == "completed"

    execution_lock_key = f"harness_run:{task_type}:{run_id}"
    execution_lock_owner = f"{run_id}:{uuid.uuid4()}"
    execution_lock_acquired = False
    try:
        claimed_owner = await claim_task_operation(
            execution_lock_key,
            execution_lock_owner,
            ttl_seconds=2 * 60 * 60,
        )
        if claimed_owner != execution_lock_owner:
            _log.info("skipping duplicate harness execution run_id=%s task_type=%s", run_id, task_type)
            return False
        execution_lock_acquired = True
    except Exception as exc:
        _log.warning("harness execution lock unavailable run_id=%s error=%s", run_id, exc)

    if task_type != HarnessTaskType.SESSION_RESUME_APPROVAL.value:
        _maybe_call(service, "mark_running", run_id)

    try:
        if task_type == HarnessTaskType.DOCUMENT_INGEST.value:
            _maybe_call(service, "set_current_step", run_id, "ingest_document")
            _accept_runtime_command_for_run(service, run_id=run_id, task_type=task_type)
            input_json = cast(dict[str, Any], run.get("input_json") or {})
            storage_uri = str(input_json.get("storage_uri") or input_json.get("file_path") or "")
            user_id = str(run.get("user_id") or "") or None
            knowledge_base_id = str(input_json.get("knowledge_base_id") or "").strip() or None

            with get_object_store().materialize_to_local_path(storage_uri) as local_path:
                result = await anyio.to_thread.run_sync(
                    lambda: _normalize_ingest_result(
                        _add_knowledge_base_with_optional_library(
                            file_path=local_path,
                            source_uri=storage_uri,
                            user_id=user_id,
                            knowledge_base_id=knowledge_base_id,
                        )
                    )
                )
            verification = verification_service.build_document_ingest_result(
                ok=bool(result.get("ok")),
                stage=str(result.get("stage") or "") or None,
                error_code=str(result.get("error_code") or "") or None,
                error_message=str(result.get("error_message") or "") or None,
            )
            service.complete_with_verification(run_id, verification)
            return bool(result.get("ok"))

        if task_type == HarnessTaskType.AGENT_ORCHESTRATION.value:
            input_json = cast(dict[str, Any], run.get("input_json") or {})
            metadata_json = cast(dict[str, Any], run.get("metadata_json") or {})
            _accept_runtime_command_for_run(service, run_id=run_id, task_type=task_type)
            graph = input_json.get("graph") if isinstance(input_json, dict) else {}
            graph = graph if isinstance(graph, dict) else {}
            agents = cast(list[dict[str, Any]], graph.get("agents") if isinstance(graph.get("agents"), list) else [])
            selected_agent_ids = input_json.get("selected_agent_ids") if isinstance(input_json, dict) else []
            selected_agent_ids = selected_agent_ids if isinstance(selected_agent_ids, list) else []
            user_id = str(run.get("user_id") or "") or None
            loop_count = int(input_json.get("loop_count") or 1) if isinstance(input_json, dict) else 1
            review_agent = cast(
                dict[str, Any],
                graph.get("review_agent") if isinstance(graph.get("review_agent"), dict) else {},
            )
            review_agent_enabled = bool(review_agent.get("enabled", True))
            skill_catalog_lookup = _build_skill_catalog_lookup(graph)
            loaded_skill_ids = {
                _normalize_skill_key(str(item.get("skill_id") or ""))
                for item in graph.get("skill_pool") or []
                if isinstance(item, dict)
            }

            active_agents = [
                agent
                for agent in agents
                if isinstance(agent, dict)
                and (
                    not selected_agent_ids
                    or str(agent.get("agent_id") or "") in {str(agent_id) for agent_id in selected_agent_ids}
                )
            ]
            active_agent_ids = [str(agent.get("agent_id") or "") for agent in active_agents]
            capability_snapshot = _build_capability_snapshot(
                graph,
                active_agent_ids=active_agent_ids,
                handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
            )
            capability_by_agent_id = {
                str(item.get("agent_id") or "").strip(): dict(item)
                for item in capability_snapshot.get("agent_capabilities") or []
                if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
            }
            if not active_agents:
                verification = verification_service.build_agent_orchestration_result(
                    ok=False,
                    active_agent_ids=[],
                    blocked_agents=[],
                    loop_count=loop_count,
                    review_agent_enabled=review_agent_enabled,
                    error_code="missing_active_agents",
                    error_message="no agents selected for orchestration",
                    capability_snapshot={},
                    handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
                    selected_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
                )
                service.complete_with_verification(run_id, verification)
                return False

            _maybe_call(service, "set_current_step", run_id, "prepare_orchestration")
            _maybe_call(
                service,
                "record_event",
                run_id,
                event_type="orchestration.review_agent_attached",
                details={
                    "enabled": review_agent_enabled,
                    "hidden": bool(review_agent.get("hidden", True)),
                    "name": str(review_agent.get("name") or "Compliance reviewer"),
                    "model": str(review_agent.get("model") or "gpt-5.1-codex-mini"),
                },
            )

            blocked_agents: list[dict[str, object]] = []
            for agent in active_agents:
                agent_id = str(agent.get("agent_id") or "")
                capability = capability_by_agent_id.get(agent_id, {})
                required_skills = _normalize_string_list(capability.get("required_skill_ids")) or [
                    _normalize_skill_key(str(skill_id))
                    for skill_id in (agent.get("skill_ids") or [])
                    if str(skill_id).strip()
                ]
                missing_skills = _normalize_string_list(capability.get("missing_required_skill_ids")) or [
                    skill_id for skill_id in required_skills if skill_id not in loaded_skill_ids
                ]
                required_skill_details = [
                    _build_skill_requirement_detail(skill_id, skill_catalog_lookup=skill_catalog_lookup)
                    for skill_id in required_skills
                ]
                missing_skill_details = [
                    detail
                    for detail in required_skill_details
                    if str(detail.get("skill_id") or "") in set(missing_skills)
                ]
                missing_required_tool_ids = _normalize_string_list(capability.get("missing_required_tool_ids"))
                missing_required_mcp_server_ids = _normalize_string_list(
                    capability.get("missing_required_mcp_server_ids")
                )
                availability_blockers = _normalize_string_list(capability.get("availability_blockers"))
                readiness_blockers = _normalize_string_list(capability.get("readiness_blockers"))
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.agent_ready",
                    details={
                        "agent_id": agent_id,
                        "agent_name": str(agent.get("name") or agent_id),
                        "role": str(agent.get("role") or "specialist"),
                        "skill_ids": required_skills,
                        "skill_details": required_skill_details,
                        "missing_skills": missing_skills,
                        "missing_skill_details": missing_skill_details,
                        "missing_required_tool_ids": missing_required_tool_ids,
                        "missing_required_mcp_server_ids": missing_required_mcp_server_ids,
                        "availability_blockers": availability_blockers,
                        "readiness_blockers": readiness_blockers,
                    },
                )
                if (
                    missing_skills
                    or missing_required_tool_ids
                    or missing_required_mcp_server_ids
                    or availability_blockers
                    or readiness_blockers
                ):
                    blocked_agents.append(
                        {
                            "agent_id": agent_id,
                            "agent_name": str(agent.get("name") or agent_id),
                            "missing_skills": missing_skills,
                            "missing_skill_details": missing_skill_details,
                            "missing_required_tool_ids": missing_required_tool_ids,
                            "missing_required_mcp_server_ids": missing_required_mcp_server_ids,
                            "availability_blockers": availability_blockers,
                            "readiness_blockers": readiness_blockers,
                        }
                    )

            if blocked_agents:
                capability_blocked = any(
                    item.get("missing_required_tool_ids")
                    or item.get("missing_required_mcp_server_ids")
                    or item.get("availability_blockers")
                    or item.get("readiness_blockers")
                    for item in blocked_agents
                )
                verification = verification_service.build_agent_orchestration_result(
                    ok=False,
                    active_agent_ids=active_agent_ids,
                    blocked_agents=blocked_agents,
                    loop_count=loop_count,
                    review_agent_enabled=review_agent_enabled,
                    error_code="agent_capability_blocked" if capability_blocked else "missing_skill_approval",
                    error_message=(
                        "one or more agents are blocked by missing capabilities"
                        if capability_blocked
                        else "one or more agents are blocked by missing skills"
                    ),
                    capability_snapshot=capability_snapshot,
                    handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
                    selected_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
                )
                service.complete_with_verification(run_id, verification)
                return False

            task_str = str(input_json.get("task") or "") or str(input_json.get("project_name") or "")
            task_checklist = _normalize_task_checklist(input_json.get("task_checklist") if isinstance(input_json, dict) else None)
            checklist_context = _format_task_checklist_context(task_checklist)
            if checklist_context:
                task_str = f"{task_str}\n\n{checklist_context}".strip() if task_str.strip() else checklist_context
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.checklist_loaded",
                    details={
                        "checklist_count": len(task_checklist),
                        "open_item_count": len(
                            [item for item in task_checklist if str(item.get("status") or "pending") != "completed"]
                        ),
                        "open_items_preview": _task_checklist_preview(task_checklist),
                    },
                )
            default_timeout = int(input_json.get("timeout_seconds") or 60)
            recovery_mode = str(metadata_json.get("review_recovery_mode") or "").strip() or None
            knowledge_base_ids = _normalize_string_list(input_json.get("knowledge_base_ids") if isinstance(input_json, dict) else None)
            if not knowledge_base_ids:
                knowledge_base_ids = _normalize_string_list(graph.get("knowledge_base_ids") if isinstance(graph, dict) else None)
            execution_graph = _filter_graph_for_execution(graph, selected_agent_ids)
            provider_registry = _build_provider_registry(user_id=user_id)
            ordered_agents, provider_config, review_config = build_orchestration_execution_plan(execution_graph)
            review_enabled = bool(review_config.get("enabled", True))
            resume_payload = input_json.get("orchestration_resume") if isinstance(input_json, dict) else None
            approved_resume = bool(isinstance(resume_payload, dict) and resume_payload.get("review_decision") == "approved")
            resume_continue_mode = (
                str(resume_payload.get("continue_mode") or "").strip()
                if isinstance(resume_payload, dict)
                else ""
            )
            stream_continuation_pending = approved_resume and resume_continue_mode == "accept_partial_stream_output"
            initial_state, next_step_index = _restore_orchestration_state(task_str, resume_payload if isinstance(resume_payload, dict) else None)
            if knowledge_base_ids:
                initial_state["knowledge_base_ids"] = knowledge_base_ids
            if knowledge_base_ids and task_str.strip() and not str(initial_state.get("knowledge_context") or "").strip():
                try:
                    knowledge_docs = await anyio.to_thread.run_sync(
                        lambda: get_rag_engine().retrieve_context(
                            task_str,
                            user_id=user_id,
                            knowledge_base_ids=knowledge_base_ids,
                        )
                    )
                    knowledge_context, knowledge_sources = _format_orchestration_knowledge_context(
                        list(knowledge_docs or [])
                    )
                    if knowledge_context:
                        initial_state["knowledge_context"] = knowledge_context
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.knowledge_context_loaded",
                            details={
                                "knowledge_base_ids": knowledge_base_ids,
                                "source_count": len(knowledge_sources),
                                "sources": knowledge_sources,
                            },
                        )
                    else:
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.knowledge_context_unavailable",
                            details={
                                "knowledge_base_ids": knowledge_base_ids,
                                "reason": "no_matching_context",
                            },
                        )
                except Exception as exc:
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.knowledge_context_unavailable",
                        details={
                            "knowledge_base_ids": knowledge_base_ids,
                            "reason": "knowledge_retrieval_failed",
                            "error": str(exc),
                        },
                    )
            if approved_resume and not stream_continuation_pending:
                input_json = dict(input_json)
                input_json.pop("orchestration_resume", None)
                _maybe_call(service, "update_run_input_json", run_id, input_json)
            elif stream_continuation_pending:
                continuation = initial_state.get("continuation") if isinstance(initial_state, dict) else None
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.stream_continuation_resumed",
                    details={
                        "agent_id": str((continuation or {}).get("agent_id") or ""),
                        "agent_name": str((continuation or {}).get("agent_name") or ""),
                        "partial_length": len(str((continuation or {}).get("partial_output") or "")),
                        "next_step_index": next_step_index,
                    },
                )
                _maybe_call(
                    service,
                    "patch_runtime_state",
                    run_id,
                    continuation={
                        "enabled": True,
                        "mode": "continue_with_partial_stream_output",
                        "status": "resumed",
                        "agent_id": str((continuation or {}).get("agent_id") or ""),
                        "agent_name": str((continuation or {}).get("agent_name") or ""),
                        "step_index": int(next_step_index),
                        "prefix_length": len(str((continuation or {}).get("partial_output") or "")),
                        "resumed_at": int(time.time() * 1000),
                    },
                )
            execution_steps: list[dict[str, Any]] = [
                {"loop_number": loop_number, "agent_config": agent_conf}
                for loop_number in range(1, max(loop_count, 1) + 1)
                for agent_conf in ordered_agents
            ]

            _maybe_call(service, "set_current_step", run_id, "executing_graph")
            final_state = initial_state
            for step_index in range(next_step_index, len(execution_steps)):
                step = execution_steps[step_index]
                agent_conf = cast(dict[str, Any], step["agent_config"])
                final_state["loop_index"] = int(step["loop_number"]) - 1
                previous_state = _snapshot_orchestration_state(final_state, fallback_task=task_str)
                previous_state["messages"] = []
                stream_review_state = {
                    "last_reviewed_chars": 0,
                    "check_count": 0,
                }

                async def _review_stream_chunk(
                    chunk_text: str,
                    accumulated_text: str,
                    chunk_index: int,
                    *,
                    agent_conf: dict[str, Any] = agent_conf,
                    step_index: int = step_index,
                    stream_review_state: dict[str, int] = stream_review_state,
                ):
                    if not review_enabled:
                        return None
                    blocked = await _review_incremental_stream_output(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_conf=agent_conf,
                        stream_review_state=stream_review_state,
                        chunk_text=chunk_text,
                        accumulated_text=accumulated_text,
                        chunk_index=chunk_index,
                    )
                    if blocked:
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            review={
                                "stage": "agent_output_stream",
                                "status": "pending",
                                "agent_id": str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or ""),
                                "agent_name": str(agent_conf.get("cluster_name") or agent_conf.get("name") or "agent"),
                                "review_output": str(blocked.get("review_output") or ""),
                                "check_count": int(blocked.get("check_count") or 0),
                                "segment_index": int(blocked.get("segment_index") or 0),
                                "segment_count": int(blocked.get("segment_count") or 0),
                                "segment_start_char": int(blocked.get("segment_start_char") or 0),
                                "segment_end_char": int(blocked.get("segment_end_char") or 0),
                                "last_reviewed_char": int(blocked.get("last_reviewed_char") or 0),
                            },
                            continuation={
                                "enabled": True,
                                "mode": "continue_with_partial_stream_output",
                                "status": "pending",
                                "agent_id": str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or ""),
                                "agent_name": str(agent_conf.get("cluster_name") or agent_conf.get("name") or "agent"),
                                "step_index": int(step_index),
                                "prefix_length": len(str(blocked.get("partial_output") or "")),
                            },
                        )
                    return blocked

                try:
                    final_state.update(
                        await invoke_orchestration_step(
                            agent_config=agent_conf,
                            provider_config=provider_config,
                            default_timeout=default_timeout,
                            provider_registry=provider_registry,
                            state=cast(OrchestrationState, final_state),
                            on_stream_chunk=_review_stream_chunk,
                        )
                    )
                except OutputGuardrailTrip as exc:
                    agent_output_key = str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or "")
                    agent_name = str(agent_conf.get("cluster_name") or agent_conf.get("name") or agent_output_key)
                    blocked_payload = dict(exc.payload or {})
                    partial_content = str(exc.partial_content or "")
                    serialized_resume = _build_review_blocked_resume_payload(
                        state=previous_state,
                        next_step_index=step_index,
                        rollback_state=_serialize_orchestration_resume_state(previous_state, step_index)["state"],
                        continuation={
                            "resume_agent_id": str(agent_conf.get("agent_id") or agent_output_key),
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "partial_output": partial_content,
                            "review_output": str(blocked_payload.get("review_output") or ""),
                            "review_stage": "agent_output_stream",
                            "step_index": step_index,
                            "loop_number": int(step["loop_number"]),
                        },
                    )
                    input_json = dict(input_json)
                    input_json["orchestration_resume"] = serialized_resume
                    _maybe_call(service, "update_run_input_json", run_id, input_json)
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.review_stream_blocked",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(blocked_payload.get("review_output") or ""),
                            "chunk_index": int(blocked_payload.get("chunk_index") or 0),
                            "check_count": int(blocked_payload.get("check_count") or 0),
                            "segment_start_char": int(blocked_payload.get("segment_start_char") or 0),
                            "segment_end_char": int(blocked_payload.get("segment_end_char") or 0),
                            "last_reviewed_char": int(blocked_payload.get("last_reviewed_char") or 0),
                            "partial_length": len(partial_content),
                        },
                    )
                    review_payload = {
                        "agent_id": agent_output_key,
                        "agent_name": agent_name,
                        "review_output": str(blocked_payload.get("review_output") or ""),
                        "step_index": step_index,
                        "loop_number": int(step["loop_number"]),
                        "review_stage": "agent_output_stream",
                        "check_count": int(blocked_payload.get("check_count") or 0),
                        "segment_index": int(blocked_payload.get("segment_index") or 0),
                        "segment_count": int(blocked_payload.get("segment_count") or 0),
                        "segment_start_char": int(blocked_payload.get("segment_start_char") or 0),
                        "segment_end_char": int(blocked_payload.get("segment_end_char") or 0),
                        "last_reviewed_char": int(blocked_payload.get("last_reviewed_char") or 0),
                        "segment_preview": str(blocked_payload.get("segment_preview") or ""),
                        "partial_output": partial_content[-1000:],
                    }
                    review_payload = _attach_approval_artifact_snapshot(
                        review_payload,
                        agent_conf=agent_conf,
                        partial_output=partial_content,
                        state=previous_state,
                        artifact_source="partial_stream",
                    )
                    _record_auto_review_decision(
                        service,
                        run_id=run_id,
                        reason=f"Review agent blocked streaming output from {agent_name}",
                        payload_json=review_payload,
                        requested_by=user_id,
                        review_agent_name=str(review_agent.get("name") or "review_agent"),
                        comment=str(blocked_payload.get("review_output") or "") or None,
                    )
                    _record_review_notification(
                        service,
                        run_id=run_id,
                        verdict="rejected",
                        title=f"Review blocked output from {agent_name}",
                        message=str(blocked_payload.get("review_output") or "") or None,
                        reviewer=str(review_agent.get("name") or "review_agent"),
                        review_stage="agent_output_stream",
                        agent_id=agent_output_key,
                        agent_name=agent_name,
                    )
                    _maybe_call(
                        service,
                        "patch_runtime_state",
                        run_id,
                        review={
                            "stage": "agent_output_stream",
                            "status": "rejected",
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(blocked_payload.get("review_output") or ""),
                            "check_count": int(blocked_payload.get("check_count") or 0),
                            "segment_index": int(blocked_payload.get("segment_index") or 0),
                            "segment_count": int(blocked_payload.get("segment_count") or 0),
                            "segment_start_char": int(blocked_payload.get("segment_start_char") or 0),
                            "segment_end_char": int(blocked_payload.get("segment_end_char") or 0),
                            "last_reviewed_char": int(blocked_payload.get("last_reviewed_char") or 0),
                        },
                        continuation={
                            "enabled": True,
                            "mode": "continue_with_partial_stream_output",
                            "status": "rejected",
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "step_index": int(step_index),
                            "prefix_length": len(partial_content),
                        },
                    )
                    return _finalize_auto_review_rejection(
                        service,
                        verification_service,
                        run_id=run_id,
                        active_agent_ids=active_agent_ids,
                        loop_count=loop_count,
                        review_agent_enabled=review_agent_enabled,
                        error_code="review_rejected",
                        error_message=f"Review agent blocked streaming output from {agent_name}",
                        review_details=review_payload,
                        agent_outputs=dict(previous_state.get("agent_outputs") or {}),
                        output_artifacts=dict(previous_state.get("output_artifacts") or {}),
                        recovery_mode=recovery_mode,
                        capability_snapshot=capability_snapshot,
                        selected_agent_ids=selected_agent_ids,
                    )
                agent_output_key = str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or "")
                agent_name = str(agent_conf.get("cluster_name") or agent_conf.get("name") or agent_output_key)
                if stream_continuation_pending:
                    continuation = final_state.get("continuation") if isinstance(final_state, dict) else None
                    continuation_resume_agent_id = str(
                        (continuation or {}).get("resume_agent_id")
                        or (continuation or {}).get("agent_id")
                        or ""
                    )
                    if continuation_resume_agent_id and continuation_resume_agent_id == str(agent_conf.get("agent_id") or agent_output_key):
                        final_state.pop("continuation", None)
                        stream_continuation_pending = False
                        input_json = dict(input_json)
                        input_json.pop("orchestration_resume", None)
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.stream_continuation_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                            },
                        )
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            continuation={
                                "enabled": True,
                                "status": "completed",
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "step_index": int(step_index),
                                "completed_at": int(time.time() * 1000),
                            },
                        )
                latest_output = str((final_state.get("agent_outputs") or {}).get(agent_output_key) or "")
                agent_artifact = dict((final_state.get("output_artifacts") or {}).get(agent_output_key) or {})
                if (
                    str(agent_conf.get("cluster_strategy") or "") == "brainstorm"
                    and bool(agent_conf.get("cluster_auto_research", False))
                    and agent_artifact
                ):
                    research_payload: dict[str, Any]
                    try:
                        research_payload = await _run_cluster_auto_research(
                            task=task_str,
                            artifact=agent_artifact,
                        )
                    except Exception as exc:
                        research_payload = {
                            "queries": _build_cluster_research_queries(task=task_str, artifact=agent_artifact),
                            "result_count": 0,
                            "searches": [],
                            "digest": "",
                            "error": str(exc),
                        }
                    memory_status = _persist_cluster_research_memory(
                        user_id=user_id,
                        run_id=run_id,
                        cluster_id=agent_output_key,
                        research_digest=str(research_payload.get("digest") or ""),
                    )
                    research_payload["memory"] = memory_status
                    updated_artifacts = dict(final_state.get("output_artifacts") or {})
                    current_cluster_artifact = dict(updated_artifacts.get(agent_output_key) or {})
                    current_cluster_artifact["research"] = research_payload
                    updated_artifacts[agent_output_key] = current_cluster_artifact
                    final_state["output_artifacts"] = updated_artifacts
                    if str(research_payload.get("digest") or "").strip():
                        updated_outputs = dict(final_state.get("agent_outputs") or {})
                        updated_outputs[agent_output_key] = (
                            f"{latest_output}\n\nResearch Digest:\n{str(research_payload.get('digest') or '').strip()}"
                        ).strip()
                        final_state["agent_outputs"] = updated_outputs
                        latest_output = str(updated_outputs.get(agent_output_key) or latest_output)
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.cluster_research_completed",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "queries": list(research_payload.get("queries") or []),
                            "result_count": int(research_payload.get("result_count") or 0),
                            "paper_count": len(list(research_payload.get("papers") or [])),
                            "browser_preview_count": len(list(research_payload.get("browser_previews") or [])),
                            "memory_stored": bool(memory_status.get("stored")),
                            "error": str(research_payload.get("error") or "") or None,
                        },
                    )
                    current_runtime_state = _maybe_call(service, "_load_persisted_runtime_state", run_id) or {}
                    current_cluster_ids = list(
                        ((current_runtime_state.get("research") or {}) if isinstance(current_runtime_state, dict) else {}).get("cluster_ids") or []
                    )
                    _maybe_call(
                        service,
                        "patch_runtime_state",
                        run_id,
                        research={
                            "enabled": True,
                            "mode": str(research_payload.get("research_mode") or "paper_first"),
                            "paper_count": len(list(research_payload.get("papers") or [])),
                            "browser_preview_count": len(list(research_payload.get("browser_previews") or [])),
                            "source_count": len(list(research_payload.get("sources") or [])),
                            "cluster_ids": sorted({*current_cluster_ids, agent_output_key}),
                        },
                    )
                    if review_enabled and (
                        str(research_payload.get("digest") or "").strip()
                        or list(research_payload.get("papers") or [])
                        or list(research_payload.get("latest_progress") or [])
                    ):
                        research_review_result = await _review_cluster_research_evidence(
                            review_config=review_config,
                            provider_config=provider_config,
                            default_timeout=default_timeout,
                            provider_registry=provider_registry,
                            task=task_str,
                            agent_id=agent_output_key,
                            agent_name=agent_name,
                            research_payload=research_payload,
                        )
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.cluster_research_review_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "approved": bool(research_review_result.get("approved")),
                                "review_output": str(research_review_result.get("review_output") or ""),
                            },
                        )
                        if not bool(research_review_result.get("approved")):
                            rollback_state = _serialize_orchestration_resume_state(previous_state, step_index)["state"]
                            rollback_artifacts = dict(rollback_state.get("output_artifacts") or {})
                            rollback_artifacts[agent_output_key] = {
                                **agent_artifact,
                                "research": {
                                    **research_payload,
                                    "blocked": True,
                                    "review_output": str(research_review_result.get("review_output") or ""),
                                },
                            }
                            rollback_outputs = dict(rollback_state.get("agent_outputs") or {})
                            rollback_outputs[agent_output_key] = latest_output
                            rollback_state["output_artifacts"] = rollback_artifacts
                            rollback_state["agent_outputs"] = rollback_outputs
                            serialized_resume = _build_review_blocked_resume_payload(
                                state=final_state,
                                next_step_index=step_index + 1,
                                rollback_state=rollback_state,
                            )
                            input_json = dict(input_json)
                            input_json["orchestration_resume"] = serialized_resume
                            _maybe_call(service, "update_run_input_json", run_id, input_json)
                            review_payload = {
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(research_review_result.get("review_output") or ""),
                                "step_index": step_index,
                                "loop_number": int(step["loop_number"]),
                                "review_stage": "cluster_research",
                                "research_queries": list(research_payload.get("queries") or []),
                            }
                            review_payload = _attach_approval_artifact_snapshot(
                                review_payload,
                                artifact=dict(rollback_artifacts.get(agent_output_key) or {}),
                                artifact_source="research_artifact",
                            )
                            _record_auto_review_decision(
                                service,
                                run_id=run_id,
                                reason=f"Review agent blocked research evidence from {agent_name}",
                                payload_json=review_payload,
                                requested_by=user_id,
                                review_agent_name=str(review_agent.get("name") or "review_agent"),
                                comment=str(research_review_result.get("review_output") or "") or None,
                            )
                            _record_review_notification(
                                service,
                                run_id=run_id,
                                verdict="rejected",
                                title=f"Review blocked research evidence from {agent_name}",
                                message=str(research_review_result.get("review_output") or "") or None,
                                reviewer=str(review_agent.get("name") or "review_agent"),
                                review_stage="cluster_research",
                                agent_id=agent_output_key,
                                agent_name=agent_name,
                            )
                            _maybe_call(
                                service,
                                "patch_runtime_state",
                                run_id,
                                review={
                                    "stage": "cluster_research",
                                    "status": "rejected",
                                    "agent_id": agent_output_key,
                                    "agent_name": agent_name,
                                    "review_output": str(research_review_result.get("review_output") or ""),
                                },
                            )
                            return _finalize_auto_review_rejection(
                                service,
                                verification_service,
                                run_id=run_id,
                                active_agent_ids=active_agent_ids,
                                loop_count=loop_count,
                                review_agent_enabled=review_agent_enabled,
                                error_code="review_rejected",
                                error_message=f"Review agent blocked research evidence from {agent_name}",
                                review_details=review_payload,
                                agent_outputs=dict(rollback_outputs),
                                output_artifacts=dict(rollback_artifacts),
                                recovery_mode=recovery_mode,
                                selected_agent_ids=selected_agent_ids,
                            )
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.step_completed",
                    details={
                        "agent_id": agent_output_key,
                        "agent_name": agent_name,
                        "step_index": step_index,
                        "loop_number": int(step["loop_number"]),
                    },
                )
                runtime_step_event = build_runtime_step_completed_event(
                    run_id=run_id,
                    step_index=step_index,
                    agent_id=agent_output_key,
                    agent_name=agent_name,
                    loop_number=int(step["loop_number"]),
                )
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type=runtime_step_event.event_type,
                    details=dict(runtime_step_event.payload),
                )

                if review_enabled and latest_output:
                    segmented_review = await _run_segmented_output_review(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_id=agent_output_key,
                        agent_name=agent_name,
                        output=latest_output,
                    )
                    if segmented_review["mode"] == "segmented":
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.review_segment_scan_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "approved": bool(segmented_review.get("approved")),
                                "segment_count": int(segmented_review.get("segment_count") or 0),
                                "segments_reviewed": int(segmented_review.get("segments_reviewed") or 0),
                                "blocked_segment_index": (
                                    _coerce_non_negative_int(
                                        cast(dict[str, Any], segmented_review.get("blocked_segment")).get("segment_index")
                                    )
                                    if isinstance(segmented_review.get("blocked_segment"), dict)
                                    else None
                                ),
                            },
                        )
                    if not bool(segmented_review.get("approved")):
                        blocked_segment = dict(segmented_review.get("blocked_segment") or {})
                        serialized_resume = _build_review_blocked_resume_payload(
                            state=final_state,
                            next_step_index=step_index + 1,
                            rollback_state=_serialize_orchestration_resume_state(previous_state, step_index)["state"],
                        )
                        input_json = dict(input_json)
                        input_json["orchestration_resume"] = serialized_resume
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        review_payload = {
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(blocked_segment.get("review_output") or ""),
                            "step_index": step_index,
                            "loop_number": int(step["loop_number"]),
                            "review_stage": "agent_output_segment",
                            "check_count": int(segmented_review.get("segments_reviewed") or 0),
                            "segment_index": int(blocked_segment.get("segment_index") or 0),
                            "segment_count": int(segmented_review.get("segment_count") or 0),
                            "segment_start_char": int(blocked_segment.get("start_char") or 0),
                            "segment_end_char": int(blocked_segment.get("end_char") or 0),
                            "last_reviewed_char": int(blocked_segment.get("end_char") or 0),
                            "segment_preview": str(blocked_segment.get("content") or "")[:400],
                        }
                        review_payload = _attach_approval_artifact_snapshot(
                            review_payload,
                            artifact=agent_artifact,
                            artifact_source="output_artifact",
                        )
                        _record_auto_review_decision(
                            service,
                            run_id=run_id,
                            reason=f"Review agent blocked streamed output from {agent_name}",
                            payload_json=review_payload,
                            requested_by=user_id,
                            review_agent_name=str(review_agent.get("name") or "review_agent"),
                            comment=str(blocked_segment.get("review_output") or "") or None,
                        )
                        _record_review_notification(
                            service,
                            run_id=run_id,
                            verdict="rejected",
                            title=f"Review blocked an output segment from {agent_name}",
                            message=str(blocked_segment.get("review_output") or "") or None,
                            reviewer=str(review_agent.get("name") or "review_agent"),
                            review_stage="agent_output_segment",
                            agent_id=agent_output_key,
                            agent_name=agent_name,
                        )
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            review={
                                "stage": "agent_output_segment",
                                "status": "rejected",
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(blocked_segment.get("review_output") or ""),
                                "check_count": int(segmented_review.get("segments_reviewed") or 0),
                                "segment_index": int(blocked_segment.get("segment_index") or 0),
                                "segment_count": int(segmented_review.get("segment_count") or 0),
                                "segment_start_char": int(blocked_segment.get("start_char") or 0),
                                "segment_end_char": int(blocked_segment.get("end_char") or 0),
                                "last_reviewed_char": int(blocked_segment.get("end_char") or 0),
                            },
                        )
                        return _finalize_auto_review_rejection(
                            service,
                            verification_service,
                            run_id=run_id,
                            active_agent_ids=active_agent_ids,
                            loop_count=loop_count,
                            review_agent_enabled=review_agent_enabled,
                            error_code="review_rejected",
                            error_message=f"Review agent blocked streamed output from {agent_name}",
                            review_details=review_payload,
                            agent_outputs=dict(previous_state.get("agent_outputs") or {}),
                            output_artifacts=dict(previous_state.get("output_artifacts") or {}),
                            recovery_mode=recovery_mode,
                            capability_snapshot=capability_snapshot,
                            selected_agent_ids=selected_agent_ids,
                        )

                    review_result = await review_orchestration_output(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_id=agent_output_key,
                        agent_name=agent_name,
                        output=latest_output,
                    )
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.review_completed",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "approved": bool(review_result.get("approved")),
                            "review_output": str(review_result.get("review_output") or ""),
                        },
                    )
                    if not bool(review_result.get("approved")):
                        serialized_resume = _build_review_blocked_resume_payload(
                            state=final_state,
                            next_step_index=step_index + 1,
                            rollback_state=_serialize_orchestration_resume_state(previous_state, step_index)["state"],
                        )
                        input_json = dict(input_json)
                        input_json["orchestration_resume"] = serialized_resume
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        review_payload = {
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(review_result.get("review_output") or ""),
                            "step_index": step_index,
                            "loop_number": int(step["loop_number"]),
                            "review_stage": "agent_output_final",
                        }
                        review_payload = _attach_approval_artifact_snapshot(
                            review_payload,
                            artifact=agent_artifact,
                            artifact_source="output_artifact",
                        )
                        _record_auto_review_decision(
                            service,
                            run_id=run_id,
                            reason=f"Review agent blocked output from {agent_name}",
                            payload_json=review_payload,
                            requested_by=user_id,
                            review_agent_name=str(review_agent.get("name") or "review_agent"),
                            comment=str(review_result.get("review_output") or "") or None,
                        )
                        _record_review_notification(
                            service,
                            run_id=run_id,
                            verdict="rejected",
                            title=f"Review blocked output from {agent_name}",
                            message=str(review_result.get("review_output") or "") or None,
                            reviewer=str(review_agent.get("name") or "review_agent"),
                            review_stage="agent_output_final",
                            agent_id=agent_output_key,
                            agent_name=agent_name,
                        )
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            review={
                                "stage": "agent_output_final",
                                "status": "rejected",
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(review_result.get("review_output") or ""),
                            },
                        )
                        return _finalize_auto_review_rejection(
                            service,
                            verification_service,
                            run_id=run_id,
                            active_agent_ids=active_agent_ids,
                            loop_count=loop_count,
                            review_agent_enabled=review_agent_enabled,
                            error_code="review_rejected",
                            error_message=f"Review agent blocked output from {agent_name}",
                            review_details=review_payload,
                            agent_outputs=dict(final_state.get("agent_outputs") or {}),
                            output_artifacts=dict(final_state.get("output_artifacts") or {}),
                            recovery_mode=recovery_mode,
                            capability_snapshot=capability_snapshot,
                            selected_agent_ids=selected_agent_ids,
                        )

            agent_outputs = final_state.get("agent_outputs", {})
            output_artifacts = final_state.get("output_artifacts", {})
            errors = final_state.get("errors", [])

            _maybe_call(
                service,
                "record_event",
                run_id,
                event_type="orchestration.completed",
                details={
                    "agent_outputs": agent_outputs,
                    "output_artifacts": output_artifacts,
                    "errors": errors,
                    "review_agent_enabled": review_agent_enabled,
                    "recovery_mode": recovery_mode,
                },
            )
            runtime_completed_event = build_runtime_completed_event(
                run_id=run_id,
                agent_outputs=agent_outputs,
                output_artifacts=output_artifacts,
                errors=list(errors),
                review_agent_enabled=review_agent_enabled,
                recovery_mode=recovery_mode,
            )
            _maybe_call(
                service,
                "record_event",
                run_id,
                event_type=runtime_completed_event.event_type,
                details=dict(runtime_completed_event.payload),
            )
            _record_review_notification(
                service,
                run_id=run_id,
                verdict="approved" if not bool(errors) else "failed",
                title="Run ready for follow-up",
                message=(
                    "Review cleared this orchestration run."
                    if not bool(errors)
                    else "; ".join(errors) or "The orchestration run finished with execution errors."
                ),
                reviewer=str(review_agent.get("name") or "review_agent") if review_agent_enabled else "system",
            )

            verification = verification_service.build_agent_orchestration_result(
                ok=not bool(errors),
                active_agent_ids=active_agent_ids,
                blocked_agents=[],
                loop_count=loop_count,
                review_agent_enabled=review_agent_enabled,
                error_code="execution_errors" if errors else None,
                error_message="; ".join(errors) if errors else None,
                agent_outputs=agent_outputs,
                output_artifacts=output_artifacts,
                recovery_mode=recovery_mode,
                capability_snapshot=capability_snapshot,
                handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
                selected_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
            )
            service.complete_with_verification(run_id, verification)
            return not bool(errors)

        if task_type == HarnessTaskType.SESSION_RESUME_APPROVAL.value:
            _maybe_call(service, "set_current_step", run_id, "load_checkpoint")
            _accept_runtime_command_for_run(service, run_id=run_id, task_type=task_type)
            session_id = str(run.get("session_id") or "") or None
            if not session_id:
                verification = verification_service.build_session_resume_result(
                    ok=False,
                    session_id=None,
                    interrupted=None,
                    error_code="missing_session_id",
                    error_message="missing session_id for session resume execution",
                )
                service.complete_with_verification(run_id, verification)
                return False

            checkpoint = await CheckpointAdapter().load(session_id)
            if checkpoint is None:
                verification = verification_service.build_session_resume_result(
                    ok=False,
                    session_id=session_id,
                    interrupted=None,
                    error_code="checkpoint_missing",
                    error_message="resume checkpoint not found",
                )
                service.complete_with_verification(run_id, verification)
                return False

            _maybe_call(service, "set_current_step", run_id, "resume_graph")
            _maybe_call(service, "mark_resumed", run_id)
            resume_result = await GraphResumeService().resume_approved_session(
                session_id=session_id,
                checkpoint=checkpoint,
            )
            ok = bool(resume_result.get("ok"))
            messages = resume_result.get("messages")
            normalized_messages = messages if isinstance(messages, list) else []
            user_id = str(run.get("user_id") or "") or None
            if ok and user_id and normalized_messages:
                persist_session_messages(
                    user_id=user_id,
                    session_id=session_id,
                    messages=normalized_messages,
                )
            interrupted_value = resume_result.get("interrupted") if isinstance(resume_result, dict) else None
            verification = verification_service.build_session_resume_result(
                ok=ok,
                session_id=session_id,
                interrupted=interrupted_value if isinstance(interrupted_value, bool) else None,
                error_code=str(resume_result.get("error_code") or "") or None,
                error_message=str(resume_result.get("error_message") or "") or None,
            )
            service.complete_with_verification(run_id, verification)
            return ok

        verification = verification_service.build_document_ingest_result(
            ok=False,
            stage="unsupported_task_type",
            error_code="unsupported_task_type",
            error_message="unsupported harness task type",
        )
        service.complete_with_verification(run_id, verification)
        return False
    except Exception as exc:
        if task_type == HarnessTaskType.SESSION_RESUME_APPROVAL.value:
            verification = verification_service.build_session_resume_result(
                ok=False,
                session_id=str(run.get("session_id") or "") or None,
                interrupted=None,
                error_code="task_exception",
                error_message=str(exc),
            )
        elif task_type == HarnessTaskType.AGENT_ORCHESTRATION.value:
            input_json = cast(dict[str, Any], run.get("input_json") or {})
            selected_agent_ids = input_json.get("selected_agent_ids") if isinstance(input_json, dict) else []
            selected_agent_ids = selected_agent_ids if isinstance(selected_agent_ids, list) else []
            graph = input_json.get("graph") if isinstance(input_json, dict) else {}
            graph = graph if isinstance(graph, dict) else {}
            review_agent = cast(
                dict[str, Any],
                graph.get("review_agent") if isinstance(graph.get("review_agent"), dict) else {},
            )
            verification = verification_service.build_agent_orchestration_result(
                ok=False,
                active_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
                blocked_agents=[],
                loop_count=int(input_json.get("loop_count") or 1) if isinstance(input_json, dict) else 1,
                review_agent_enabled=bool(review_agent.get("enabled", True)),
                error_code="task_exception",
                error_message=str(exc),
                capability_snapshot=_build_capability_snapshot(
                    graph,
                    active_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
                    handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
                ),
                handoff_scope="selected_agents" if selected_agent_ids else "all_agents",
                selected_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
            )
        else:
            verification = verification_service.build_document_ingest_result(
                ok=False,
                stage="exception",
                error_code="task_exception",
                error_message=str(exc),
            )
        service.complete_with_verification(run_id, verification)
        return False
    finally:
        if execution_lock_acquired:
            await release_task_operation(
                execution_lock_key,
                expected_task_id=execution_lock_owner,
            )


async def resume_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    return await run_harness_task(ctx, run_id)
