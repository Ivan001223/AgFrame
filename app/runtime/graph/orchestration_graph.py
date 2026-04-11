from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict, deque
from collections.abc import Callable
from inspect import isawaitable
from typing import Annotated, Any, Awaitable, NotRequired, TypedDict, cast

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.runtime.llm.llm_factory import get_llm_for_provider
from app.runtime.llm.provider_registry import (
    ModelProviderRegistry,
    get_provider_registry,
    infer_tool_calling_support,
)
from app.runtime.llm.stream_adapter import coerce_stream_text, stream_llm_events

_log = logging.getLogger("runtime.graph.orchestration")

FIRST_PRINCIPLES_DIRECTIVE = (
    "Non-negotiable operating rule: always reason from first principles. "
    "Reduce the task to fundamental constraints, verify assumptions explicitly, "
    "and build conclusions from the most basic truths available rather than habit, imitation, or unsupported analogy."
)

GAME_THEORY_DIRECTIVE = (
    "For brainstorm clusters, also reason through a game-theoretic lens. "
    "Model the relevant actors, incentives, payoffs, information asymmetries, commitments, cooperation failures, "
    "competitive responses, and likely equilibrium outcomes before choosing a strategy."
)

READ_ONLY_TOOL_IDS = frozenset(
    {
        "web_search",
        "calculator",
        "knowledge_retriever",
        "read_document",
        "get_current_time",
    }
)


def _merge_string(left: str, right: str) -> str:
    return str(right or left or "")


def _merge_string_dict(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    merged = dict(left or {})
    merged.update(dict(right or {}))
    return merged


def _merge_artifact_dict(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(left or {})
    merged.update(dict(right or {}))
    return merged


def _merge_error_list(left: list[str], right: list[str]) -> list[str]:
    return [*list(left or []), *list(right or [])]


class OrchestrationState(TypedDict):
    """
    Studio canvas orchestration multi-agent shared state.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    agent_outputs: Annotated[dict[str, str], _merge_string_dict]
    output_artifacts: Annotated[dict[str, dict[str, Any]], _merge_artifact_dict]
    current_agent: Annotated[str, _merge_string]
    loop_index: int
    errors: Annotated[list[str], _merge_error_list]
    knowledge_base_ids: NotRequired[list[str]]
    knowledge_context: NotRequired[str]
    continuation: NotRequired[dict[str, Any]]


class OutputGuardrailTrip(Exception):
    def __init__(self, *, payload: dict[str, Any], partial_content: str):
        super().__init__(str(payload.get("review_output") or "stream output blocked"))
        self.payload = payload
        self.partial_content = partial_content


def _coerce_timeout(value: Any, default_timeout: int) -> int:
    try:
        return int(value) if value is not None else default_timeout
    except (TypeError, ValueError):
        return default_timeout


def _message_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _trim_tool_payload(value: Any, *, max_chars: int = 4000) -> str:
    content = str(value or "").strip()
    if len(content) <= max_chars:
        return content
    return content[: max(max_chars - 3, 1)].rstrip() + "..."


def _build_tool_catalog_lookup(graph_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _normalize_skill_key(str(item.get("tool_id") or "")): dict(item)
        for item in graph_json.get("tool_catalog") or []
        if isinstance(item, dict) and _normalize_skill_key(str(item.get("tool_id") or ""))
    }


def _build_mcp_catalog_lookup(graph_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _normalize_skill_key(str(item.get("server_id") or "")): dict(item)
        for item in graph_json.get("mcp_server_catalog") or []
        if isinstance(item, dict) and _normalize_skill_key(str(item.get("server_id") or ""))
    }


def _build_tool_guidance_block(
    *,
    enabled_tool_ids: list[str],
    tool_catalog_by_id: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    for tool_id in enabled_tool_ids:
        item = tool_catalog_by_id.get(_normalize_skill_key(tool_id), {})
        title = str(item.get("title") or _humanize_identifier(tool_id)).strip()
        description = str(item.get("description") or "").strip()
        line = f"- {title}"
        if description:
            line += f": {description}"
        lines.append(line)
    return "\n".join(lines)


def _build_mcp_guidance_block(
    *,
    mcp_server_ids: list[str],
    missing_mcp_server_ids: list[str],
    mcp_catalog_by_id: dict[str, dict[str, Any]],
) -> str:
    sections: list[str] = []

    available_lines: list[str] = []
    for server_id in mcp_server_ids:
        item = mcp_catalog_by_id.get(_normalize_skill_key(server_id), {})
        title = str(item.get("title") or _humanize_identifier(server_id)).strip()
        description = str(item.get("description") or "").strip()
        line = f"- {title}"
        if description:
            line += f": {description}"
        available_lines.append(line)
    if available_lines:
        sections.append("Configured MCP servers relevant to this node:\n" + "\n".join(available_lines))

    missing_lines: list[str] = []
    for server_id in missing_mcp_server_ids:
        item = mcp_catalog_by_id.get(_normalize_skill_key(server_id), {})
        title = str(item.get("title") or _humanize_identifier(server_id)).strip()
        description = str(item.get("description") or "").strip()
        line = f"- {title}"
        if description:
            line += f": {description}"
        missing_lines.append(line)
    if missing_lines:
        sections.append("Relevant MCP servers not currently enabled in project inventory:\n" + "\n".join(missing_lines))

    return "\n\n".join(sections)


def _build_capability_execution_contract(summary: dict[str, Any]) -> str:
    approved_skill_ids = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
    executable_tool_ids = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
    planning_only_tool_ids = [
        str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
    ]
    disabled_tool_ids = [str(item) for item in summary.get("disabled_tool_ids") or [] if str(item).strip()]
    planning_only_mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
    missing_mcp_server_ids = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
    requires_tool_calling = bool(summary.get("requires_tool_calling", False))
    tool_execution_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"
    tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()
    if tool_execution_support == "unsupported" and executable_tool_ids:
        planning_only_tool_ids = list(dict.fromkeys([*planning_only_tool_ids, *executable_tool_ids]))
        executable_tool_ids = []

    lines: list[str] = []
    lines.append(
        "Approved skill packs are guidance-only: "
        + (", ".join(approved_skill_ids) if approved_skill_ids else "none")
        + "."
    )
    lines.append(
        "Directly executable tools in this runtime: "
        + (", ".join(executable_tool_ids) if executable_tool_ids else "none")
        + "."
    )
    if planning_only_tool_ids:
        lines.append(
            "Planning-only tools visible in policy/capability metadata but not executable right now: "
            + ", ".join(planning_only_tool_ids)
            + "."
        )
    if disabled_tool_ids:
        lines.append(
            "Disabled tools that stay unavailable until feature flags change: "
            + ", ".join(disabled_tool_ids)
            + "."
        )
    lines.append(
        "MCP inventory is planning metadata only in this runtime: "
        + (", ".join(planning_only_mcp_server_ids) if planning_only_mcp_server_ids else "none")
        + "."
    )
    if missing_mcp_server_ids:
        lines.append(
            "Relevant MCP inventory gaps: " + ", ".join(missing_mcp_server_ids) + "."
        )
    if requires_tool_calling:
        requirement_line = "This node expects direct tool-calling support."
        if tool_execution_support != "supported":
            requirement_line = (
                "This node expects direct tool-calling support, but the current runtime/provider route does not fully satisfy that contract."
            )
        if tool_execution_support_reason:
            requirement_line += f" Reason: {tool_execution_support_reason}."
        lines.append(requirement_line)
    return "\n".join(lines)


def _resolve_enabled_tools(enabled_tool_ids: list[str]) -> list[Any]:
    if not enabled_tool_ids:
        return []
    from app.skills.common.tools import ALL_TOOLS

    tool_by_id = {
        _normalize_skill_key(str(getattr(tool, "name", "") or "")): tool
        for tool in ALL_TOOLS
        if _normalize_skill_key(str(getattr(tool, "name", "") or ""))
    }
    return [
        tool_by_id[normalized_tool_id]
        for normalized_tool_id in [
            _normalize_skill_key(tool_id)
            for tool_id in enabled_tool_ids
            if _normalize_skill_key(tool_id) in tool_by_id
        ]
    ]


async def _invoke_bound_tool(tool: Any, args: Any) -> str:
    if hasattr(tool, "ainvoke"):
        return str(await tool.ainvoke(args))
    if hasattr(tool, "invoke"):
        return str(tool.invoke(args))
    if callable(tool):
        result = tool(args)
        if isawaitable(result):
            result = await result
        return str(result)
    raise TypeError(f"Unsupported tool type for {tool!r}")


def _tool_is_read_only(tool_name: str) -> bool:
    return _normalize_skill_key(tool_name) in READ_ONLY_TOOL_IDS


async def _execute_tool_call(
    raw_tool_call: dict[str, Any],
    *,
    tool_by_name: dict[str, Any],
    run_index: int,
) -> dict[str, Any]:
    tool_name = _normalize_skill_key(str(raw_tool_call.get("name") or ""))
    tool_call_id = str(raw_tool_call.get("id") or f"tool_call_{run_index + 1}")
    tool_args = raw_tool_call.get("args") or {}
    selected_tool = tool_by_name.get(tool_name)
    status = "success"
    if selected_tool is None:
        tool_result = f"Tool unavailable: {tool_name or 'unknown'}"
        status = "error"
    else:
        try:
            tool_result = await _invoke_bound_tool(selected_tool, tool_args)
        except Exception as exc:
            tool_result = f"Tool execution failed: {exc!s}"
            status = "error"

    trimmed_tool_result = _trim_tool_payload(tool_result, max_chars=4000)
    return {
        "tool_name": tool_name or "unknown_tool",
        "tool_call_id": tool_call_id,
        "status": status,
        "tool_message_content": trimmed_tool_result,
        "run_record": {
            "tool_id": tool_name or "unknown_tool",
            "call_id": tool_call_id,
            "status": status,
            "args_preview": _trim_tool_payload(tool_args, max_chars=240),
            "result_preview": _trim_tool_payload(tool_result, max_chars=400),
            "result_char_count": len(str(tool_result or "")),
        },
    }


async def _invoke_llm_with_tool_loop(
    llm: Any,
    messages: list[BaseMessage],
    *,
    enabled_tools: list[Any],
    timeout_seconds: int,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
    max_rounds: int = 4,
) -> tuple[str, list[dict[str, Any]], str]:
    if not enabled_tools:
        content, mode = await _invoke_llm_with_streaming_fallback(
            llm,
            messages,
            on_stream_chunk=on_stream_chunk,
            on_stream_event=on_stream_event,
        )
        return content, [], mode

    tool_by_name = {
        _normalize_skill_key(str(getattr(tool, "name", "") or "")): tool
        for tool in enabled_tools
        if _normalize_skill_key(str(getattr(tool, "name", "") or ""))
    }
    bound_llm = llm
    bind_tools = getattr(llm, "bind_tools", None)
    if callable(bind_tools):
        try:
            maybe_bound = bind_tools(list(tool_by_name.values()))
            if maybe_bound is not None:
                bound_llm = maybe_bound
        except Exception:
            bound_llm = llm

    conversation = list(messages)
    tool_runs: list[dict[str, Any]] = []
    last_ai_message: AIMessage | None = None

    for round_index in range(max(max_rounds, 1)):
        with anyio.fail_after(timeout_seconds):
            response = await bound_llm.ainvoke(conversation)
        ai_message = response if isinstance(response, AIMessage) else AIMessage(content=_message_text(response))
        last_ai_message = ai_message
        tool_calls = list(getattr(ai_message, "tool_calls", []) or [])
        if not tool_calls:
            return _message_text(ai_message).strip(), tool_runs, "tool_loop"

        conversation.append(ai_message)
        tool_results_by_index: dict[int, dict[str, Any]] = {}
        pending_read_only_calls: list[tuple[int, dict[str, Any]]] = []

        async def flush_read_only_batch() -> None:
            nonlocal pending_read_only_calls
            if not pending_read_only_calls:
                return
            indexes, batch = zip(*pending_read_only_calls, strict=False)
            results = await asyncio.gather(
                *[
                    _execute_tool_call(
                        raw_tool_call,
                        tool_by_name=tool_by_name,
                        run_index=len(tool_runs) + batch_index,
                    )
                    for batch_index, raw_tool_call in pending_read_only_calls
                ]
            )
            for index, result in zip(indexes, results, strict=False):
                tool_results_by_index[index] = result
            pending_read_only_calls = []

        for index, raw_tool_call in enumerate(tool_calls):
            tool_name = _normalize_skill_key(str(raw_tool_call.get("name") or ""))
            if _tool_is_read_only(tool_name):
                pending_read_only_calls.append((index, raw_tool_call))
                continue

            await flush_read_only_batch()
            tool_results_by_index[index] = await _execute_tool_call(
                raw_tool_call,
                tool_by_name=tool_by_name,
                run_index=len(tool_runs) + index,
            )

        await flush_read_only_batch()

        for index, _ in enumerate(tool_calls):
            tool_result = tool_results_by_index[index]
            conversation.append(
                ToolMessage(
                    content=str(tool_result.get("tool_message_content") or ""),
                    tool_call_id=str(tool_result.get("tool_call_id") or ""),
                    name=str(tool_result.get("tool_name") or "") or None,
                    status=str(tool_result.get("status") or "success"),
                )
            )
            tool_runs.append(dict(tool_result.get("run_record") or {}))

    fallback_content = _message_text(last_ai_message).strip() if last_ai_message is not None else ""
    if fallback_content:
        return fallback_content, tool_runs, "tool_loop"
    return (
        "Tool execution stopped after reaching the maximum tool-call rounds. Provide a final answer without more tool calls.",
        tool_runs,
        "tool_loop",
    )


async def _invoke_llm_with_streaming_fallback(
    llm: Any,
    messages: list[BaseMessage],
    *,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[str, str]:
    stream_method = getattr(llm, "astream", None)
    stream_events_method = getattr(llm, "astream_events", None)
    if callable(stream_method) or callable(stream_events_method):
        collected: list[str] = []
        chunk_index = 0
        async for stream_event in stream_llm_events(llm, messages, on_event=on_stream_event):
            if str(stream_event.get("type") or "") != "token":
                continue
            text = coerce_stream_text(stream_event.get("text"))
            if text:
                collected.append(text)
                if on_stream_chunk is not None:
                    callback_result = on_stream_chunk(text, "".join(collected), chunk_index)
                    if isawaitable(callback_result):
                        callback_result = await callback_result
                    if isinstance(callback_result, dict) and bool(callback_result.get("blocked")):
                        partial_content = str(callback_result.get("partial_output") or "".join(collected))
                        raise OutputGuardrailTrip(
                            payload=callback_result,
                            partial_content=partial_content,
                        )
                chunk_index += 1
        content = "".join(collected).strip()
        if content:
            return content, "astream"

    response = await llm.ainvoke(messages)
    return str(response.content) if response.content else "", "ainvoke"


def _build_continuation_context(agent_id: str, state: OrchestrationState) -> dict[str, Any] | None:
    continuation = state.get("continuation")
    if not isinstance(continuation, dict):
        return None
    target_agent_id = str(continuation.get("resume_agent_id") or continuation.get("agent_id") or "").strip()
    if not target_agent_id or target_agent_id != agent_id:
        return None
    partial_output = str(continuation.get("partial_output") or "")
    if not partial_output.strip():
        return None
    return {
        "partial_output": partial_output,
        "review_output": str(continuation.get("review_output") or ""),
        "agent_name": str(continuation.get("agent_name") or agent_id),
        "loop_number": continuation.get("loop_number"),
    }


def _merge_continuation_output(prefix: str, generated: str) -> str:
    approved_prefix = str(prefix or "")
    generated_text = str(generated or "")
    if not approved_prefix:
        return generated_text
    if not generated_text:
        return approved_prefix
    if generated_text.startswith(approved_prefix):
        return generated_text

    max_overlap = min(len(approved_prefix), len(generated_text))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if approved_prefix.endswith(generated_text[:size]):
            overlap = size
            break
    return approved_prefix + generated_text[overlap:]


def _normalize_skill_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def _humanize_identifier(value: str) -> str:
    parts = [part for part in value.replace("-", "_").split("_") if part]
    return " ".join(part.capitalize() for part in parts) or value


def _build_team_capability_roster(
    *,
    agents: list[dict[str, Any]],
    capability_summary_by_agent_id: dict[str, dict[str, Any]],
) -> str:
    def resolve_readiness(summary: dict[str, Any]) -> tuple[str, list[str], list[str]]:
        loaded_skills = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
        missing_skills = [str(item) for item in summary.get("missing_skill_ids") or [] if str(item).strip()]
        configured_allowed_tools = [
            str(item) for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
        ]
        disabled_tools = [str(item) for item in summary.get("disabled_tool_ids") or [] if str(item).strip()]
        provider_limited_tools = [
            str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        ]
        configured_allowed_mcp = [
            str(item) for item in summary.get("configured_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        missing_mcp = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
        blockers = [str(item) for item in summary.get("readiness_blockers") or [] if str(item).strip()]
        warnings = [str(item) for item in summary.get("readiness_warnings") or [] if str(item).strip()]
        if not blockers and missing_skills:
            blockers = [
                "Missing approved skills before this node can run: " + ", ".join(missing_skills)
            ]
        has_explicit_capability_requirements = bool(
            loaded_skills or missing_skills or configured_allowed_tools or configured_allowed_mcp
        )
        if not warnings:
            if has_explicit_capability_requirements and missing_mcp:
                warnings.append("Relevant MCP servers are not enabled in project inventory: " + ", ".join(missing_mcp))
            if has_explicit_capability_requirements and provider_limited_tools:
                warnings.append(
                    "Current provider route cannot execute these tools directly: "
                    + ", ".join(provider_limited_tools)
                )
            if has_explicit_capability_requirements and disabled_tools:
                warnings.append(
                    "Some relevant tools stay disabled until feature flags change: "
                    + ", ".join(disabled_tools)
                )
        status = str(summary.get("readiness_status") or "").strip()
        if status not in {"ready", "limited", "blocked"}:
            status = "blocked" if blockers else "limited" if warnings else "ready"
        return status, blockers, warnings

    def resolve_summary(agent: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(agent.get("agent_id") or "").strip()
        if agent_id and agent_id in capability_summary_by_agent_id:
            return capability_summary_by_agent_id[agent_id]
        cluster_agent_id = str(agent.get("cluster_agent_id") or "").strip()
        if cluster_agent_id and cluster_agent_id in capability_summary_by_agent_id:
            return capability_summary_by_agent_id[cluster_agent_id]
        return {}

    lines: list[str] = []
    for agent in agents:
        agent_id = str(agent.get("agent_id") or "").strip()
        if not agent_id:
            continue
        summary = resolve_summary(agent)
        loaded_skills = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
        suggested_skills = [str(item) for item in summary.get("suggested_skill_ids") or [] if str(item).strip()]
        enabled_tools = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
        provider_limited_tools = [
            str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        ]
        configured_allowed_tools = [
            str(item) for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
        ]
        configured_denied_tools = [
            str(item) for item in summary.get("configured_denied_tool_ids") or [] if str(item).strip()
        ]
        mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
        missing_mcp_server_ids = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
        configured_allowed_mcp_server_ids = [
            str(item) for item in summary.get("configured_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        configured_denied_mcp_server_ids = [
            str(item) for item in summary.get("configured_denied_mcp_server_ids") or [] if str(item).strip()
        ]
        delegation_lane_ids = [str(item) for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
        delegation_focus = str(summary.get("delegation_focus") or "").strip()
        tool_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"
        readiness_status, readiness_blockers, readiness_warnings = resolve_readiness(summary)
        provider_route = str(summary.get("provider_route") or "project default").strip() or "project default"
        review_mode = str(summary.get("review_mode") or "direct handoff").strip() or "direct handoff"
        tool_policy_parts: list[str] = []
        mcp_policy_parts: list[str] = []
        if configured_allowed_tools:
            tool_policy_parts.append(f"allow {', '.join(configured_allowed_tools)}")
        if configured_denied_tools:
            tool_policy_parts.append(f"deny {', '.join(configured_denied_tools)}")
        if configured_allowed_mcp_server_ids:
            mcp_policy_parts.append(f"allow {', '.join(configured_allowed_mcp_server_ids)}")
        if configured_denied_mcp_server_ids:
            mcp_policy_parts.append(f"deny {', '.join(configured_denied_mcp_server_ids)}")
        lines.append(
            f"- {str(agent.get('name') or agent_id)} ({str(agent.get('role') or 'specialist')}): "
            f"skills={', '.join(loaded_skills) if loaded_skills else 'none'}; "
            f"suggested={', '.join(suggested_skills) if suggested_skills else 'none'}; "
            f"tools={', '.join(enabled_tools) if enabled_tools else 'none'}; "
            f"provider_limited_tools={', '.join(provider_limited_tools) if provider_limited_tools else 'none'}; "
            f"tool_policy={' / '.join(tool_policy_parts) if tool_policy_parts else 'inherit'}; "
            f"mcp={', '.join(mcp_server_ids) if mcp_server_ids else 'none'}; "
            f"mcp_gaps={', '.join(missing_mcp_server_ids) if missing_mcp_server_ids else 'none'}; "
            f"mcp_policy={' / '.join(mcp_policy_parts) if mcp_policy_parts else 'inherit'}; "
            f"lanes={', '.join(delegation_lane_ids) if delegation_lane_ids else 'generalist'}; "
            f"focus={delegation_focus or 'general synthesis'}; "
            f"readiness={readiness_status}; "
            f"blockers={', '.join(readiness_blockers) if readiness_blockers else 'none'}; "
            f"warnings={', '.join(readiness_warnings) if readiness_warnings else 'none'}; "
            f"tool_support={tool_support}; "
            f"provider={provider_route}; review={review_mode}."
        )
    return "\n".join(lines)


def _resolve_agent_capability_summary(
    *,
    agent: dict[str, Any],
    capability_summary_by_agent_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    agent_id = str(agent.get("agent_id") or "").strip()
    if agent_id and agent_id in capability_summary_by_agent_id:
        return capability_summary_by_agent_id[agent_id]
    cluster_agent_id = str(agent.get("cluster_agent_id") or "").strip()
    if cluster_agent_id and cluster_agent_id in capability_summary_by_agent_id:
        return capability_summary_by_agent_id[cluster_agent_id]
    return {}


def _format_capability_snapshot(summary: dict[str, Any]) -> str:
    loaded_skills = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
    required_skills = [str(item) for item in summary.get("required_skill_ids") or [] if str(item).strip()]
    missing_required_skills = [
        str(item) for item in summary.get("missing_required_skill_ids") or [] if str(item).strip()
    ]
    required_tools = [str(item) for item in summary.get("required_tool_ids") or [] if str(item).strip()]
    missing_required_tools = [
        str(item) for item in summary.get("missing_required_tool_ids") or [] if str(item).strip()
    ]
    enabled_tools = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
    requires_tool_calling = bool(summary.get("requires_tool_calling", False))
    provider_limited_tools = [str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()]
    configured_allowed_tools = [
        str(item) for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
    ]
    configured_denied_tools = [
        str(item) for item in summary.get("configured_denied_tool_ids") or [] if str(item).strip()
    ]
    required_mcp_server_ids = [
        str(item) for item in summary.get("required_mcp_server_ids") or [] if str(item).strip()
    ]
    missing_required_mcp_server_ids = [
        str(item) for item in summary.get("missing_required_mcp_server_ids") or [] if str(item).strip()
    ]
    mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
    missing_mcp_server_ids = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
    configured_allowed_mcp_server_ids = [
        str(item) for item in summary.get("configured_allowed_mcp_server_ids") or [] if str(item).strip()
    ]
    configured_denied_mcp_server_ids = [
        str(item) for item in summary.get("configured_denied_mcp_server_ids") or [] if str(item).strip()
    ]
    delegation_lane_ids = [str(item) for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
    delegation_focus = str(summary.get("delegation_focus") or "").strip()
    tool_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"
    availability_status = str(summary.get("availability_status") or "").strip()
    availability_blockers = [str(item) for item in summary.get("availability_blockers") or [] if str(item).strip()]
    availability_warnings = [str(item) for item in summary.get("availability_warnings") or [] if str(item).strip()]
    readiness_status = str(summary.get("readiness_status") or "").strip()
    readiness_blockers = [str(item) for item in summary.get("readiness_blockers") or [] if str(item).strip()]
    readiness_warnings = [str(item) for item in summary.get("readiness_warnings") or [] if str(item).strip()]
    if availability_status not in {"available", "limited", "unavailable"}:
        availability_status = "unavailable" if (
            missing_required_skills
            or missing_required_tools
            or missing_required_mcp_server_ids
            or (requires_tool_calling and tool_support == "unsupported")
        ) else "limited" if (requires_tool_calling and tool_support != "supported") else "available"
    if readiness_status not in {"ready", "limited", "blocked"}:
        readiness_status = "blocked" if summary.get("missing_skill_ids") else "limited" if (
            missing_mcp_server_ids or provider_limited_tools
        ) else "ready"
    provider_route = str(summary.get("provider_route") or "project default").strip() or "project default"
    review_mode = str(summary.get("review_mode") or "direct handoff").strip() or "direct handoff"
    tool_policy_parts: list[str] = []
    mcp_policy_parts: list[str] = []
    if configured_allowed_tools:
        tool_policy_parts.append(f"allow {', '.join(configured_allowed_tools)}")
    if configured_denied_tools:
        tool_policy_parts.append(f"deny {', '.join(configured_denied_tools)}")
    if configured_allowed_mcp_server_ids:
        mcp_policy_parts.append(f"allow {', '.join(configured_allowed_mcp_server_ids)}")
    if configured_denied_mcp_server_ids:
        mcp_policy_parts.append(f"deny {', '.join(configured_denied_mcp_server_ids)}")
    return (
        f"skills={', '.join(loaded_skills) if loaded_skills else 'none'}; "
        f"required_skills={', '.join(required_skills) if required_skills else 'none'}; "
        f"missing_required_skills={', '.join(missing_required_skills) if missing_required_skills else 'none'}; "
        f"required_tools={', '.join(required_tools) if required_tools else 'none'}; "
        f"missing_required_tools={', '.join(missing_required_tools) if missing_required_tools else 'none'}; "
        f"tools={', '.join(enabled_tools) if enabled_tools else 'none'}; "
        f"requires_tool_calling={'yes' if requires_tool_calling else 'no'}; "
        f"provider_limited_tools={', '.join(provider_limited_tools) if provider_limited_tools else 'none'}; "
        f"tool_policy={' / '.join(tool_policy_parts) if tool_policy_parts else 'inherit'}; "
        f"required_mcp={', '.join(required_mcp_server_ids) if required_mcp_server_ids else 'none'}; "
        f"missing_required_mcp={', '.join(missing_required_mcp_server_ids) if missing_required_mcp_server_ids else 'none'}; "
        f"mcp={', '.join(mcp_server_ids) if mcp_server_ids else 'none'}; "
        f"mcp_gaps={', '.join(missing_mcp_server_ids) if missing_mcp_server_ids else 'none'}; "
        f"mcp_policy={' / '.join(mcp_policy_parts) if mcp_policy_parts else 'inherit'}; "
        f"lanes={', '.join(delegation_lane_ids) if delegation_lane_ids else 'generalist'}; "
        f"focus={delegation_focus or 'general synthesis'}; "
        f"availability={availability_status}; "
        f"availability_blockers={', '.join(availability_blockers) if availability_blockers else 'none'}; "
        f"availability_warnings={', '.join(availability_warnings) if availability_warnings else 'none'}; "
        f"readiness={readiness_status}; "
        f"blockers={', '.join(readiness_blockers) if readiness_blockers else 'none'}; "
        f"warnings={', '.join(readiness_warnings) if readiness_warnings else 'none'}; "
        f"tool_support={tool_support}; "
        f"provider={provider_route}; review={review_mode}"
    )


def _format_delegation_partner(recommendation: dict[str, Any]) -> str:
    agent_name = str(recommendation.get("agent_name") or recommendation.get("agent_id") or "agent").strip()
    fit = str(recommendation.get("fit") or "weak").strip() or "weak"
    rationale = str(recommendation.get("rationale") or "").strip()
    interaction = str(recommendation.get("interaction") or "").strip()
    parts = [f"{agent_name} ({fit})"]
    if interaction:
        parts.append(f"via {interaction}")
    if rationale:
        parts.append(f"- {rationale}")
    return " ".join(parts)


def _format_agent_preview_list(entries: list[dict[str, Any]]) -> str:
    names = [
        str(item.get("agent_name") or item.get("agent_id") or "").strip()
        for item in entries
        if isinstance(item, dict) and str(item.get("agent_name") or item.get("agent_id") or "").strip()
    ]
    if not names:
        return "none"
    return ", ".join(names)


def _build_structured_delegation_contract(contract: dict[str, Any] | None) -> str:
    if not isinstance(contract, dict):
        return ""

    primary_role_mode = str(contract.get("primary_role_mode") or "generalist").strip() or "generalist"
    supporting_role_modes = [
        str(item).strip()
        for item in contract.get("supporting_role_modes") or []
        if str(item).strip()
    ]
    work_strategy = str(contract.get("work_strategy") or "flexible").strip() or "flexible"
    primary_focus = str(contract.get("primary_focus") or "").strip()
    upstream_agents = [
        dict(item)
        for item in contract.get("upstream_agents") or []
        if isinstance(item, dict)
    ]
    downstream_agents = [
        dict(item)
        for item in contract.get("downstream_agents") or []
        if isinstance(item, dict)
    ]
    preferred_collaborators = [
        dict(item)
        for item in contract.get("preferred_collaborators") or []
        if isinstance(item, dict)
    ]
    weak_handoff_targets = [
        dict(item)
        for item in contract.get("weak_handoff_targets") or []
        if isinstance(item, dict)
    ]
    watchouts = [
        " ".join(str(item).split())
        for item in contract.get("watchouts") or []
        if str(item).strip()
    ]

    lines = [
        f"Primary role mode: {primary_role_mode}.",
        "Supporting role modes: "
        + (", ".join(supporting_role_modes) if supporting_role_modes else "none")
        + ".",
        f"Work strategy: {work_strategy}.",
        "Coordinate parallel work: "
        + ("yes" if bool(contract.get("should_coordinate_parallel_work")) else "no")
        + ".",
        "Produce final output from this node: "
        + ("yes" if bool(contract.get("should_produce_final_output")) else "no")
        + ".",
        f"Upstream agents: {_format_agent_preview_list(upstream_agents)}.",
        f"Downstream agents: {_format_agent_preview_list(downstream_agents)}.",
        f"Preferred collaborators: {_format_agent_preview_list(preferred_collaborators)}.",
    ]
    if primary_focus:
        lines.append(f"Primary focus: {primary_focus}.")
    if weak_handoff_targets:
        lines.append(f"Weak handoff targets: {_format_agent_preview_list(weak_handoff_targets)}.")
    if watchouts:
        lines.append("Watchouts: " + " | ".join(watchouts[:4]))
    return "\n".join(lines)


def _build_orchestration_routing_summary(
    orchestration_summary: dict[str, Any] | None,
    *,
    expected_agent_count: int | None = None,
) -> str:
    if not isinstance(orchestration_summary, dict):
        return ""

    total_agent_count = int(orchestration_summary.get("total_agent_count") or 0)
    if expected_agent_count is not None and total_agent_count and total_agent_count != expected_agent_count:
        return ""

    readiness = str(orchestration_summary.get("readiness") or "ready").strip() or "ready"
    start_agents = [
        dict(item)
        for item in orchestration_summary.get("start_agents") or []
        if isinstance(item, dict)
    ]
    terminal_agents = [
        dict(item)
        for item in orchestration_summary.get("terminal_agents") or []
        if isinstance(item, dict)
    ]
    phases = [
        dict(item)
        for item in orchestration_summary.get("phases") or []
        if isinstance(item, dict)
    ]
    agent_routing = (
        dict(orchestration_summary.get("agent_routing") or {})
        if isinstance(orchestration_summary.get("agent_routing"), dict)
        else {}
    )
    repair_priorities = [
        dict(item)
        for item in orchestration_summary.get("repair_priorities") or []
        if isinstance(item, dict)
    ]
    single_owner_capability_risks = [
        dict(item)
        for item in orchestration_summary.get("single_owner_capability_risks") or []
        if isinstance(item, dict)
    ]

    lines = [
        f"Graph readiness: {readiness}.",
        f"Start nodes: {_format_agent_preview_list(start_agents)}.",
        f"Terminal nodes: {_format_agent_preview_list(terminal_agents)}.",
    ]

    phase_parts: list[str] = []
    for phase in phases:
        phase_id = str(phase.get("phase_id") or "").strip()
        agents = [dict(item) for item in phase.get("agents") or [] if isinstance(item, dict)]
        if not phase_id or not agents:
            continue
        phase_parts.append(f"{phase_id}={_format_agent_preview_list(agents)}")
    if phase_parts:
        lines.append("Phase anchors: " + "; ".join(phase_parts) + ".")

    routing_parts: list[str] = []
    routing_labels = {
        "coordinator_anchors": "coordinator",
        "research_anchors": "research",
        "implementation_anchors": "implementation",
        "verification_anchors": "verification",
        "skill_capable_anchors": "skills",
        "tool_capable_anchors": "tools",
        "mcp_capable_anchors": "mcp",
    }
    for key, label in routing_labels.items():
        entries = [dict(item) for item in agent_routing.get(key) or [] if isinstance(item, dict)]
        if not entries:
            continue
        routing_parts.append(f"{label}={_format_agent_preview_list(entries)}")
    if routing_parts:
        lines.append("Routing anchors: " + "; ".join(routing_parts) + ".")

    if repair_priorities:
        repair_parts = [
            f"{str(item.get('priority_id') or '').strip()} ({str(item.get('severity') or 'low').strip()} x{int(item.get('count') or 0)})"
            for item in repair_priorities
            if str(item.get("priority_id") or "").strip()
        ]
        if repair_parts:
            lines.append("Repair watchlist: " + "; ".join(repair_parts) + ".")

    if single_owner_capability_risks:
        risk_parts: list[str] = []
        for risk in single_owner_capability_risks[:4]:
            capability_kind = str(risk.get("kind") or "capability").strip()
            capability_id = str(risk.get("capability_id") or "").strip()
            owners = [dict(item) for item in risk.get("owner_agents") or [] if isinstance(item, dict)]
            if not capability_id:
                continue
            risk_parts.append(f"{capability_kind}:{capability_id} -> {_format_agent_preview_list(owners)}")
        if risk_parts:
            lines.append("Single-owner watchlist: " + "; ".join(risk_parts) + ".")

    return "\n".join(lines)


def _build_skill_guidance_block(
    *,
    loaded_skill_ids: list[str],
    skill_catalog_by_id: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    for skill_id in loaded_skill_ids:
        item = skill_catalog_by_id.get(_normalize_skill_key(skill_id), {})
        title = str(item.get("title") or _humanize_identifier(skill_id)).strip()
        prompt_hint = str(item.get("prompt_hint") or "").strip()
        suggested_tools = [
            _humanize_identifier(str(tool_id))
            for tool_id in item.get("suggested_tool_ids") or []
            if str(tool_id).strip()
        ]
        if not prompt_hint and not suggested_tools:
            continue
        line = f"- {title}: {prompt_hint or 'Use this approved skill pack when relevant.'}"
        if suggested_tools:
            line += f" Suggested tools when enabled: {', '.join(suggested_tools)}."
        lines.append(line)
    return "\n".join(lines)


def _build_agent_collaboration_contract(
    *,
    agent_id: str,
    agents_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    capability_summary_by_agent_id: dict[str, dict[str, Any]],
) -> str:
    incoming: list[str] = []
    outgoing: list[str] = []
    agent_summary = capability_summary_by_agent_id.get(agent_id, {})
    downstream_fit_by_agent_id = {
        str(item.get("agent_id") or "").strip(): dict(item)
        for item in agent_summary.get("downstream_handoff_scores") or []
        if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
    }
    recommended_collaborators = [
        dict(item)
        for item in agent_summary.get("recommended_collaborators") or []
        if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
    ]

    for edge in edges:
        source = str(edge.get("source_agent_id") or "").strip()
        target = str(edge.get("target_agent_id") or "").strip()
        interaction = str(edge.get("interaction") or "handoff").strip() or "handoff"
        if target == agent_id:
            source_agent = agents_by_id.get(source, {})
            source_summary = _resolve_agent_capability_summary(
                agent=source_agent,
                capability_summary_by_agent_id=capability_summary_by_agent_id,
            )
            source_name = str(source_agent.get("name") or source or "unknown agent").strip()
            incoming.append(
                f"- {source_name} -> you via {interaction}; upstream profile: {_format_capability_snapshot(source_summary)}."
            )
        if source == agent_id:
            target_agent = agents_by_id.get(target, {})
            target_summary = _resolve_agent_capability_summary(
                agent=target_agent,
                capability_summary_by_agent_id=capability_summary_by_agent_id,
            )
            target_name = str(target_agent.get("name") or target or "unknown agent").strip()
            fit_summary = downstream_fit_by_agent_id.get(target, {})
            fit = str(fit_summary.get("fit") or "").strip()
            rationale = str(fit_summary.get("rationale") or "").strip()
            fit_text = f"fit={fit}; " if fit else ""
            rationale_text = f"why={rationale}; " if rationale else ""
            outgoing.append(
                f"- you -> {target_name} via {interaction}; {fit_text}{rationale_text}"
                f"downstream profile: {_format_capability_snapshot(target_summary)}."
            )

    sections: list[str] = []
    if incoming:
        sections.append("Incoming collaboration edges:\n" + "\n".join(incoming))
    if outgoing:
        sections.append("Outgoing collaboration edges:\n" + "\n".join(outgoing))
    if not outgoing:
        sections.append("No explicit downstream edge is configured, so produce a self-contained final output.")
    if recommended_collaborators:
        sections.append(
            "Best-fit collaborators if you need an extra handoff:\n"
            + "\n".join(f"- {_format_delegation_partner(item)}" for item in recommended_collaborators[:3])
        )
    sections.append(
        "Shape your output for clean handoff: keep assumptions explicit, call out unresolved risks, and leave concrete next actions for the next edge. "
        "When useful, include compact sections such as Summary, Next actions, Open questions, and Risks so the next node can pick up your work cleanly."
    )
    return "\n\n".join(sections)


def _resolve_agent_output_key(agent: dict[str, Any]) -> str:
    if bool(agent.get("cluster_summary")):
        return str(agent.get("cluster_agent_id") or agent.get("agent_id") or "").strip()
    return str(agent.get("agent_id") or "").strip()


def _trim_handoff_payload(value: str, *, max_chars: int = 2200) -> str:
    content = str(value or "").strip()
    if len(content) <= max_chars:
        return content
    return content[: max(max_chars - 3, 1)].rstrip() + "..."


def _dedupe_preserve_order(values: list[str], *, limit: int = 3) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value or "").strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
        if len(output) >= limit:
            break
    return output


def _extract_handoff_summary(content: str, *, max_chars: int = 280) -> str:
    paragraphs = [
        " ".join(part.strip().split())
        for part in re.split(r"\n\s*\n", str(content or "").strip())
        if str(part).strip()
    ]
    if not paragraphs:
        return ""
    summary = paragraphs[0]
    if len(summary) <= max_chars:
        return summary
    return summary[: max(max_chars - 3, 1)].rstrip() + "..."


def _extract_action_items(content: str) -> list[str]:
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
    return _dedupe_preserve_order(candidates)


def _extract_open_questions(content: str) -> list[str]:
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
    return _dedupe_preserve_order(candidates)


def _extract_risk_flags(content: str) -> list[str]:
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
    return _dedupe_preserve_order(candidates)


def _build_agent_output_artifact(
    *,
    agent_config: dict[str, Any],
    content: str,
    outgoing_edges: list[dict[str, Any]],
    agent_directory: dict[str, dict[str, Any]],
    incoming_edges: list[dict[str, Any]] | None = None,
    state: OrchestrationState | None = None,
    tool_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    consumed_handoffs: list[dict[str, Any]] = []
    if state is None:
        outputs: dict[str, str] = {}
        artifacts: dict[str, dict[str, Any]] = {}
    else:
        outputs = cast(dict[str, str], state.get("agent_outputs") or {})
        artifacts = state.get("output_artifacts") or {}

    for edge in incoming_edges or []:
        source_agent_id = str(edge.get("source_agent_id") or "").strip()
        if not source_agent_id:
            continue
        source_agent = agent_directory.get(source_agent_id, {})
        source_name = str(source_agent.get("name") or source_agent_id or "unknown agent").strip()
        interaction = str(edge.get("interaction") or "handoff").strip() or "handoff"
        output_key = _resolve_agent_output_key(source_agent or {"agent_id": source_agent_id})
        source_output = str(outputs.get(output_key) or "").strip()
        source_artifact = dict(artifacts.get(output_key) or {})
        artifact_summary = _summarize_output_artifact(source_artifact)

        if not artifact_summary and not source_output:
            continue

        handoff: dict[str, Any] = {
            "source_agent_id": source_agent_id,
            "source_agent_name": source_name,
            "interaction": interaction,
        }
        if artifact_summary:
            handoff["artifact_summary"] = artifact_summary
        if source_output:
            handoff["output_preview"] = _trim_handoff_payload(source_output, max_chars=400)
            handoff["output_char_count"] = len(source_output)
        consumed_handoffs.append(handoff)

    downstream_handoffs: list[dict[str, str]] = []
    for edge in outgoing_edges:
        target_agent_id = str(edge.get("target_agent_id") or "").strip()
        if not target_agent_id:
            continue
        target_agent = agent_directory.get(target_agent_id, {})
        downstream_handoffs.append(
            {
                "target_agent_id": target_agent_id,
                "target_agent_name": str(target_agent.get("name") or target_agent_id).strip(),
                "interaction": str(edge.get("interaction") or "handoff").strip() or "handoff",
            }
        )

    artifact = {
        "node_kind": "agent",
        "agent_id": str(agent_config.get("agent_id") or "").strip(),
        "agent_name": str(agent_config.get("name") or agent_config.get("agent_id") or "agent").strip(),
        "role": str(agent_config.get("role") or "specialist").strip() or "specialist",
        "handoff_summary": _extract_handoff_summary(content),
        "action_items": _extract_action_items(content),
        "open_questions": _extract_open_questions(content),
        "risk_flags": _extract_risk_flags(content),
        "output_preview": _trim_handoff_payload(content, max_chars=600),
        "output_char_count": len(str(content or "")),
        "downstream_handoffs": downstream_handoffs,
        "final_output": len(downstream_handoffs) == 0,
    }
    mcp_server_ids = [str(item) for item in agent_config.get("mcp_server_ids") or [] if str(item).strip()]
    missing_mcp_server_ids = [
        str(item) for item in agent_config.get("missing_mcp_server_ids") or [] if str(item).strip()
    ]
    allowed_tool_ids = [str(item) for item in agent_config.get("allowed_tool_ids") or [] if str(item).strip()]
    denied_tool_ids = [str(item) for item in agent_config.get("denied_tool_ids") or [] if str(item).strip()]
    allowed_mcp_server_ids = [
        str(item) for item in agent_config.get("allowed_mcp_server_ids") or [] if str(item).strip()
    ]
    denied_mcp_server_ids = [
        str(item) for item in agent_config.get("denied_mcp_server_ids") or [] if str(item).strip()
    ]
    if mcp_server_ids:
        artifact["mcp_server_ids"] = mcp_server_ids
    if missing_mcp_server_ids:
        artifact["missing_mcp_server_ids"] = missing_mcp_server_ids
    if allowed_tool_ids:
        artifact["allowed_tool_ids"] = allowed_tool_ids
    if denied_tool_ids:
        artifact["denied_tool_ids"] = denied_tool_ids
    if allowed_mcp_server_ids:
        artifact["allowed_mcp_server_ids"] = allowed_mcp_server_ids
    if denied_mcp_server_ids:
        artifact["denied_mcp_server_ids"] = denied_mcp_server_ids
    if consumed_handoffs:
        artifact["consumed_handoffs"] = consumed_handoffs
    if tool_runs:
        artifact["tool_runs"] = list(tool_runs)
    return artifact


def _summarize_output_artifact(artifact: dict[str, Any]) -> str:
    if not isinstance(artifact, dict) or not artifact:
        return ""
    if str(artifact.get("node_kind") or "") == "cluster":
        summary_parts: list[str] = []
        winning_strategy = str(artifact.get("winning_strategy") or artifact.get("winning_vote") or "").strip()
        if winning_strategy:
            summary_parts.append(f"winning strategy={winning_strategy}")
        next_step = str(artifact.get("next_step") or "").strip()
        if next_step:
            summary_parts.append(f"next step={next_step}")
        dominant_risks = str(artifact.get("dominant_risks") or "").strip()
        if dominant_risks:
            summary_parts.append(f"dominant risks={dominant_risks}")
        return "; ".join(summary_parts)
    summary_parts = []
    handoff_summary = str(artifact.get("handoff_summary") or "").strip()
    if handoff_summary:
        summary_parts.append(f"summary={handoff_summary}")
    action_items = [str(item).strip() for item in artifact.get("action_items") or [] if str(item).strip()]
    if action_items:
        summary_parts.append(f"next actions={', '.join(action_items[:3])}")
    open_questions = [str(item).strip() for item in artifact.get("open_questions") or [] if str(item).strip()]
    if open_questions:
        summary_parts.append(f"open questions={', '.join(open_questions[:2])}")
    risk_flags = [str(item).strip() for item in artifact.get("risk_flags") or [] if str(item).strip()]
    if risk_flags:
        summary_parts.append(f"risks={', '.join(risk_flags[:2])}")
    tool_ids = [
        str(item.get("tool_id") or "").strip()
        for item in artifact.get("tool_runs") or []
        if isinstance(item, dict) and str(item.get("tool_id") or "").strip()
    ]
    if tool_ids:
        summary_parts.append(f"tools={', '.join(list(dict.fromkeys(tool_ids))[:3])}")
    return "; ".join(summary_parts)
    return ""


def _build_upstream_handoff_context(
    *,
    agent_id: str,
    agent_directory: dict[str, dict[str, Any]],
    incoming_edges: list[dict[str, Any]],
    state: OrchestrationState,
) -> str:
    if not incoming_edges:
        return ""

    outputs = state.get("agent_outputs", {}) or {}
    artifacts = state.get("output_artifacts", {}) or {}
    blocks: list[str] = []

    for edge in incoming_edges:
        source_agent_id = str(edge.get("source_agent_id") or "").strip()
        if not source_agent_id:
            continue
        source_agent = agent_directory.get(source_agent_id, {})
        source_name = str(source_agent.get("name") or source_agent_id or "unknown agent").strip()
        interaction = str(edge.get("interaction") or "handoff").strip() or "handoff"
        output_key = _resolve_agent_output_key(source_agent or {"agent_id": source_agent_id})
        source_output = str(outputs.get(output_key) or "").strip()
        artifact_summary = _summarize_output_artifact(dict(artifacts.get(output_key) or {}))

        if not source_output and not artifact_summary:
            continue

        lines = [f"- From {source_name} via {interaction}."]
        if artifact_summary:
            lines.append(f"  Structured artifact: {artifact_summary}")
        if source_output:
            trimmed_output = _trim_handoff_payload(source_output)
            indented_output = "\n".join(f"    {line}" for line in trimmed_output.splitlines())
            lines.append("  Completed upstream output:")
            lines.append(indented_output)
        blocks.append("\n".join(lines))

    if not blocks:
        return ""

    return "Direct upstream handoffs already completed:\n" + "\n\n".join(blocks)


def _topological_sort(agents: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    """
    Kahn's algorithm for topological sorting of agents based on edges.
    Returns agent IDs in execution order.
    """
    in_degree: dict[str, int] = {str(a["agent_id"]): 0 for a in agents}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        source = str(edge.get("source_agent_id", ""))
        target = str(edge.get("target_agent_id", ""))
        if source in in_degree and target in in_degree:
            adj[source].append(target)
            in_degree[target] += 1

    queue = deque([node for node, deg in in_degree.items() if deg == 0])
    ordered: list[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # In case of cycles, some nodes will be left with in_degree > 0
    # For now, just append the rest in their original order
    for node, deg in in_degree.items():
        if deg > 0 and node not in ordered:
            ordered.append(node)

    return ordered


def _build_execution_stages(agents: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[str]]:
    in_degree: dict[str, int] = {str(agent["agent_id"]): 0 for agent in agents}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        source = str(edge.get("source_agent_id") or "")
        target = str(edge.get("target_agent_id") or "")
        if source in in_degree and target in in_degree:
            adjacency[source].append(target)
            in_degree[target] += 1

    ready = [node for node, degree in in_degree.items() if degree == 0]
    stages: list[list[str]] = []
    seen: set[str] = set()

    while ready:
        current_stage = list(ready)
        ready = []
        stages.append(current_stage)
        for node in current_stage:
            seen.add(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready.append(neighbor)

    if len(seen) != len(in_degree):
        for agent in agents:
            agent_id = str(agent.get("agent_id") or "")
            if agent_id and agent_id not in seen:
                stages.append([agent_id])

    return stages


def _build_cluster_member_prompt(cluster_config: dict[str, Any], member_config: dict[str, Any]) -> str:
    cluster_name = str(cluster_config.get("name") or "cluster")
    cluster_strategy = str(cluster_config.get("cluster_strategy") or "custom")
    cluster_description = str(cluster_config.get("description") or "")
    cluster_prompt = str(cluster_config.get("system_prompt") or "")
    member_prompt = str(member_config.get("system_prompt") or "")
    base = [
        f"You are part of the cluster '{cluster_name}'.",
        f"Cluster strategy: {cluster_strategy}.",
    ]
    if cluster_description:
        base.append(f"Cluster brief: {cluster_description}")
    if cluster_prompt:
        base.append(f"Cluster instructions: {cluster_prompt}")
    if member_prompt:
        base.append(f"Your member-specific instructions: {member_prompt}")
    if cluster_strategy == "brainstorm":
        base.extend(
            [
                "For brainstorm clusters, push for energetic disagreement when useful and stress-test assumptions.",
                "Use game theory explicitly: identify players, incentives, payoff tradeoffs, strategic moves, and equilibrium risks before recommending a path.",
                "Before your vote line, include this compact analysis block exactly once when possible:",
                "PLAYERS:",
                "INCENTIVES:",
                "RISKS:",
                "EQUILIBRIUM:",
                "Each response must end with a single explicit vote line in the exact format: VOTE: CONSERVATIVE, VOTE: BALANCED, or VOTE: AGGRESSIVE.",
                "When you vote, choose the strategy you believe should advance after this round.",
            ]
        )
    return "\n".join(base)


def _extract_brainstorm_votes(prior_sections: list[str]) -> dict[str, int]:
    counts = {"conservative": 0, "balanced": 0, "aggressive": 0}
    pattern = re.compile(r"VOTE:\s*(CONSERVATIVE|BALANCED|AGGRESSIVE)", re.IGNORECASE)
    for section in prior_sections:
        match = pattern.search(section)
        if not match:
            continue
        counts[match.group(1).lower()] += 1
    return counts


def _extract_member_analysis_block(content: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"^(PLAYERS|INCENTIVES|RISKS|EQUILIBRIUM|VOTE):(?:[ \t]*.*)?$",
            content,
            re.MULTILINE,
        )
    )
    sections = {"PLAYERS": "", "INCENTIVES": "", "RISKS": "", "EQUILIBRIUM": ""}
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections[label] = content[start:end].strip()
    return sections


def _merge_round_analysis(entries: list[dict[str, str]]) -> dict[str, str]:
    buckets: dict[str, list[str]] = {
        "key_players": [],
        "incentive_map": [],
        "dominant_risks": [],
        "expected_equilibrium": [],
    }
    seen: dict[str, set[str]] = {key: set() for key in buckets}
    for entry in entries:
        parsed = _extract_member_analysis_block(entry.get("output", ""))
        mapping = {
            "key_players": parsed["PLAYERS"],
            "incentive_map": parsed["INCENTIVES"],
            "dominant_risks": parsed["RISKS"],
            "expected_equilibrium": parsed["EQUILIBRIUM"],
        }
        for key, value in mapping.items():
            normalized = value.strip()
            if normalized and normalized not in seen[key]:
                seen[key].add(normalized)
                buckets[key].append(normalized)
    return {key: " | ".join(values) for key, values in buckets.items()}


def _build_equilibrium_shift_details(
    previous_round: dict[str, Any] | None,
    current_round: dict[str, Any],
) -> dict[str, Any]:
    if previous_round is None:
        return {
            "shift_type": "initial",
            "changed_fields": [],
            "vote_shift": None,
            "summary": "Initial round baseline.",
        }

    changes: list[str] = []
    changed_fields: list[str] = []
    vote_shift: dict[str, str] | None = None
    if str(previous_round.get("winning_vote") or "") != str(current_round.get("winning_vote") or ""):
        changed_fields.append("winning_vote")
        vote_shift = {
            "from": str(previous_round.get("winning_vote") or "unknown"),
            "to": str(current_round.get("winning_vote") or "unknown"),
        }
        changes.append(
            f"Winning vote shifted from {previous_round.get('winning_vote') or 'unknown'} to {current_round.get('winning_vote') or 'unknown'}."
        )

    field_labels = {
        "key_players": "Player set",
        "incentive_map": "Incentive map",
        "dominant_risks": "Dominant risks",
        "expected_equilibrium": "Expected equilibrium",
    }
    for field, label in field_labels.items():
        previous_value = str(previous_round.get(field) or "").strip()
        current_value = str(current_round.get(field) or "").strip()
        if previous_value != current_value and current_value:
            changed_fields.append(field)
            if previous_value:
                changes.append(f"{label} changed from '{previous_value}' to '{current_value}'.")
            else:
                changes.append(f"{label} emerged as '{current_value}'.")

    shift_type = "none"
    if vote_shift is not None:
        shift_type = "vote_change"
    elif changed_fields:
        shift_type = "state_change"

    return {
        "shift_type": shift_type,
        "changed_fields": changed_fields,
        "vote_shift": vote_shift,
        "summary": " ".join(changes) if changes else "No major equilibrium shift from the previous round.",
    }


def _build_brainstorm_round_history(
    *,
    member_node_ids: list[str],
    outputs: dict[str, str],
) -> list[dict[str, Any]]:
    rounds: dict[int, list[dict[str, str]]] = defaultdict(list)
    for node_id in member_node_ids:
        content = str(outputs.get(node_id) or "").strip()
        if not content:
            continue
        match = re.search(r"__round_(\d+)__", node_id)
        round_index = int(match.group(1)) if match else 1
        rounds[round_index].append(
            {
                "member_node_id": node_id,
                "output": content,
            }
        )

    history: list[dict[str, Any]] = []
    previous_round: dict[str, Any] | None = None
    for round_index in sorted(rounds):
        entries = rounds[round_index]
        prior_sections = [entry["output"] for entry in entries]
        vote_tally = _extract_brainstorm_votes(prior_sections)
        round_analysis = _merge_round_analysis(entries)
        current_round = {
            "round_index": round_index,
            "member_outputs": entries,
            "vote_tally": vote_tally,
            "winning_vote": _winning_brainstorm_vote(vote_tally),
            **round_analysis,
        }
        shift_details = _build_equilibrium_shift_details(previous_round, current_round)
        current_round["equilibrium_shift"] = str(shift_details.get("summary") or "")
        current_round["equilibrium_shift_details"] = shift_details
        history.append(current_round)
        previous_round = current_round
    return history


def _winning_brainstorm_vote(votes: dict[str, int]) -> str:
    order = ["balanced", "conservative", "aggressive"]
    return max(order, key=lambda key: (votes.get(key, 0), -order.index(key))).upper()


def _compose_system_instruction(*parts: str) -> str:
    normalized = [FIRST_PRINCIPLES_DIRECTIVE]
    normalized.extend(part.strip() for part in parts if part and part.strip())
    return "\n\n".join(normalized)


def _compose_brainstorm_instruction(*parts: str) -> str:
    normalized = [FIRST_PRINCIPLES_DIRECTIVE, GAME_THEORY_DIRECTIVE]
    normalized.extend(part.strip() for part in parts if part and part.strip())
    return "\n\n".join(normalized)


def _build_brainstorm_summary_prompt(
    *,
    cluster_name: str,
    task: str,
    prior_sections: list[str],
    rounds: int,
) -> list[BaseMessage]:
    context = "\n\n".join(prior_sections) if prior_sections else "No member output was captured."
    votes = _extract_brainstorm_votes(prior_sections)
    winning_vote = _winning_brainstorm_vote(votes)
    return [
        SystemMessage(
            content=_compose_brainstorm_instruction(
                f"You are the synthesis chair for the brainstorm cluster '{cluster_name}'.",
                "Summarize the roundtable debate into three options: conservative, balanced, and aggressive.",
                "Frame each option using strategic interaction, incentive alignment, and likely equilibrium behavior.",
                "Then declare a winning strategy with a short rationale and a clear next-step recommendation.",
                "You must preserve the three-option structure even if one option is weak.",
            )
        ),
        HumanMessage(
            content=(
                f"Overall task: {task}\n"
                f"Debate rounds completed: {rounds}\n\n"
                "Observed vote tally:\n"
                f"- Conservative: {votes['conservative']}\n"
                f"- Balanced: {votes['balanced']}\n"
                f"- Aggressive: {votes['aggressive']}\n"
                f"Current winning vote: {winning_vote}\n\n"
                "Respond using this exact section structure:\n"
                "Conservative Strategy:\n"
                "Balanced Strategy:\n"
                "Aggressive Strategy:\n"
                "Winning Strategy:\n"
                "Game Theory Rationale:\n"
                "Key Players:\n"
                "Incentive Map:\n"
                "Dominant Risks:\n"
                "Expected Equilibrium:\n"
                "Next Step:\n\n"
                "Cluster debate transcript:\n"
                f"{context}"
            )
        ),
    ]


def _build_brainstorm_round_context(*, agent_config: dict[str, Any], state: OrchestrationState) -> str:
    if str(agent_config.get("cluster_strategy") or "") != "brainstorm":
        return ""
    current_round = int(agent_config.get("cluster_round_index") or 1)
    if current_round <= 1:
        return ""

    cluster_agent_id = str(agent_config.get("cluster_agent_id") or "")
    prior_member_ids = [
        str(node_id)
        for node_id in agent_config.get("cluster_member_node_ids") or []
        if str(node_id) and f"__round_{current_round - 1}__" in str(node_id)
    ]
    outputs = state.get("agent_outputs", {})
    prior_sections = [f"{node_id}:\n{outputs.get(node_id, '')}" for node_id in prior_member_ids if outputs.get(node_id)]
    if not prior_sections:
        return f"This is round {current_round}. No prior round transcript was available, so rebuild from first principles."

    votes = _extract_brainstorm_votes(prior_sections)
    winner = _winning_brainstorm_vote(votes)
    return (
        f"This is brainstorm round {current_round} for cluster '{cluster_agent_id or agent_config.get('name') or 'cluster'}'.\n"
        f"Previous round vote tally: Conservative={votes['conservative']}, Balanced={votes['balanced']}, Aggressive={votes['aggressive']}.\n"
        f"Current leading direction from the previous round: {winner}.\n"
        "React to the leading direction explicitly: either strengthen it with better first-principles support, or overturn it with a stronger alternative.\n"
        "Use game theory in this round: identify the key players, incentives, strategic responses, and equilibrium shifts that support or weaken the current leading direction."
    )


def _build_prior_research_context(*, agent_id: str, state: OrchestrationState) -> str:
    output_artifacts = state.get("output_artifacts", {}) or {}
    sections: list[str] = []
    for artifact_agent_id, artifact in output_artifacts.items():
        if str(artifact_agent_id) == str(agent_id):
            continue
        if not isinstance(artifact, dict):
            continue
        research = artifact.get("research")
        if not isinstance(research, dict):
            continue
        cluster_name = str(artifact.get("cluster_name") or artifact_agent_id or "cluster")
        winning_strategy = str(artifact.get("winning_strategy") or "").strip()
        next_step = str(artifact.get("next_step") or "").strip()
        latest_progress = [
            str(item).strip()
            for item in (research.get("latest_progress") or [])
            if str(item).strip()
        ][:3]
        paper_titles = [
            str(item.get("title") or "").strip()
            for item in (research.get("papers") or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ][:3]
        block_lines = [f"Evidence from prior cluster '{cluster_name}':"]
        if winning_strategy:
            block_lines.append(f"- Winning strategy: {winning_strategy}")
        game_theory_rationale = str(artifact.get("game_theory_rationale") or "").strip()
        if game_theory_rationale:
            block_lines.append(f"- Game-theoretic rationale: {game_theory_rationale}")
        key_players = str(artifact.get("key_players") or "").strip()
        if key_players:
            block_lines.append(f"- Key players: {key_players}")
        incentive_map = str(artifact.get("incentive_map") or "").strip()
        if incentive_map:
            block_lines.append(f"- Incentive map: {incentive_map}")
        dominant_risks = str(artifact.get("dominant_risks") or "").strip()
        if dominant_risks:
            block_lines.append(f"- Dominant risks: {dominant_risks}")
        expected_equilibrium = str(artifact.get("expected_equilibrium") or "").strip()
        if expected_equilibrium:
            block_lines.append(f"- Expected equilibrium: {expected_equilibrium}")
        if next_step:
            block_lines.append(f"- Recommended next step: {next_step}")
        if latest_progress:
            block_lines.append("- Latest progress signals:")
            block_lines.extend(f"  - {item}" for item in latest_progress)
        if paper_titles:
            block_lines.append("- Relevant papers:")
            block_lines.extend(f"  - {item}" for item in paper_titles)
        sections.append("\n".join(block_lines))
    return "\n\n".join(sections)


def _parse_labeled_sections(content: str, labels: list[str]) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"^(Conservative Strategy|Balanced Strategy|Aggressive Strategy|Winning Strategy|Game Theory Rationale|Key Players|Incentive Map|Dominant Risks|Expected Equilibrium|Next Step):\s*$",
            content,
            re.MULTILINE,
        )
    )
    sections = dict.fromkeys(labels, "")
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        if label in sections:
            sections[label] = content[start:end].strip()
    return sections


def _build_cluster_output_artifact(
    *,
    cluster_agent_id: str,
    cluster_name: str,
    cluster_strategy: str,
    member_node_ids: list[str],
    outputs: dict[str, str],
    rounds: int,
    summary: str,
    prior_sections: list[str],
    mcp_server_ids: list[str] | None = None,
    missing_mcp_server_ids: list[str] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "node_kind": "cluster",
        "cluster_agent_id": cluster_agent_id,
        "cluster_name": cluster_name,
        "cluster_strategy": cluster_strategy,
        "member_node_ids": member_node_ids,
        "member_count": len(member_node_ids),
        "rounds_completed": rounds,
        "raw_summary": summary,
    }
    if mcp_server_ids:
        artifact["mcp_server_ids"] = [str(item) for item in mcp_server_ids if str(item).strip()]
    if missing_mcp_server_ids:
        artifact["missing_mcp_server_ids"] = [
            str(item) for item in missing_mcp_server_ids if str(item).strip()
        ]
    if cluster_strategy != "brainstorm":
        return artifact

    votes = _extract_brainstorm_votes(prior_sections)
    winning_vote = _winning_brainstorm_vote(votes)
    sections = _parse_labeled_sections(
        summary,
        [
            "Conservative Strategy",
            "Balanced Strategy",
            "Aggressive Strategy",
            "Winning Strategy",
            "Game Theory Rationale",
            "Key Players",
            "Incentive Map",
            "Dominant Risks",
            "Expected Equilibrium",
            "Next Step",
        ],
    )
    artifact.update(
        {
            "vote_tally": votes,
            "winning_vote": winning_vote,
            "round_history": _build_brainstorm_round_history(member_node_ids=member_node_ids, outputs=outputs),
            "strategies": {
                "conservative": sections["Conservative Strategy"],
                "balanced": sections["Balanced Strategy"],
                "aggressive": sections["Aggressive Strategy"],
            },
            "winning_strategy": sections["Winning Strategy"],
            "game_theory_rationale": sections["Game Theory Rationale"],
            "key_players": sections["Key Players"],
            "incentive_map": sections["Incentive Map"],
            "dominant_risks": sections["Dominant Risks"],
            "expected_equilibrium": sections["Expected Equilibrium"],
            "next_step": sections["Next Step"],
        }
    )
    return artifact


def _expand_cluster_agents(
    agents: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_agents: list[dict[str, Any]] = []
    expanded_edges: list[dict[str, Any]] = []
    entry_map: dict[str, str] = {}
    exit_map: dict[str, str] = {}

    for agent in agents:
        agent_id = str(agent.get("agent_id") or "")
        if not agent_id:
            continue
        if str(agent.get("node_kind") or "agent") != "cluster":
            expanded_agents.append(agent)
            entry_map[agent_id] = agent_id
            exit_map[agent_id] = agent_id
            continue

        members = [member for member in agent.get("cluster_members") or [] if isinstance(member, dict)]
        cluster_name = str(agent.get("name") or agent_id)
        cluster_strategy = str(agent.get("cluster_strategy") or "custom")
        cluster_rounds = max(1, min(int(agent.get("brainstorm_rounds") or 1), 5))
        effective_rounds = cluster_rounds if cluster_strategy == "brainstorm" else 1

        if not members:
            members = [
                {
                    "member_id": f"{agent_id}_lead",
                    "name": f"{cluster_name} lead",
                    "role": "lead",
                    "system_prompt": "Propose the first workable plan for the cluster.",
                    "model": str(agent.get("model") or "gpt-5.2"),
                }
            ]

        member_node_ids: list[str] = []
        previous_member_node_id: str | None = None
        for round_index in range(1, effective_rounds + 1):
            for member_index, member in enumerate(members, start=1):
                member_node_id = (
                    f"{agent_id}__round_{round_index}__member_{member_index}"
                    if effective_rounds > 1
                    else f"{agent_id}__member_{member_index}"
                )
                member_node_ids.append(member_node_id)
                expanded_agents.append(
                    {
                        "agent_id": member_node_id,
                        "name": (
                            f"{cluster_name} / {member.get('name') or f'member {member_index}'}"
                            f"{f' / round {round_index}' if effective_rounds > 1 else ''}"
                        ),
                        "role": str(member.get("role") or "specialist"),
                        "description": str(agent.get("description") or ""),
                        "system_prompt": _build_cluster_member_prompt(agent, member),
                        "model": str(member.get("model") or agent.get("model") or "gpt-5.2"),
                        "preferred_provider_id": member.get("preferred_provider_id") or agent.get("preferred_provider_id"),
                        "fallback_provider_id": member.get("fallback_provider_id") or agent.get("fallback_provider_id"),
                        "temperature": float(member.get("temperature", agent.get("temperature", 0.2)) or 0.2),
                        "timeout_seconds": _coerce_timeout(
                            member.get("timeout_seconds", agent.get("timeout_seconds")),
                            _coerce_timeout(agent.get("timeout_seconds"), 60),
                        ),
                        "cluster_agent_id": agent_id,
                        "cluster_strategy": cluster_strategy,
                        "cluster_round_index": round_index,
                        "cluster_member_node_ids": member_node_ids,
                    }
                )
                if previous_member_node_id is not None:
                    expanded_edges.append(
                        {
                            "edge_id": f"{agent_id}__internal_{previous_member_node_id}__{member_node_id}",
                            "source_agent_id": previous_member_node_id,
                            "target_agent_id": member_node_id,
                            "interaction": "cluster_internal",
                        }
                    )
                previous_member_node_id = member_node_id

        summary_node_id = f"{agent_id}__summary"
        expanded_agents.append(
            {
                "agent_id": summary_node_id,
                "name": f"{cluster_name} / summary",
                "role": "cluster_summary",
                "cluster_summary": True,
                "cluster_agent_id": agent_id,
                "cluster_name": cluster_name,
                "cluster_strategy": cluster_strategy,
                "brainstorm_rounds": effective_rounds,
                "cluster_member_node_ids": member_node_ids,
                "cluster_member_count": len(members),
                "cluster_auto_research": bool(agent.get("cluster_auto_research", False)),
                "cluster_auto_review": bool(agent.get("cluster_auto_review", True)),
                "model": str(agent.get("model") or "gpt-5.2"),
                "preferred_provider_id": agent.get("preferred_provider_id"),
                "fallback_provider_id": agent.get("fallback_provider_id"),
                "temperature": float(agent.get("temperature", 0.2) or 0.2),
                "timeout_seconds": _coerce_timeout(agent.get("timeout_seconds"), 60),
            }
        )
        expanded_edges.append(
            {
                "edge_id": f"{agent_id}__summary_edge",
                "source_agent_id": member_node_ids[-1],
                "target_agent_id": summary_node_id,
                "interaction": "cluster_summary",
            }
        )
        entry_map[agent_id] = member_node_ids[0]
        exit_map[agent_id] = summary_node_id

    for edge in edges:
        source = str(edge.get("source_agent_id") or "")
        target = str(edge.get("target_agent_id") or "")
        mapped_source = exit_map.get(source)
        mapped_target = entry_map.get(target)
        if not mapped_source or not mapped_target:
            continue
        expanded_edges.append(
            {
                **edge,
                "source_agent_id": mapped_source,
                "target_agent_id": mapped_target,
            }
        )

    return expanded_agents, expanded_edges


def build_orchestration_execution_plan(graph_json: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    agents = graph_json.get("agents") or []
    edges = graph_json.get("edges") or []
    provider_config = graph_json.get("provider_config") or {}
    review_config = graph_json.get("review_agent") or {}
    expanded_agents, expanded_edges = _expand_cluster_agents(agents, edges)
    capability_summary_by_agent_id = {
        str(item.get("agent_id") or "").strip(): dict(item)
        for item in graph_json.get("agent_capability_summaries") or []
        if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
    }
    skill_catalog_by_id = {
        _normalize_skill_key(str(item.get("skill_id") or "")): dict(item)
        for item in graph_json.get("skill_catalog") or []
        if isinstance(item, dict) and _normalize_skill_key(str(item.get("skill_id") or ""))
    }
    tool_catalog_by_id = _build_tool_catalog_lookup(graph_json)
    mcp_catalog_by_id = _build_mcp_catalog_lookup(graph_json)
    execution_order = _topological_sort(expanded_agents, expanded_edges)
    execution_stages = _build_execution_stages(expanded_agents, expanded_edges)
    stage_index_by_agent_id = {
        agent_id: index
        for index, stage in enumerate(execution_stages)
        for agent_id in stage
    }
    agent_by_id = {str(agent.get("agent_id") or ""): agent for agent in expanded_agents}
    team_capability_roster = _build_team_capability_roster(
        agents=expanded_agents,
        capability_summary_by_agent_id=capability_summary_by_agent_id,
    )
    orchestration_summary_brief = _build_orchestration_routing_summary(
        graph_json.get("orchestration_summary") if isinstance(graph_json.get("orchestration_summary"), dict) else None,
        expected_agent_count=len(agents),
    )
    ordered_agents: list[dict[str, Any]] = []
    for agent_id in execution_order:
        agent = agent_by_id.get(agent_id)
        if not agent:
            continue
        enriched_agent = dict(agent)
        summary = _resolve_agent_capability_summary(
            agent=enriched_agent,
            capability_summary_by_agent_id=capability_summary_by_agent_id,
        )
        if summary:
            enabled_tool_ids = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
            mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
            missing_mcp_server_ids = [
                str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()
            ]
            delegation_lane_ids = [str(item) for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
            delegation_focus = str(summary.get("delegation_focus") or "").strip()
            delegation_contract = (
                dict(summary.get("delegation_contract") or {})
                if isinstance(summary.get("delegation_contract"), dict)
                else {}
            )
            readiness_status = str(summary.get("readiness_status") or "").strip() or "ready"
            readiness_blockers = [
                str(item) for item in summary.get("readiness_blockers") or [] if str(item).strip()
            ]
            readiness_warnings = [
                str(item) for item in summary.get("readiness_warnings") or [] if str(item).strip()
            ]
            enriched_agent["capability_brief"] = str(summary.get("capability_brief") or "")
            enriched_agent["capability_execution_contract"] = _build_capability_execution_contract(summary)
            enriched_agent["delegation_lane_ids"] = delegation_lane_ids
            enriched_agent["delegation_focus"] = delegation_focus
            enriched_agent["delegation_contract"] = delegation_contract
            enriched_agent["structured_delegation_contract"] = _build_structured_delegation_contract(
                delegation_contract
            )
            enriched_agent["readiness_status"] = readiness_status
            enriched_agent["readiness_blockers"] = readiness_blockers
            enriched_agent["readiness_warnings"] = readiness_warnings
            enriched_agent["recommended_collaborators"] = [
                dict(item) for item in summary.get("recommended_collaborators") or [] if isinstance(item, dict)
            ]
            enriched_agent["downstream_handoff_scores"] = [
                dict(item) for item in summary.get("downstream_handoff_scores") or [] if isinstance(item, dict)
            ]
            enriched_agent["skill_guidance"] = _build_skill_guidance_block(
                loaded_skill_ids=[str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()],
                skill_catalog_by_id=skill_catalog_by_id,
            )
            enriched_agent["enabled_tool_ids"] = enabled_tool_ids
            enriched_agent["mcp_server_ids"] = mcp_server_ids
            enriched_agent["missing_mcp_server_ids"] = missing_mcp_server_ids
            enriched_agent["tool_guidance"] = _build_tool_guidance_block(
                enabled_tool_ids=enabled_tool_ids,
                tool_catalog_by_id=tool_catalog_by_id,
            )
            enriched_agent["mcp_guidance"] = _build_mcp_guidance_block(
                mcp_server_ids=mcp_server_ids,
                missing_mcp_server_ids=missing_mcp_server_ids,
                mcp_catalog_by_id=mcp_catalog_by_id,
            )
        enriched_agent["team_capability_roster"] = team_capability_roster
        enriched_agent["orchestration_summary_brief"] = orchestration_summary_brief
        enriched_agent["execution_stage_index"] = stage_index_by_agent_id.get(str(agent_id), 0)
        enriched_agent["agent_directory"] = agent_by_id
        enriched_agent["incoming_edges"] = [
            dict(edge)
            for edge in expanded_edges
            if str(edge.get("target_agent_id") or "").strip() == str(agent_id)
        ]
        enriched_agent["outgoing_edges"] = [
            dict(edge)
            for edge in expanded_edges
            if str(edge.get("source_agent_id") or "").strip() == str(agent_id)
        ]
        enriched_agent["collaboration_contract"] = _build_agent_collaboration_contract(
            agent_id=str(agent_id),
            agents_by_id=agent_by_id,
            edges=expanded_edges,
            capability_summary_by_agent_id=capability_summary_by_agent_id,
        )
        ordered_agents.append(enriched_agent)
    return ordered_agents, provider_config, review_config


async def invoke_orchestration_step(
    *,
    agent_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    state: OrchestrationState,
    is_review: bool = False,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    node = _build_agent_node(
        agent_config,
        provider_config,
        default_timeout,
        provider_registry,
        is_review=is_review,
        on_stream_chunk=on_stream_chunk,
        on_stream_event=on_stream_event,
    )
    return await node(state)


async def review_orchestration_output(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    output: str,
) -> dict[str, object]:
    if not bool(review_config.get("enabled", True)):
        return {"approved": True, "review_output": "PASS: review disabled"}

    model = str(review_config.get("model") or "gpt-5.1-codex-mini")
    preferred = review_config.get("preferred_provider_id") or provider_config.get("preferred_provider_id")
    fallback = review_config.get("fallback_provider_id") or provider_config.get("fallback_provider_id")
    timeout = _coerce_timeout(review_config.get("timeout_seconds"), default_timeout)
    resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
    llm = get_llm_for_provider(
        model=resolved.model,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        temperature=float(review_config.get("temperature", 0) or 0),
        timeout_seconds=timeout,
    )
    messages = [
        SystemMessage(
            content=_compose_system_instruction(
                f"You are {str(review_config.get('name') or 'Reviewer')}, the runtime safety reviewer for a collaborative workflow.",
                str(review_config.get("system_prompt") or ""),
                "Review only the provided latest node output.",
                "Reply with exactly one leading decision token: PASS or BLOCK.",
                "Use BLOCK only when the output is risky, unsafe, policy-violating, or requires human review before downstream use.",
            )
        ),
        HumanMessage(
            content=(
                f"Task: {task}\n"
                f"Node id: {agent_id}\n"
                f"Node name: {agent_name}\n\n"
                "Latest node output to review:\n"
                f"{output}"
            )
        ),
    ]
    try:
        with anyio.fail_after(timeout):
            review_output, _ = await _invoke_llm_with_streaming_fallback(llm, messages)
    except Exception as exc:
        review_output = f"BLOCK: review agent failed and requires human confirmation. Error: {exc!s}"
    normalized = review_output.strip().upper()
    approved = normalized.startswith("PASS")
    if not approved and not normalized.startswith("BLOCK"):
        review_output = (
            "BLOCK: review agent returned an invalid decision token and requires human confirmation. "
            f"Raw output: {review_output or '<empty>'}"
        )
    return {
        "approved": approved,
        "review_output": review_output or ("PASS: no issues detected" if approved else "BLOCK: human review required"),
    }


def _build_agent_node(
    agent_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    *,
    is_review: bool = False,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[OrchestrationState], Awaitable[dict[str, Any]]]:
    agent_id = str(agent_config["agent_id"])
    name = str(agent_config.get("name", agent_id))
    is_cluster_summary = bool(agent_config.get("cluster_summary"))
    system_prompt = str(agent_config.get("system_prompt", ""))
    model = str(agent_config.get("model", "gpt-5.2"))
    temperature = float(agent_config.get("temperature", 0.2))
    agent_timeout = agent_config.get("timeout_seconds")
    timeout = int(agent_timeout) if agent_timeout else default_timeout

    preferred = agent_config.get("preferred_provider_id") or provider_config.get("preferred_provider_id")
    fallback = agent_config.get("fallback_provider_id") or provider_config.get("fallback_provider_id")

    async def _invoke_agent(state: OrchestrationState) -> dict[str, Any]:
        """LangGraph node representing a single canvas agent."""
        _log.info("Executing canvas agent node: %s (%s)", name, agent_id)
        continuation_context = _build_continuation_context(agent_id, state)
        configured_enabled_tool_ids = [
            str(item) for item in agent_config.get("enabled_tool_ids") or [] if str(item).strip()
        ]

        if is_cluster_summary:
            cluster_agent_id = str(agent_config.get("cluster_agent_id") or agent_id)
            cluster_name = str(agent_config.get("cluster_name") or cluster_agent_id)
            cluster_strategy = str(agent_config.get("cluster_strategy") or "custom")
            member_node_ids = [str(node_id) for node_id in agent_config.get("cluster_member_node_ids") or [] if str(node_id)]
            outputs = state.get("agent_outputs", {})
            artifacts = state.get("output_artifacts", {}).copy()
            sections = [
                f"{member_id}:\n{outputs.get(member_id, '')}".strip()
                for member_id in member_node_ids
                if outputs.get(member_id)
            ]
            summary = "\n\n".join(section for section in sections if section) or f"{cluster_name} completed with no member output."
            if cluster_strategy == "brainstorm":
                resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
                llm = get_llm_for_provider(
                    model=resolved.model,
                    base_url=resolved.base_url,
                    api_key=resolved.api_key,
                    temperature=temperature,
                    timeout_seconds=timeout,
                )
                try:
                    prompt_messages = _build_brainstorm_summary_prompt(
                        cluster_name=cluster_name,
                        task=str(state.get("task") or ""),
                        prior_sections=sections,
                        rounds=int(agent_config.get("brainstorm_rounds") or 1),
                    )
                    if continuation_context:
                        prompt_messages = list(prompt_messages) + [
                            HumanMessage(
                                content=(
                                    "A previous approved partial summary for this same node already exists. "
                                    "Continue from that exact prefix without restarting or repeating it.\n\n"
                                    f"Approved partial summary:\n{continuation_context['partial_output']}\n\n"
                                    f"Review note:\n{continuation_context['review_output'] or 'none'}"
                                )
                            )
                        ]
                    with anyio.fail_after(timeout):
                        summary, _ = await _invoke_llm_with_streaming_fallback(
                            llm,
                            prompt_messages,
                            on_stream_chunk=on_stream_chunk,
                            on_stream_event=on_stream_event,
                        )
                        if continuation_context:
                            summary = _merge_continuation_output(
                                str(continuation_context.get("partial_output") or ""),
                                summary,
                            )
                except TimeoutError:
                    summary = f"{cluster_name} summary timed out.\n\n{summary}"
                except OutputGuardrailTrip:
                    raise
                except Exception as exc:
                    summary = f"{cluster_name} summary failed: {exc!s}\n\n{summary}"
            new_outputs = outputs.copy()
            new_outputs[cluster_agent_id] = summary
            artifacts[cluster_agent_id] = _build_cluster_output_artifact(
                cluster_agent_id=cluster_agent_id,
                cluster_name=cluster_name,
                cluster_strategy=cluster_strategy,
                member_node_ids=member_node_ids,
                outputs=outputs,
                rounds=int(agent_config.get("brainstorm_rounds") or 1),
                summary=summary,
                prior_sections=sections,
                mcp_server_ids=[
                    str(item) for item in agent_config.get("mcp_server_ids") or [] if str(item).strip()
                ],
                missing_mcp_server_ids=[
                    str(item)
                    for item in agent_config.get("missing_mcp_server_ids") or []
                    if str(item).strip()
                ],
            )
            if "cluster_member_count" in agent_config:
                artifacts[cluster_agent_id]["member_count"] = int(agent_config.get("cluster_member_count") or 0)
            return {
                "current_agent": cluster_agent_id,
                "agent_outputs": new_outputs,
                "output_artifacts": artifacts,
                "errors": list(state.get("errors", [])),
                "messages": [AIMessage(content=f"[{cluster_name}] cluster summary:\n{summary}")],
            }

        resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
        llm = get_llm_for_provider(
            model=resolved.model,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            temperature=temperature,
            timeout_seconds=timeout,
        )
        runtime_tool_support, runtime_tool_support_reason = infer_tool_calling_support(
            model=resolved.model,
            base_url=resolved.base_url,
            provider_id=resolved.provider_id,
        )
        llm_supports_tool_binding = callable(getattr(llm, "bind_tools", None))
        runtime_enabled_tool_ids = list(configured_enabled_tool_ids)
        if runtime_enabled_tool_ids and not is_review and not is_cluster_summary:
            if runtime_tool_support == "unsupported":
                runtime_enabled_tool_ids = []
            elif not llm_supports_tool_binding:
                runtime_tool_support = "unsupported"
                runtime_tool_support_reason = "This runtime adapter does not expose native tool binding."
                runtime_enabled_tool_ids = []
        enabled_tools = _resolve_enabled_tools(runtime_enabled_tool_ids) if not is_review and not is_cluster_summary else []

        if is_review:
            prior_outputs = [
                f"Agent {aid} output:\n{output}"
                for aid, output in state.get("agent_outputs", {}).items()
                if str(aid).strip()
            ]
            review_context = (
                f"The overall user task is: {state.get('task')}\n\n"
                "Review the completed agent collaboration for correctness, safety, and handoff quality.\n"
                "Return a concise review summary."
            )
            capability_roster = str(agent_config.get("team_capability_roster") or "").strip()
            if capability_roster:
                review_context += (
                    "\n\nCapability map for the collaborating team:\n"
                    f"{capability_roster}\n"
                )
            orchestration_summary_brief = str(agent_config.get("orchestration_summary_brief") or "").strip()
            if orchestration_summary_brief:
                review_context += (
                    "\n\nGraph orchestration brief:\n"
                    f"{orchestration_summary_brief}\n"
                )
            if prior_outputs:
                review_context += "\n\nCollected agent outputs:\n" + "\n\n".join(prior_outputs)
            messages = [
                SystemMessage(
                    content=_compose_system_instruction(
                        f"You are {name}, the review agent for a collaborative workflow.",
                        f"Your instructions: {system_prompt}",
                    )
                ),
                HumanMessage(content=review_context),
            ]
        else:
            instruction_builder = (
                _compose_brainstorm_instruction
                if str(agent_config.get("cluster_strategy") or "") == "brainstorm"
                else _compose_system_instruction
            )
            sys_msg = SystemMessage(
                content=instruction_builder(
                    f"You are {name}, a specialist agent in a collaborative workflow.",
                    f"Your instructions: {system_prompt}",
                )
            )
            task_context = f"The overall user task is: {state.get('task')}\n"
            capability_brief = str(agent_config.get("capability_brief") or "").strip()
            if capability_brief:
                task_context += (
                    "\nYour approved collaboration capabilities are:\n"
                    f"{capability_brief}\n"
                )
            capability_execution_contract = str(agent_config.get("capability_execution_contract") or "").strip()
            if capability_execution_contract:
                task_context += (
                    "\nExecution contract for this node:\n"
                    f"{capability_execution_contract}\n"
                    "Treat this contract as the hard boundary between planning metadata and actually executable actions.\n"
                )
            delegation_lane_ids = [
                str(item) for item in agent_config.get("delegation_lane_ids") or [] if str(item).strip()
            ]
            if delegation_lane_ids:
                task_context += (
                    "\nStructured delegation lanes for this node:\n"
                    f"{', '.join(delegation_lane_ids)}\n"
                )
            delegation_focus = str(agent_config.get("delegation_focus") or "").strip()
            if delegation_focus:
                task_context += (
                    "\nPreferred delegation lane for this node:\n"
                    f"{delegation_focus}\n"
                    "Lean into this lane when deciding whether to work directly or hand off to a teammate with a better fit.\n"
                )
            structured_delegation_contract = str(agent_config.get("structured_delegation_contract") or "").strip()
            if structured_delegation_contract:
                task_context += (
                    "\nStructured delegation contract for this node:\n"
                    f"{structured_delegation_contract}\n"
                    "Use this contract to decide whether this node should coordinate, execute, verify, or close the loop itself.\n"
                )
            recommended_collaborators = [
                dict(item)
                for item in agent_config.get("recommended_collaborators") or []
                if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
            ]
            if recommended_collaborators:
                task_context += (
                    "\nBest-fit collaborators already identified from the current canvas:\n"
                    + "\n".join(f"- {_format_delegation_partner(item)}" for item in recommended_collaborators[:3])
                    + "\n"
                )
            skill_guidance = str(agent_config.get("skill_guidance") or "").strip()
            if skill_guidance:
                task_context += (
                    "\nApproved skill pack guidance:\n"
                    f"{skill_guidance}\n"
                )
            mcp_guidance = str(agent_config.get("mcp_guidance") or "").strip()
            if mcp_guidance:
                task_context += (
                    "\nRelevant project MCP inventory for this node:\n"
                    f"{mcp_guidance}\n"
                    "Treat this MCP inventory as planning metadata only. "
                    "Do not claim an MCP server was executed unless the runtime explicitly reports that execution. "
                    "If missing MCP inventory blocks the task, call out the capability gap directly.\n"
                )
            orchestration_summary_brief = str(agent_config.get("orchestration_summary_brief") or "").strip()
            if orchestration_summary_brief:
                task_context += (
                    "\nCurrent graph orchestration brief:\n"
                    f"{orchestration_summary_brief}\n"
                    "Use this brief to decide who should coordinate, where parallel fan-out is safe, and which lane should close the loop.\n"
                )
            capability_roster = str(agent_config.get("team_capability_roster") or "").strip()
            if capability_roster:
                task_context += (
                    "\nTeam capability map:\n"
                    f"{capability_roster}\n"
                    "Choose handoffs that fit those capabilities instead of assuming every node can do every kind of work.\n"
                )
            collaboration_contract = str(agent_config.get("collaboration_contract") or "").strip()
            if collaboration_contract:
                task_context += (
                    "\nYour collaboration contract on this canvas is:\n"
                    f"{collaboration_contract}\n"
                )
            tool_guidance = str(agent_config.get("tool_guidance") or "").strip() if runtime_enabled_tool_ids else ""
            if tool_guidance:
                task_context += (
                    "\nExecutable tools available from this node:\n"
                    f"{tool_guidance}\n"
                    "Only call these tools when they materially improve the answer. If a tool is not listed here, do not assume it is executable.\n"
                )
            elif configured_enabled_tool_ids and runtime_tool_support_reason:
                task_context += (
                    "\nTool execution is configured for this node, but the current runtime cannot honor native tool calls.\n"
                    f"Reason: {runtime_tool_support_reason}\n"
                )
            brainstorm_round_context = _build_brainstorm_round_context(agent_config=agent_config, state=state)
            if brainstorm_round_context:
                task_context += f"\n{brainstorm_round_context}\n"
            knowledge_context = str(state.get("knowledge_context") or "").strip()
            if knowledge_context:
                task_context += (
                    "\nProject knowledge base context is available below. "
                    "Treat it as reference material retrieved from the project's selected knowledge bases. "
                    "Use it when relevant, but still verify assumptions from first principles.\n"
                    f"{knowledge_context}\n"
                )
            prior_research_context = _build_prior_research_context(agent_id=agent_id, state=state)
            if prior_research_context:
                task_context += (
                    "\nExternal research evidence from earlier clusters is available. "
                    "Use it to refine your reasoning, but still verify assumptions from first principles.\n"
                    f"{prior_research_context}\n"
                )
            upstream_handoff_context = _build_upstream_handoff_context(
                agent_id=agent_id,
                agent_directory=dict(agent_config.get("agent_directory") or {}),
                incoming_edges=list(agent_config.get("incoming_edges") or []),
                state=state,
            )
            if upstream_handoff_context:
                task_context += f"\n{upstream_handoff_context}\n"
            if continuation_context:
                task_context += (
                    "\nThis node is resuming from an approved partial output that was interrupted by live review. "
                    "Continue from the exact end of that approved prefix without restarting, repeating, or summarizing the prefix.\n"
                    f"\nApproved partial output:\n{continuation_context['partial_output']}\n"
                )
                if str(continuation_context.get("review_output") or "").strip():
                    task_context += f"\nReview note:\n{str(continuation_context.get('review_output') or '').strip()}\n"
            messages = [sys_msg, HumanMessage(content=task_context)] + state.get("messages", [])

        content = ""
        error_msg = None
        tool_runs: list[dict[str, Any]] = []
        try:
            with anyio.fail_after(timeout):
                content, tool_runs, _ = await _invoke_llm_with_tool_loop(
                    llm,
                    messages,
                    enabled_tools=enabled_tools,
                    timeout_seconds=timeout,
                    on_stream_chunk=None if is_review else on_stream_chunk,
                    on_stream_event=None if is_review else on_stream_event,
                )
        except OutputGuardrailTrip:
            raise
        except TimeoutError:
            error_msg = f"Agent {name} ({agent_id}) timed out after {timeout} seconds."
            content = f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Agent {name} ({agent_id}) execution failed: {e!s}"
            content = f"Error: {error_msg}"

        if continuation_context and not is_review and not is_cluster_summary and not error_msg:
            content = _merge_continuation_output(
                str(continuation_context.get("partial_output") or ""),
                content,
            )

        new_outputs = state.get("agent_outputs", {}).copy()
        updated_artifacts = state.get("output_artifacts", {}).copy()
        if is_review:
            new_outputs["review_agent"] = content
        else:
            new_outputs[agent_id] = content
            updated_artifacts[agent_id] = _build_agent_output_artifact(
                agent_config=agent_config,
                content=content,
                incoming_edges=list(agent_config.get("incoming_edges") or []),
                outgoing_edges=list(agent_config.get("outgoing_edges") or []),
                agent_directory=dict(agent_config.get("agent_directory") or {}),
                state=state,
                tool_runs=tool_runs,
            )
        errors = list(state.get("errors", []))
        if error_msg:
            _log.warning(error_msg)
            errors.append(error_msg)

        return {
            "current_agent": agent_id if not is_review else "review_agent",
            "agent_outputs": new_outputs,
            "output_artifacts": updated_artifacts,
            "errors": errors,
            "messages": [AIMessage(content=f"[{name}] says:\n{content}")],
        }

    return _invoke_agent


def compile_orchestration_graph(
    graph_json: dict[str, Any],
    *,
    task: str,
    loop_count: int = 1,
    default_timeout: int = 60,
    provider_registry: ModelProviderRegistry | None = None,
):
    """
    Compiles a Harness Studio JSON definition into an executable LangGraph StateGraph.
    Uses topological sorting to determine linear execution order.
    """
    builder = StateGraph(OrchestrationState)

    agents = graph_json.get("agents") or []
    edges = graph_json.get("edges") or []
    provider_config = graph_json.get("provider_config") or {}
    review_config = graph_json.get("review_agent") or {}
    registry = provider_registry or get_provider_registry()
    agents, edges = _expand_cluster_agents(agents, edges)
    capability_summary_by_agent_id = {
        str(item.get("agent_id") or ""): dict(item)
        for item in graph_json.get("agent_capability_summaries") or []
        if isinstance(item, dict) and str(item.get("agent_id") or "").strip()
    }
    skill_catalog_by_id = {
        _normalize_skill_key(str(item.get("skill_id") or "")): dict(item)
        for item in graph_json.get("skill_catalog") or []
        if isinstance(item, dict) and _normalize_skill_key(str(item.get("skill_id") or ""))
    }
    tool_catalog_by_id = _build_tool_catalog_lookup(graph_json)
    mcp_catalog_by_id = _build_mcp_catalog_lookup(graph_json)
    team_capability_roster = _build_team_capability_roster(
        agents=agents,
        capability_summary_by_agent_id=capability_summary_by_agent_id,
    )
    orchestration_summary_brief = _build_orchestration_routing_summary(
        graph_json.get("orchestration_summary") if isinstance(graph_json.get("orchestration_summary"), dict) else None,
        expected_agent_count=len(agents),
    )

    if not agents:
        # Empty graph fallback
        async def empty_node(state: OrchestrationState):
            return {"errors": ["No visible agents on canvas"]}
        builder.add_node("empty", cast(Any, empty_node))
        builder.add_edge(START, "empty")
        builder.add_edge("empty", END)
        return builder.compile()

    execution_order = _topological_sort(agents, edges)
    execution_stages = _build_execution_stages(agents, edges)
    stage_index_by_agent_id = {
        agent_id: index
        for index, stage in enumerate(execution_stages)
        for agent_id in stage
    }
    agent_by_id = {str(a["agent_id"]): a for a in agents}

    # 1. Build Nodes
    node_names = []
    for aid in execution_order:
        agent_conf = agent_by_id.get(aid)
        if not agent_conf:
            continue
        summary = _resolve_agent_capability_summary(
            agent=agent_conf,
            capability_summary_by_agent_id=capability_summary_by_agent_id,
        )
        enriched_agent_conf = dict(agent_conf)
        if summary:
            enabled_tool_ids = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
            mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
            missing_mcp_server_ids = [
                str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()
            ]
            delegation_lane_ids = [str(item) for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
            delegation_focus = str(summary.get("delegation_focus") or "").strip()
            delegation_contract = (
                dict(summary.get("delegation_contract") or {})
                if isinstance(summary.get("delegation_contract"), dict)
                else {}
            )
            readiness_status = str(summary.get("readiness_status") or "").strip() or "ready"
            readiness_blockers = [
                str(item) for item in summary.get("readiness_blockers") or [] if str(item).strip()
            ]
            readiness_warnings = [
                str(item) for item in summary.get("readiness_warnings") or [] if str(item).strip()
            ]
            enriched_agent_conf["capability_brief"] = str(summary.get("capability_brief") or "")
            enriched_agent_conf["capability_execution_contract"] = _build_capability_execution_contract(summary)
            enriched_agent_conf["delegation_lane_ids"] = delegation_lane_ids
            enriched_agent_conf["delegation_focus"] = delegation_focus
            enriched_agent_conf["delegation_contract"] = delegation_contract
            enriched_agent_conf["structured_delegation_contract"] = _build_structured_delegation_contract(
                delegation_contract
            )
            enriched_agent_conf["readiness_status"] = readiness_status
            enriched_agent_conf["readiness_blockers"] = readiness_blockers
            enriched_agent_conf["readiness_warnings"] = readiness_warnings
            enriched_agent_conf["recommended_collaborators"] = [
                dict(item) for item in summary.get("recommended_collaborators") or [] if isinstance(item, dict)
            ]
            enriched_agent_conf["downstream_handoff_scores"] = [
                dict(item) for item in summary.get("downstream_handoff_scores") or [] if isinstance(item, dict)
            ]
            enriched_agent_conf["skill_guidance"] = _build_skill_guidance_block(
                loaded_skill_ids=[str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()],
                skill_catalog_by_id=skill_catalog_by_id,
            )
            enriched_agent_conf["enabled_tool_ids"] = enabled_tool_ids
            enriched_agent_conf["mcp_server_ids"] = mcp_server_ids
            enriched_agent_conf["missing_mcp_server_ids"] = missing_mcp_server_ids
            enriched_agent_conf["tool_guidance"] = _build_tool_guidance_block(
                enabled_tool_ids=enabled_tool_ids,
                tool_catalog_by_id=tool_catalog_by_id,
            )
            enriched_agent_conf["mcp_guidance"] = _build_mcp_guidance_block(
                mcp_server_ids=mcp_server_ids,
                missing_mcp_server_ids=missing_mcp_server_ids,
                mcp_catalog_by_id=mcp_catalog_by_id,
            )
        enriched_agent_conf["team_capability_roster"] = team_capability_roster
        enriched_agent_conf["orchestration_summary_brief"] = orchestration_summary_brief
        enriched_agent_conf["execution_stage_index"] = stage_index_by_agent_id.get(str(aid), 0)
        enriched_agent_conf["agent_directory"] = agent_by_id
        enriched_agent_conf["incoming_edges"] = [
            dict(edge)
            for edge in edges
            if str(edge.get("target_agent_id") or "").strip() == str(aid)
        ]
        enriched_agent_conf["outgoing_edges"] = [
            dict(edge)
            for edge in edges
            if str(edge.get("source_agent_id") or "").strip() == str(aid)
        ]
        enriched_agent_conf["collaboration_contract"] = _build_agent_collaboration_contract(
            agent_id=str(aid),
            agents_by_id=agent_by_id,
            edges=edges,
            capability_summary_by_agent_id=capability_summary_by_agent_id,
        )
        node_name = f"agent_{aid}"
        node_names.append(node_name)
        builder.add_node(
            node_name,
            cast(Any, _build_agent_node(enriched_agent_conf, provider_config, default_timeout, registry)),
        )

    review_enabled = bool(review_config.get("enabled", True))
    if review_enabled:
        review_agent = {
            "agent_id": "review_agent",
            "name": str(review_config.get("name") or "Reviewer"),
            "system_prompt": str(review_config.get("system_prompt") or ""),
            "model": str(review_config.get("model") or "gpt-5.1-codex-mini"),
            "preferred_provider_id": review_config.get("preferred_provider_id"),
            "fallback_provider_id": review_config.get("fallback_provider_id"),
            "temperature": float(review_config.get("temperature", 0)),
            "timeout_seconds": review_config.get("timeout_seconds"),
            "team_capability_roster": team_capability_roster,
            "orchestration_summary_brief": orchestration_summary_brief,
        }
        builder.add_node(
            "review_agent",
            cast(
                Any,
                _build_agent_node(review_agent, provider_config, default_timeout, registry, is_review=True),
            ),
        )

    # 2. Build Graph Edges
    if node_names:
        incoming_counts = {str(agent.get("agent_id") or ""): 0 for agent in agents}
        outgoing_counts = {str(agent.get("agent_id") or ""): 0 for agent in agents}
        for edge in edges:
            source_agent_id = str(edge.get("source_agent_id") or "").strip()
            target_agent_id = str(edge.get("target_agent_id") or "").strip()
            if source_agent_id in outgoing_counts:
                outgoing_counts[source_agent_id] += 1
            if target_agent_id in incoming_counts:
                incoming_counts[target_agent_id] += 1
            if source_agent_id and target_agent_id:
                builder.add_edge(f"agent_{source_agent_id}", f"agent_{target_agent_id}")

        start_agent_ids = [
            str(agent_id)
            for agent_id in execution_order
            if incoming_counts.get(str(agent_id), 0) == 0
        ] or [execution_order[0]]
        terminal_agent_ids = [
            str(agent_id)
            for agent_id in execution_order
            if outgoing_counts.get(str(agent_id), 0) == 0
        ] or [execution_order[-1]]

        for agent_id in start_agent_ids:
            builder.add_edge(START, f"agent_{agent_id}")

        terminal_node_name = None
        if review_enabled:
            for agent_id in terminal_agent_ids:
                builder.add_edge(f"agent_{agent_id}", "review_agent")
            end_or_loop_node = "review_agent"
        elif len(terminal_agent_ids) > 1:
            async def terminal_join(state: OrchestrationState):
                return {}

            terminal_node_name = "terminal_join"
            builder.add_node(terminal_node_name, cast(Any, terminal_join))
            for agent_id in terminal_agent_ids:
                builder.add_edge(f"agent_{agent_id}", terminal_node_name)
            end_or_loop_node = terminal_node_name
        else:
            end_or_loop_node = f"agent_{terminal_agent_ids[0]}"

        # 3. Add looping logic
        def get_loop_router(max_loops: int):
            def route(state: OrchestrationState) -> str:
                current_loop = int(state.get("loop_index", 0)) + 1
                if current_loop < max_loops:
                    return "loop_bump"
                return END
            return route

        if loop_count > 1:
            async def bump_loop(state: OrchestrationState):
                return {"loop_index": int(state.get("loop_index", 0)) + 1}
            builder.add_node("loop_bump", cast(Any, bump_loop))
            builder.add_conditional_edges(end_or_loop_node, get_loop_router(loop_count), ["loop_bump", END])
            for agent_id in start_agent_ids:
                builder.add_edge("loop_bump", f"agent_{agent_id}")
        else:
            builder.add_edge(end_or_loop_node, END)

    return builder.compile()
