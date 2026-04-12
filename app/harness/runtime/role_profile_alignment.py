from __future__ import annotations

from typing import Any

_ROLE_PROFILE_IDS = {"coordinator", "research", "implementation", "verification", "generalist"}
_NON_SPECIFIC_LANE_IDS = {"generalist", "reasoning_only"}
_PROFILE_SPLIT_LANE_HINTS: dict[str, list[str]] = {
    "coordinator": ["coordination", "memory", "grounding"],
    "research": ["research", "grounding", "repository"],
    "implementation": ["implementation", "repository", "grounding"],
    "verification": ["verification", "grounding", "repository"],
    "generalist": ["grounding", "implementation", "verification"],
}


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_string(item)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_profile_id(value: Any) -> str:
    profile_id = _normalize_string(value) or "generalist"
    if profile_id in _ROLE_PROFILE_IDS:
        return profile_id
    return "generalist"


def _normalize_preview(agent_id: str, agent_name: str) -> dict[str, str]:
    return {
        "agent_id": agent_id,
        "agent_name": agent_name or agent_id,
    }


def _pick_focus_lane_ids(
    *,
    profile_id: str,
    shared_lane_ids: list[str],
    preferred_lane_ids: list[str],
    taken_lane_ids: set[str] | None = None,
) -> list[str]:
    taken = set(taken_lane_ids or set())
    candidate_lane_ids: list[str] = []
    for lane_id in [*preferred_lane_ids, *shared_lane_ids, *(_PROFILE_SPLIT_LANE_HINTS.get(profile_id) or [])]:
        normalized_lane_id = _normalize_string(lane_id)
        if not normalized_lane_id or normalized_lane_id in taken or normalized_lane_id in candidate_lane_ids:
            continue
        candidate_lane_ids.append(normalized_lane_id)
    return candidate_lane_ids[:1]


def build_role_profile_alignment_diagnostics(
    agent_capabilities: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    normalized_agents: list[dict[str, Any]] = []
    drift_agents: list[dict[str, Any]] = []

    for item in agent_capabilities or []:
        if not isinstance(item, dict):
            continue
        agent_id = _normalize_string(item.get("agent_id"))
        agent_name = _normalize_string(item.get("agent_name") or agent_id) or agent_id
        if not agent_id or not agent_name:
            continue
        if not isinstance(item.get("role_profile_suggestion"), dict):
            continue
        role_profile = dict(item.get("role_profile_suggestion") or {})
        profile_id = _normalize_profile_id(role_profile.get("profile_id"))
        available_skill_ids = _normalize_string_list(role_profile.get("available_skill_ids"))
        suggested_tool_ids = _normalize_string_list(role_profile.get("suggested_tool_ids"))
        suggested_mcp_server_ids = _normalize_string_list(role_profile.get("suggested_mcp_server_ids"))
        restrictive_tool_ids = _normalize_string_list(role_profile.get("restrictive_tool_ids"))
        restrictive_mcp_server_ids = _normalize_string_list(role_profile.get("restrictive_mcp_server_ids"))
        loaded_skill_ids = set(_normalize_string_list(item.get("loaded_skill_ids")))
        current_tool_ids = {
            *_normalize_string_list(item.get("enabled_tool_ids")),
            *_normalize_string_list(item.get("provider_limited_tool_ids")),
        }
        current_mcp_server_ids = set(_normalize_string_list(item.get("mcp_server_ids")))
        denied_tool_ids = {
            *_normalize_string_list(item.get("configured_denied_tool_ids")),
            *_normalize_string_list(item.get("denied_tool_ids")),
        }
        denied_mcp_server_ids = {
            *_normalize_string_list(item.get("configured_denied_mcp_server_ids")),
            *_normalize_string_list(item.get("denied_mcp_server_ids")),
        }
        delegation_lane_ids = [
            lane_id
            for lane_id in _normalize_string_list(item.get("delegation_lane_ids"))
            if lane_id not in _NON_SPECIFIC_LANE_IDS
        ]

        missing_skill_ids = [
            skill_id for skill_id in available_skill_ids if skill_id not in loaded_skill_ids
        ]
        missing_tool_ids = [
            tool_id for tool_id in suggested_tool_ids if tool_id not in current_tool_ids
        ]
        missing_mcp_server_ids = [
            server_id
            for server_id in suggested_mcp_server_ids
            if server_id not in current_mcp_server_ids
        ]
        outstanding_restrictive_tool_ids = [
            tool_id for tool_id in restrictive_tool_ids if tool_id not in denied_tool_ids
        ]
        outstanding_restrictive_mcp_server_ids = [
            server_id
            for server_id in restrictive_mcp_server_ids
            if server_id not in denied_mcp_server_ids
        ]

        normalized_agent = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "profile_id": profile_id,
            "delegation_lane_ids": delegation_lane_ids,
            "loaded_skill_ids": _normalize_string_list(item.get("loaded_skill_ids")),
            "tool_ids": sorted(current_tool_ids),
            "mcp_server_ids": _normalize_string_list(item.get("mcp_server_ids")),
        }
        normalized_agents.append(normalized_agent)

        if (
            missing_skill_ids
            or missing_tool_ids
            or missing_mcp_server_ids
            or outstanding_restrictive_tool_ids
            or outstanding_restrictive_mcp_server_ids
        ):
            drift_agents.append(
                {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "profile_id": profile_id,
                    "missing_skill_ids": missing_skill_ids,
                    "missing_tool_ids": missing_tool_ids,
                    "missing_mcp_server_ids": missing_mcp_server_ids,
                    "outstanding_restrictive_tool_ids": outstanding_restrictive_tool_ids,
                    "outstanding_restrictive_mcp_server_ids": outstanding_restrictive_mcp_server_ids,
                }
            )

    drift_agents.sort(
        key=lambda item: (
            -(
                len(item.get("missing_skill_ids") or [])
                + len(item.get("missing_tool_ids") or [])
                + len(item.get("missing_mcp_server_ids") or [])
                + len(item.get("outstanding_restrictive_tool_ids") or [])
                + len(item.get("outstanding_restrictive_mcp_server_ids") or [])
            ),
            str(item.get("agent_name") or ""),
        )
    )

    overlap_risks: list[dict[str, Any]] = []
    for profile_id in _ROLE_PROFILE_IDS:
        profile_agents = [
            agent for agent in normalized_agents if str(agent.get("profile_id") or "") == profile_id
        ]
        if len(profile_agents) < 2:
            continue
        for index, left in enumerate(profile_agents):
            left_lane_ids = set(_normalize_string_list(left.get("delegation_lane_ids")))
            left_skill_ids = set(_normalize_string_list(left.get("loaded_skill_ids")))
            left_tool_ids = set(_normalize_string_list(left.get("tool_ids")))
            left_mcp_server_ids = set(_normalize_string_list(left.get("mcp_server_ids")))
            for right in profile_agents[index + 1 :]:
                right_lane_ids = set(_normalize_string_list(right.get("delegation_lane_ids")))
                right_skill_ids = set(_normalize_string_list(right.get("loaded_skill_ids")))
                right_tool_ids = set(_normalize_string_list(right.get("tool_ids")))
                right_mcp_server_ids = set(_normalize_string_list(right.get("mcp_server_ids")))

                shared_lane_ids = sorted(left_lane_ids & right_lane_ids)
                if not shared_lane_ids:
                    continue
                unique_lane_ids = sorted(left_lane_ids ^ right_lane_ids)
                unique_skill_ids = sorted(left_skill_ids ^ right_skill_ids)
                unique_tool_ids = sorted(left_tool_ids ^ right_tool_ids)
                unique_mcp_server_ids = sorted(left_mcp_server_ids ^ right_mcp_server_ids)
                capability_delta_count = (
                    len(unique_skill_ids) + len(unique_tool_ids) + len(unique_mcp_server_ids)
                )

                is_overlap_risk = False
                if len(shared_lane_ids) >= 2 and len(unique_lane_ids) <= 1 and capability_delta_count <= 1:
                    is_overlap_risk = True
                elif not unique_lane_ids and capability_delta_count == 0:
                    is_overlap_risk = True
                elif (
                    profile_id == "coordinator"
                    and "coordination" in shared_lane_ids
                    and len(unique_lane_ids) == 0
                    and capability_delta_count <= 1
                ):
                    is_overlap_risk = True

                if not is_overlap_risk:
                    continue

                left_unique_lane_ids = sorted(left_lane_ids - right_lane_ids)
                right_unique_lane_ids = sorted(right_lane_ids - left_lane_ids)
                left_unique_skill_ids = sorted(left_skill_ids - right_skill_ids)
                right_unique_skill_ids = sorted(right_skill_ids - left_skill_ids)
                left_unique_tool_ids = sorted(left_tool_ids - right_tool_ids)
                right_unique_tool_ids = sorted(right_tool_ids - left_tool_ids)
                left_unique_mcp_server_ids = sorted(left_mcp_server_ids - right_mcp_server_ids)
                right_unique_mcp_server_ids = sorted(right_mcp_server_ids - left_mcp_server_ids)
                left_focus_lane_ids = _pick_focus_lane_ids(
                    profile_id=profile_id,
                    shared_lane_ids=shared_lane_ids,
                    preferred_lane_ids=left_unique_lane_ids,
                )
                right_focus_lane_ids = _pick_focus_lane_ids(
                    profile_id=profile_id,
                    shared_lane_ids=shared_lane_ids,
                    preferred_lane_ids=right_unique_lane_ids,
                    taken_lane_ids=set(left_focus_lane_ids),
                )

                overlap_risks.append(
                    {
                        "profile_id": profile_id,
                        "left_agent_preview": _normalize_preview(
                            str(left.get("agent_id") or ""),
                            str(left.get("agent_name") or left.get("agent_id") or ""),
                        ),
                        "right_agent_preview": _normalize_preview(
                            str(right.get("agent_id") or ""),
                            str(right.get("agent_name") or right.get("agent_id") or ""),
                        ),
                        "agent_previews": [
                            _normalize_preview(
                                str(left.get("agent_id") or ""),
                                str(left.get("agent_name") or left.get("agent_id") or ""),
                            ),
                            _normalize_preview(
                                str(right.get("agent_id") or ""),
                                str(right.get("agent_name") or right.get("agent_id") or ""),
                            ),
                        ],
                        "shared_lane_ids": shared_lane_ids,
                        "unique_lane_ids": unique_lane_ids,
                        "left_unique_lane_ids": left_unique_lane_ids,
                        "right_unique_lane_ids": right_unique_lane_ids,
                        "left_unique_skill_ids": left_unique_skill_ids,
                        "right_unique_skill_ids": right_unique_skill_ids,
                        "left_unique_tool_ids": left_unique_tool_ids,
                        "right_unique_tool_ids": right_unique_tool_ids,
                        "left_unique_mcp_server_ids": left_unique_mcp_server_ids,
                        "right_unique_mcp_server_ids": right_unique_mcp_server_ids,
                        "left_focus_lane_ids": left_focus_lane_ids,
                        "right_focus_lane_ids": right_focus_lane_ids,
                    }
                )

    overlap_risks.sort(
        key=lambda item: (
            -len(item.get("shared_lane_ids") or []),
            len(item.get("unique_lane_ids") or []),
            str(item.get("profile_id") or ""),
            ",".join(
                str(preview.get("agent_name") or "")
                for preview in item.get("agent_previews") or []
                if isinstance(preview, dict)
            ),
        )
    )

    return {
        "drift_agent_count": len(drift_agents),
        "overlap_risk_count": len(overlap_risks),
        "drift_agents": drift_agents,
        "overlap_risks": overlap_risks,
    }
