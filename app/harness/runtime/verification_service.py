from __future__ import annotations

from typing import Any

from app.harness.runtime.role_profile_alignment import build_role_profile_alignment_diagnostics


class VerificationService:
    _ROLE_PROFILE_IDS = {
        "coordinator",
        "research",
        "implementation",
        "verification",
        "generalist",
    }

    @staticmethod
    def _normalize_string(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = cls._normalize_string(item)
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @classmethod
    def _normalize_collaborator_fit(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        agent_id = cls._normalize_string(value.get("agent_id"))
        agent_name = cls._normalize_string(value.get("agent_name") or agent_id)
        if not agent_id or not agent_name:
            return None
        try:
            score = int(value.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        fit = cls._normalize_string(value.get("fit") or "weak") or "weak"
        normalized: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "score": max(0, min(score, 100)),
            "fit": fit if fit in {"strong", "good", "weak"} else "weak",
        }
        rationale = cls._normalize_string(value.get("rationale"))
        if rationale:
            normalized["rationale"] = rationale
        new_skill_ids = cls._normalize_string_list(value.get("new_skill_ids"))
        if new_skill_ids:
            normalized["new_skill_ids"] = new_skill_ids
        overlap_lane_ids = cls._normalize_string_list(value.get("overlap_lane_ids"))
        if overlap_lane_ids:
            normalized["overlap_lane_ids"] = overlap_lane_ids
        complementary_lane_ids = cls._normalize_string_list(value.get("complementary_lane_ids"))
        if complementary_lane_ids:
            normalized["complementary_lane_ids"] = complementary_lane_ids
        new_tool_ids = cls._normalize_string_list(value.get("new_tool_ids"))
        if new_tool_ids:
            normalized["new_tool_ids"] = new_tool_ids
        new_mcp_server_ids = cls._normalize_string_list(value.get("new_mcp_server_ids"))
        if new_mcp_server_ids:
            normalized["new_mcp_server_ids"] = new_mcp_server_ids
        gap_cover_mcp_server_ids = cls._normalize_string_list(value.get("gap_cover_mcp_server_ids"))
        if gap_cover_mcp_server_ids:
            normalized["gap_cover_mcp_server_ids"] = gap_cover_mcp_server_ids
        source_profile_id = cls._normalize_string(value.get("source_profile_id"))
        if source_profile_id in cls._ROLE_PROFILE_IDS:
            normalized["source_profile_id"] = source_profile_id
        target_profile_id = cls._normalize_string(value.get("target_profile_id"))
        if target_profile_id in cls._ROLE_PROFILE_IDS:
            normalized["target_profile_id"] = target_profile_id
        if value.get("same_role_profile") is not None:
            normalized["same_role_profile"] = bool(value.get("same_role_profile"))
        if value.get("same_role_profile_overlap_risk") is not None:
            normalized["same_role_profile_overlap_risk"] = bool(value.get("same_role_profile_overlap_risk"))
        if value.get("edge_present") is not None:
            normalized["edge_present"] = bool(value.get("edge_present"))
        interaction = cls._normalize_string(value.get("interaction"))
        if interaction:
            normalized["interaction"] = interaction
        return normalized

    @classmethod
    def _normalize_record_list(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(dict(item))
        return normalized

    @classmethod
    def _normalize_coordination_preview(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        agent_id = cls._normalize_string(value.get("agent_id"))
        agent_name = cls._normalize_string(value.get("agent_name") or agent_id)
        if not agent_id or not agent_name:
            return None
        return {
            "agent_id": agent_id,
            "agent_name": agent_name,
        }

    @classmethod
    def _normalize_coordination_preview_list(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value:
            preview = cls._normalize_coordination_preview(item)
            if preview is None:
                continue
            agent_id = str(preview.get("agent_id") or "").strip()
            if agent_id in seen:
                continue
            seen.add(agent_id)
            normalized.append(preview)
        return normalized

    @classmethod
    def _normalize_delegation_role_mode(cls, value: Any) -> str:
        role_mode = cls._normalize_string(value) or "generalist"
        if role_mode in {"coordinator", "research", "implementation", "verification", "generalist"}:
            return role_mode
        return "generalist"

    @classmethod
    def _normalize_delegation_work_strategy(cls, value: Any) -> str:
        work_strategy = cls._normalize_string(value) or "flexible"
        if work_strategy in {
            "synthesize_and_route",
            "gather_then_handoff",
            "implement_then_handoff",
            "verify_and_close",
            "self_contained_delivery",
            "flexible",
        }:
            return work_strategy
        return "flexible"

    @classmethod
    def _format_identifier_preview(cls, values: list[str], *, limit: int = 2) -> str:
        normalized = [cls._normalize_string(value) for value in values if cls._normalize_string(value)]
        if not normalized:
            return "none"
        preview = normalized[:limit]
        suffix = f", +{len(normalized) - limit} more" if len(normalized) > limit else ""
        return ", ".join(preview) + suffix

    @classmethod
    def _format_named_capability_examples(
        cls,
        examples: list[tuple[str, list[str]]],
        *,
        limit: int = 3,
    ) -> str:
        if not examples:
            return "none"
        preview_items: list[str] = []
        for agent_name, capability_ids in examples[:limit]:
            normalized_name = cls._normalize_string(agent_name) or "Unknown node"
            capability_preview = cls._format_identifier_preview(capability_ids)
            preview_items.append(f"{normalized_name} ({capability_preview})")
        suffix = f"; +{len(examples) - limit} more" if len(examples) > limit else ""
        return "; ".join(preview_items) + suffix

    @classmethod
    def _build_recovery_actions(
        cls,
        *,
        open_skill_pool: bool = False,
        open_project_mcp_inventory: bool = False,
        open_project_providers: bool = False,
    ) -> dict[str, bool]:
        actions: dict[str, bool] = {"focus_agent": True}
        if open_skill_pool:
            actions["open_skill_pool"] = True
        if open_project_mcp_inventory:
            actions["open_project_mcp_inventory"] = True
        if open_project_providers:
            actions["open_project_providers"] = True
        return actions

    @classmethod
    def _build_handoff_diagnostics(
        cls,
        *,
        active_agent_ids: list[str],
        capability_snapshot: dict[str, object] | None,
    ) -> dict[str, Any]:
        if not isinstance(capability_snapshot, dict):
            return {}

        active_ids = set(cls._normalize_string_list(active_agent_ids))
        weak_edges: list[dict[str, Any]] = []
        best_next_handoffs: list[dict[str, Any]] = []
        agent_capabilities = cls._normalize_record_list(capability_snapshot.get("agent_capabilities"))

        for agent_capability in agent_capabilities:
            source_agent_id = cls._normalize_string(agent_capability.get("agent_id"))
            if not source_agent_id or (active_ids and source_agent_id not in active_ids):
                continue
            source_agent_name = cls._normalize_string(agent_capability.get("agent_name") or source_agent_id)
            source_lane_ids = cls._normalize_string_list(agent_capability.get("delegation_lane_ids"))
            delegation_focus = cls._normalize_string(agent_capability.get("delegation_focus"))
            recommendations = [
                normalized
                for item in agent_capability.get("recommended_collaborators") or []
                if (normalized := cls._normalize_collaborator_fit(item)) is not None
                and normalized["fit"] in {"strong", "good"}
                and not bool(normalized.get("edge_present"))
            ]
            recommendations.sort(
                key=lambda item: (
                    0 if item.get("fit") == "strong" else 1,
                    -int(item.get("score") or 0),
                    str(item.get("agent_name") or ""),
                )
            )
            if recommendations:
                best_next_handoffs.append(
                    {
                        "source_agent_id": source_agent_id,
                        "source_agent_name": source_agent_name,
                        "source_lane_ids": source_lane_ids,
                        "delegation_focus": delegation_focus or None,
                        "target": recommendations[0],
                    }
                )

            for item in agent_capability.get("downstream_handoff_scores") or []:
                handoff_fit = cls._normalize_collaborator_fit(item)
                if handoff_fit is None:
                    continue
                if handoff_fit.get("fit") != "weak" or not bool(handoff_fit.get("edge_present")):
                    continue
                suggested_replacements = [
                    recommendation
                    for recommendation in recommendations
                    if recommendation.get("agent_id") != handoff_fit.get("agent_id")
                ][:2]
                weak_edges.append(
                    {
                        "source_agent_id": source_agent_id,
                        "source_agent_name": source_agent_name,
                        "source_lane_ids": source_lane_ids,
                        "delegation_focus": delegation_focus or None,
                        "target": handoff_fit,
                        "suggested_replacements": suggested_replacements,
                    }
                )

        weak_edges.sort(
            key=lambda item: (
                int((item.get("target") or {}).get("score") or 0),
                str(item.get("source_agent_name") or ""),
                str((item.get("target") or {}).get("agent_name") or ""),
            )
        )
        best_next_handoffs.sort(
            key=lambda item: (
                0 if (item.get("target") or {}).get("fit") == "strong" else 1,
                -int((item.get("target") or {}).get("score") or 0),
                str(item.get("source_agent_name") or ""),
            )
        )

        if not weak_edges and not best_next_handoffs:
            return {}
        return {
            "weak_downstream_edges": weak_edges,
            "best_next_handoffs": best_next_handoffs,
            "weak_edge_count": len(weak_edges),
            "best_next_count": len(best_next_handoffs),
        }

    @classmethod
    def _build_readiness_diagnostics(
        cls,
        *,
        active_agent_ids: list[str],
        capability_snapshot: dict[str, object] | None,
    ) -> dict[str, Any]:
        if not isinstance(capability_snapshot, dict):
            return {}

        active_ids = set(cls._normalize_string_list(active_agent_ids))
        agents: list[dict[str, Any]] = []
        blocked_count = 0
        limited_count = 0
        agent_capabilities = cls._normalize_record_list(capability_snapshot.get("agent_capabilities"))

        for agent_capability in agent_capabilities:
            agent_id = cls._normalize_string(agent_capability.get("agent_id"))
            if not agent_id or (active_ids and agent_id not in active_ids):
                continue
            status = cls._normalize_string(agent_capability.get("readiness_status") or "ready") or "ready"
            if status not in {"ready", "limited", "blocked"}:
                status = "ready"
            blockers = cls._normalize_string_list(agent_capability.get("readiness_blockers"))
            warnings = cls._normalize_string_list(agent_capability.get("readiness_warnings"))
            missing_skill_ids = cls._normalize_string_list(agent_capability.get("missing_skill_ids"))
            missing_skill_details = cls._normalize_record_list(agent_capability.get("missing_skill_details"))
            missing_mcp_server_ids = cls._normalize_string_list(agent_capability.get("missing_mcp_server_ids"))
            provider_limited_tool_ids = cls._normalize_string_list(agent_capability.get("provider_limited_tool_ids"))
            requires_tool_calling = bool(agent_capability.get("requires_tool_calling"))
            tool_execution_support = cls._normalize_string(agent_capability.get("tool_execution_support"))
            tool_execution_support_reason = cls._normalize_string(agent_capability.get("tool_execution_support_reason"))
            provider_route = cls._normalize_string(agent_capability.get("provider_route"))
            if status == "blocked":
                blocked_count += 1
            elif status == "limited":
                limited_count += 1
            if status == "ready" and not blockers and not warnings:
                continue
            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_name": cls._normalize_string(agent_capability.get("agent_name") or agent_id) or agent_id,
                "status": status,
                "blockers": blockers,
                "warnings": warnings,
                "recovery_actions": cls._build_recovery_actions(
                    open_skill_pool=bool(missing_skill_ids or missing_skill_details),
                    open_project_mcp_inventory=bool(missing_mcp_server_ids),
                    open_project_providers=bool(
                        provider_limited_tool_ids
                        or (requires_tool_calling and tool_execution_support != "supported")
                    ),
                ),
            }
            if missing_skill_ids:
                payload["missing_skill_ids"] = missing_skill_ids
            if missing_skill_details:
                payload["missing_skill_details"] = missing_skill_details
            if missing_mcp_server_ids:
                payload["missing_mcp_server_ids"] = missing_mcp_server_ids
            if provider_limited_tool_ids:
                payload["provider_limited_tool_ids"] = provider_limited_tool_ids
            if requires_tool_calling:
                payload["requires_tool_calling"] = True
            if tool_execution_support:
                payload["tool_execution_support"] = tool_execution_support
            if tool_execution_support_reason:
                payload["tool_execution_support_reason"] = tool_execution_support_reason
            if provider_route:
                payload["provider_route"] = provider_route
            agents.append(payload)

        if not agents:
            return {}
        return {
            "blocked_count": blocked_count,
            "limited_count": limited_count,
            "agent_count": len(agents),
            "agents": agents,
        }

    @classmethod
    def _build_availability_diagnostics(
        cls,
        *,
        active_agent_ids: list[str],
        capability_snapshot: dict[str, object] | None,
    ) -> dict[str, Any]:
        if not isinstance(capability_snapshot, dict):
            return {}

        active_ids = set(cls._normalize_string_list(active_agent_ids))
        agents: list[dict[str, Any]] = []
        unavailable_count = 0
        limited_count = 0
        agent_capabilities = cls._normalize_record_list(capability_snapshot.get("agent_capabilities"))

        for agent_capability in agent_capabilities:
            agent_id = cls._normalize_string(agent_capability.get("agent_id"))
            if not agent_id or (active_ids and agent_id not in active_ids):
                continue
            status = cls._normalize_string(agent_capability.get("availability_status") or "available") or "available"
            if status not in {"available", "limited", "unavailable"}:
                missing_required_skills = cls._normalize_string_list(agent_capability.get("missing_required_skill_ids"))
                missing_required_tools = cls._normalize_string_list(agent_capability.get("missing_required_tool_ids"))
                missing_required_mcp = cls._normalize_string_list(agent_capability.get("missing_required_mcp_server_ids"))
                requires_tool_calling = bool(agent_capability.get("requires_tool_calling"))
                tool_execution_support = cls._normalize_string(agent_capability.get("tool_execution_support"))
                if (
                    missing_required_skills
                    or missing_required_tools
                    or missing_required_mcp
                    or (requires_tool_calling and tool_execution_support == "unsupported")
                ):
                    status = "unavailable"
                elif requires_tool_calling and tool_execution_support != "supported":
                    status = "limited"
                else:
                    status = "available"
            blockers = cls._normalize_string_list(agent_capability.get("availability_blockers"))
            warnings = cls._normalize_string_list(agent_capability.get("availability_warnings"))
            missing_required_skill_ids = cls._normalize_string_list(agent_capability.get("missing_required_skill_ids"))
            missing_required_tool_ids = cls._normalize_string_list(agent_capability.get("missing_required_tool_ids"))
            missing_required_mcp_server_ids = cls._normalize_string_list(
                agent_capability.get("missing_required_mcp_server_ids")
            )
            requires_tool_calling = bool(agent_capability.get("requires_tool_calling"))
            tool_execution_support = cls._normalize_string(agent_capability.get("tool_execution_support"))
            tool_execution_support_reason = cls._normalize_string(agent_capability.get("tool_execution_support_reason"))
            provider_route = cls._normalize_string(agent_capability.get("provider_route"))
            if status == "unavailable":
                unavailable_count += 1
            elif status == "limited":
                limited_count += 1
            if status == "available" and not blockers and not warnings:
                continue
            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_name": cls._normalize_string(agent_capability.get("agent_name") or agent_id) or agent_id,
                "status": status,
                "blockers": blockers,
                "warnings": warnings,
                "recovery_actions": cls._build_recovery_actions(
                    open_skill_pool=bool(missing_required_skill_ids),
                    open_project_mcp_inventory=bool(missing_required_mcp_server_ids),
                    open_project_providers=bool(
                        requires_tool_calling and tool_execution_support != "supported"
                    ),
                ),
            }
            if missing_required_skill_ids:
                payload["missing_required_skill_ids"] = missing_required_skill_ids
            if missing_required_tool_ids:
                payload["missing_required_tool_ids"] = missing_required_tool_ids
            if missing_required_mcp_server_ids:
                payload["missing_required_mcp_server_ids"] = missing_required_mcp_server_ids
            if requires_tool_calling:
                payload["requires_tool_calling"] = True
            if tool_execution_support:
                payload["tool_execution_support"] = tool_execution_support
            if tool_execution_support_reason:
                payload["tool_execution_support_reason"] = tool_execution_support_reason
            if provider_route:
                payload["provider_route"] = provider_route
            agents.append(payload)

        if not agents:
            return {}
        return {
            "unavailable_count": unavailable_count,
            "limited_count": limited_count,
            "agent_count": len(agents),
            "agents": agents,
        }

    @classmethod
    def _build_execution_contract_diagnostics(
        cls,
        *,
        active_agent_ids: list[str],
        capability_snapshot: dict[str, object] | None,
    ) -> dict[str, Any]:
        if not isinstance(capability_snapshot, dict):
            return {}

        active_ids = set(cls._normalize_string_list(active_agent_ids))
        agents: list[dict[str, Any]] = []
        executable_tool_agent_count = 0
        planning_only_tool_agent_count = 0
        planning_only_mcp_agent_count = 0
        direct_execution_agent_count = 0
        agent_capabilities = cls._normalize_record_list(capability_snapshot.get("agent_capabilities"))

        for agent_capability in agent_capabilities:
            agent_id = cls._normalize_string(agent_capability.get("agent_id"))
            if not agent_id or (active_ids and agent_id not in active_ids):
                continue
            execution_contract = (
                dict(agent_capability.get("execution_contract") or {})
                if isinstance(agent_capability.get("execution_contract"), dict)
                else {}
            )
            approved_skill_ids = cls._normalize_string_list(
                execution_contract.get("approved_skill_ids") or agent_capability.get("loaded_skill_ids")
            )
            suggested_skill_ids = cls._normalize_string_list(
                execution_contract.get("suggested_skill_ids") or agent_capability.get("suggested_skill_ids")
            )
            executable_tool_ids = cls._normalize_string_list(
                execution_contract.get("executable_tool_ids") or agent_capability.get("enabled_tool_ids")
            )
            planning_only_tool_ids = cls._normalize_string_list(
                execution_contract.get("planning_only_tool_ids") or agent_capability.get("provider_limited_tool_ids")
            )
            disabled_tool_ids = cls._normalize_string_list(
                execution_contract.get("disabled_tool_ids") or agent_capability.get("disabled_tool_ids")
            )
            planning_only_mcp_server_ids = cls._normalize_string_list(
                execution_contract.get("planning_only_mcp_server_ids") or agent_capability.get("mcp_server_ids")
            )
            missing_mcp_server_ids = cls._normalize_string_list(
                execution_contract.get("missing_mcp_server_ids") or agent_capability.get("missing_mcp_server_ids")
            )
            requires_tool_calling = bool(agent_capability.get("requires_tool_calling"))
            tool_execution_support = cls._normalize_string(agent_capability.get("tool_execution_support"))
            tool_execution_support_reason = cls._normalize_string(agent_capability.get("tool_execution_support_reason"))
            provider_route = cls._normalize_string(agent_capability.get("provider_route"))
            if tool_execution_support == "unsupported" and executable_tool_ids:
                planning_only_tool_ids = list(dict.fromkeys([*planning_only_tool_ids, *executable_tool_ids]))
                executable_tool_ids = []
            skill_execution_mode = cls._normalize_string(
                execution_contract.get("skill_execution_mode") or "guidance_only"
            ) or "guidance_only"
            tool_access_mode = cls._normalize_string(execution_contract.get("tool_access_mode"))
            if tool_access_mode not in {"direct_execution", "planning_only", "mixed", "none"}:
                if executable_tool_ids and planning_only_tool_ids:
                    tool_access_mode = "mixed"
                elif executable_tool_ids:
                    tool_access_mode = "direct_execution"
                elif planning_only_tool_ids:
                    tool_access_mode = "planning_only"
                else:
                    tool_access_mode = "none"
            mcp_access_mode = cls._normalize_string(execution_contract.get("mcp_access_mode"))
            if mcp_access_mode not in {"planning_only", "none"}:
                mcp_access_mode = "planning_only" if planning_only_mcp_server_ids else "none"

            if not (
                approved_skill_ids
                or suggested_skill_ids
                or executable_tool_ids
                or planning_only_tool_ids
                or planning_only_mcp_server_ids
                or missing_mcp_server_ids
                or requires_tool_calling
            ):
                continue

            if executable_tool_ids:
                executable_tool_agent_count += 1
            if planning_only_tool_ids:
                planning_only_tool_agent_count += 1
            if planning_only_mcp_server_ids:
                planning_only_mcp_agent_count += 1
            if tool_access_mode in {"direct_execution", "mixed"}:
                direct_execution_agent_count += 1

            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_name": cls._normalize_string(agent_capability.get("agent_name") or agent_id) or agent_id,
                "skill_execution_mode": skill_execution_mode,
                "approved_skill_ids": approved_skill_ids,
                "tool_access_mode": tool_access_mode,
                "executable_tool_ids": executable_tool_ids,
                "planning_only_tool_ids": planning_only_tool_ids,
                "mcp_access_mode": mcp_access_mode,
                "planning_only_mcp_server_ids": planning_only_mcp_server_ids,
                "missing_mcp_server_ids": missing_mcp_server_ids,
                "recovery_actions": cls._build_recovery_actions(
                    open_project_mcp_inventory=bool(missing_mcp_server_ids),
                    open_project_providers=bool(planning_only_tool_ids or (requires_tool_calling and tool_access_mode != "direct_execution")),
                ),
            }
            if suggested_skill_ids:
                payload["suggested_skill_ids"] = suggested_skill_ids
            if disabled_tool_ids:
                payload["disabled_tool_ids"] = disabled_tool_ids
            if requires_tool_calling:
                payload["requires_tool_calling"] = True
            if tool_execution_support:
                payload["tool_execution_support"] = tool_execution_support
            if tool_execution_support_reason:
                payload["tool_execution_support_reason"] = tool_execution_support_reason
            if provider_route:
                payload["provider_route"] = provider_route
            agents.append(payload)

        if not agents:
            return {}
        return {
            "agent_count": len(agents),
            "executable_tool_agent_count": executable_tool_agent_count,
            "planning_only_tool_agent_count": planning_only_tool_agent_count,
            "planning_only_mcp_agent_count": planning_only_mcp_agent_count,
            "direct_execution_agent_count": direct_execution_agent_count,
            "agents": agents,
        }

    @classmethod
    def _build_collaboration_contract_diagnostics(
        cls,
        *,
        active_agent_ids: list[str],
        capability_snapshot: dict[str, object] | None,
        review_agent_enabled: bool,
    ) -> dict[str, Any]:
        if not isinstance(capability_snapshot, dict):
            return {}

        active_ids = set(cls._normalize_string_list(active_agent_ids))
        agents: list[dict[str, Any]] = []
        coordinator_agents: list[dict[str, Any]] = []
        parallel_coordinator_agents: list[dict[str, Any]] = []
        final_output_agents: list[dict[str, Any]] = []
        verification_agents: list[dict[str, Any]] = []
        coordinator_execution_tool_risks: list[dict[str, Any]] = []
        coordinator_mcp_context_risks: list[dict[str, Any]] = []
        considered_agent_count = 0
        role_profile_agents: list[dict[str, Any]] = []
        agent_capabilities = cls._normalize_record_list(capability_snapshot.get("agent_capabilities"))

        def append_preview_once(target: list[dict[str, Any]], preview: dict[str, Any]) -> None:
            agent_id = cls._normalize_string(preview.get("agent_id"))
            if not agent_id:
                return
            if any(cls._normalize_string(item.get("agent_id")) == agent_id for item in target):
                return
            target.append(preview)

        def sort_collaborator_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                candidates,
                key=lambda item: (
                    0
                    if cls._normalize_string(item.get("fit")) == "strong"
                    else 1
                    if cls._normalize_string(item.get("fit")) == "good"
                    else 2,
                    -int(item.get("score") or 0),
                    cls._normalize_string(item.get("agent_name") or item.get("agent_id")),
                ),
            )

        def build_delegate_candidate_previews(
            source_preview: dict[str, Any],
            candidates: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            previews: list[dict[str, Any]] = []
            append_preview_once(previews, source_preview)
            for candidate in candidates:
                append_preview_once(
                    previews,
                    {
                        "agent_id": cls._normalize_string(candidate.get("agent_id")),
                        "agent_name": cls._normalize_string(
                            candidate.get("agent_name") or candidate.get("agent_id")
                        ),
                    },
                )
            return previews

        def build_delegate_action(
            *,
            capability_label: str,
            candidates: list[dict[str, Any]],
            fallback: str,
        ) -> str:
            candidate_names = [
                cls._normalize_string(candidate.get("agent_name") or candidate.get("agent_id"))
                for candidate in candidates
                if cls._normalize_string(candidate.get("agent_name") or candidate.get("agent_id"))
            ]
            if candidate_names:
                return (
                    f"Route {capability_label} through "
                    + cls._format_identifier_preview(candidate_names, limit=2)
                    + " before the next orchestration pass."
                )
            return fallback

        for agent_capability in agent_capabilities:
            agent_id = cls._normalize_string(agent_capability.get("agent_id"))
            if not agent_id or (active_ids and agent_id not in active_ids):
                continue
            considered_agent_count += 1

            agent_name = cls._normalize_string(agent_capability.get("agent_name") or agent_id) or agent_id
            delegation_contract = (
                dict(agent_capability.get("delegation_contract") or {})
                if isinstance(agent_capability.get("delegation_contract"), dict)
                else {}
            )
            execution_contract = (
                dict(agent_capability.get("execution_contract") or {})
                if isinstance(agent_capability.get("execution_contract"), dict)
                else {}
            )
            primary_role_mode = cls._normalize_delegation_role_mode(
                delegation_contract.get("primary_role_mode")
            )
            supporting_role_modes = [
                role_mode
                for role_mode in cls._normalize_string_list(delegation_contract.get("supporting_role_modes"))
                if role_mode in {"coordinator", "research", "implementation", "verification", "generalist"}
                and role_mode != primary_role_mode
            ]
            work_strategy = cls._normalize_delegation_work_strategy(
                delegation_contract.get("work_strategy")
            )
            should_coordinate_parallel_work = bool(
                delegation_contract.get("should_coordinate_parallel_work")
            )
            should_produce_final_output = bool(
                delegation_contract.get("should_produce_final_output")
            )
            tool_access_mode = cls._normalize_string(execution_contract.get("tool_access_mode"))
            mcp_access_mode = cls._normalize_string(execution_contract.get("mcp_access_mode"))
            executable_tool_ids = cls._normalize_string_list(execution_contract.get("executable_tool_ids"))
            if not executable_tool_ids and tool_access_mode in {"direct_execution", "mixed"}:
                executable_tool_ids = cls._normalize_string_list(agent_capability.get("enabled_tool_ids"))
            planning_only_tool_ids = cls._normalize_string_list(execution_contract.get("planning_only_tool_ids"))
            planning_only_mcp_server_ids = cls._normalize_string_list(
                execution_contract.get("planning_only_mcp_server_ids")
            )
            if not planning_only_mcp_server_ids and mcp_access_mode == "planning_only":
                planning_only_mcp_server_ids = cls._normalize_string_list(agent_capability.get("mcp_server_ids"))
            recommended_collaborators = sort_collaborator_candidates(
                [
                    normalized
                    for item in agent_capability.get("recommended_collaborators") or []
                    if (normalized := cls._normalize_collaborator_fit(item)) is not None
                ]
            )
            primary_focus = cls._normalize_string(delegation_contract.get("primary_focus"))
            upstream_agents = cls._normalize_coordination_preview_list(
                delegation_contract.get("upstream_agents")
            )
            downstream_agents = cls._normalize_coordination_preview_list(
                delegation_contract.get("downstream_agents")
            )
            preferred_collaborators = cls._normalize_coordination_preview_list(
                delegation_contract.get("preferred_collaborators")
            )
            weak_handoff_targets = cls._normalize_coordination_preview_list(
                delegation_contract.get("weak_handoff_targets")
            )
            watchouts = cls._normalize_string_list(delegation_contract.get("watchouts"))
            role_profile_suggestion = (
                dict(agent_capability.get("role_profile_suggestion") or {})
                if isinstance(agent_capability.get("role_profile_suggestion"), dict)
                else {}
            )
            preferred_delegate_candidates = [
                candidate
                for candidate in recommended_collaborators
                if cls._normalize_string(candidate.get("fit")) in {"strong", "good"}
                and not bool(candidate.get("same_role_profile_overlap_risk"))
            ]
            if not preferred_delegate_candidates:
                preferred_delegate_candidates = [
                    candidate
                    for candidate in recommended_collaborators
                    if not bool(candidate.get("same_role_profile_overlap_risk"))
                ]

            execution_delegate_candidates: list[dict[str, Any]] = []
            mcp_delegate_candidates: list[dict[str, Any]] = []
            if primary_role_mode == "coordinator" or should_coordinate_parallel_work:
                for candidate in preferred_delegate_candidates:
                    candidate_profile_id = cls._normalize_string(candidate.get("target_profile_id"))
                    complementary_lane_ids = cls._normalize_string_list(candidate.get("complementary_lane_ids"))
                    new_skill_ids = cls._normalize_string_list(candidate.get("new_skill_ids"))
                    new_tool_ids = cls._normalize_string_list(candidate.get("new_tool_ids"))
                    new_mcp_server_ids = cls._normalize_string_list(candidate.get("new_mcp_server_ids"))
                    gap_cover_mcp_server_ids = cls._normalize_string_list(
                        candidate.get("gap_cover_mcp_server_ids")
                    )
                    if executable_tool_ids and (
                        new_tool_ids
                        or new_skill_ids
                        or complementary_lane_ids
                        or candidate_profile_id in {"implementation", "research", "verification"}
                    ):
                        execution_delegate_candidates.append(candidate)
                    if planning_only_mcp_server_ids and (
                        new_mcp_server_ids
                        or gap_cover_mcp_server_ids
                        or complementary_lane_ids
                        or candidate_profile_id in {"research", "implementation", "verification"}
                    ):
                        mcp_delegate_candidates.append(candidate)
                if executable_tool_ids and not execution_delegate_candidates:
                    execution_delegate_candidates = preferred_delegate_candidates[:2]
                if planning_only_mcp_server_ids and not mcp_delegate_candidates:
                    mcp_delegate_candidates = preferred_delegate_candidates[:2]
                execution_delegate_candidates = execution_delegate_candidates[:2]
                mcp_delegate_candidates = mcp_delegate_candidates[:2]

            preview = {
                "agent_id": agent_id,
                "agent_name": agent_name,
            }

            if primary_role_mode == "coordinator":
                append_preview_once(coordinator_agents, preview)
            if should_coordinate_parallel_work:
                append_preview_once(parallel_coordinator_agents, preview)
            if should_produce_final_output:
                append_preview_once(final_output_agents, preview)
            if primary_role_mode == "verification" or "verification" in supporting_role_modes or work_strategy == "verify_and_close":
                append_preview_once(verification_agents, preview)
            if primary_role_mode == "coordinator" or should_coordinate_parallel_work:
                if executable_tool_ids:
                    coordinator_execution_tool_risks.append(
                        {
                            "source_agent_id": agent_id,
                            "source_agent_name": agent_name,
                            "delegate_tool_ids": executable_tool_ids,
                            "delegate_candidates": execution_delegate_candidates,
                            "agent_previews": build_delegate_candidate_previews(
                                preview,
                                execution_delegate_candidates,
                            ),
                        }
                    )
                if planning_only_mcp_server_ids:
                    coordinator_mcp_context_risks.append(
                        {
                            "source_agent_id": agent_id,
                            "source_agent_name": agent_name,
                            "delegate_mcp_server_ids": planning_only_mcp_server_ids,
                            "delegate_candidates": mcp_delegate_candidates,
                            "agent_previews": build_delegate_candidate_previews(
                                preview,
                                mcp_delegate_candidates,
                            ),
                        }
                    )

            role_profile_agents.append(
                {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "loaded_skill_ids": cls._normalize_string_list(agent_capability.get("loaded_skill_ids")),
                    "enabled_tool_ids": cls._normalize_string_list(agent_capability.get("enabled_tool_ids")),
                    "provider_limited_tool_ids": cls._normalize_string_list(
                        agent_capability.get("provider_limited_tool_ids")
                    ),
                    "mcp_server_ids": cls._normalize_string_list(agent_capability.get("mcp_server_ids")),
                    "configured_denied_tool_ids": cls._normalize_string_list(
                        agent_capability.get("configured_denied_tool_ids")
                    ),
                    "configured_denied_mcp_server_ids": cls._normalize_string_list(
                        agent_capability.get("configured_denied_mcp_server_ids")
                    ),
                    "delegation_lane_ids": cls._normalize_string_list(
                        agent_capability.get("delegation_lane_ids")
                    ),
                    "role_profile_suggestion": role_profile_suggestion,
                }
            )

            if (
                primary_role_mode != "generalist"
                or supporting_role_modes
                or should_coordinate_parallel_work
                or should_produce_final_output
                or tool_access_mode
                or mcp_access_mode
                or executable_tool_ids
                or planning_only_tool_ids
                or planning_only_mcp_server_ids
                or primary_focus
                or upstream_agents
                or downstream_agents
                or preferred_collaborators
                or weak_handoff_targets
                or watchouts
            ):
                payload: dict[str, Any] = {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "primary_role_mode": primary_role_mode,
                    "supporting_role_modes": supporting_role_modes,
                    "work_strategy": work_strategy,
                    "should_coordinate_parallel_work": should_coordinate_parallel_work,
                    "should_produce_final_output": should_produce_final_output,
                    "recovery_actions": cls._build_recovery_actions(),
                }
                if tool_access_mode:
                    payload["tool_access_mode"] = tool_access_mode
                if mcp_access_mode:
                    payload["mcp_access_mode"] = mcp_access_mode
                if executable_tool_ids:
                    payload["executable_tool_ids"] = executable_tool_ids
                if planning_only_tool_ids:
                    payload["planning_only_tool_ids"] = planning_only_tool_ids
                if planning_only_mcp_server_ids:
                    payload["planning_only_mcp_server_ids"] = planning_only_mcp_server_ids
                if primary_focus:
                    payload["primary_focus"] = primary_focus
                if upstream_agents:
                    payload["upstream_agents"] = upstream_agents
                if downstream_agents:
                    payload["downstream_agents"] = downstream_agents
                if preferred_collaborators:
                    payload["preferred_collaborators"] = preferred_collaborators
                if weak_handoff_targets:
                    payload["weak_handoff_targets"] = weak_handoff_targets
                if watchouts:
                    payload["watchouts"] = watchouts
                if execution_delegate_candidates:
                    payload["delegate_execution_candidates"] = execution_delegate_candidates
                if mcp_delegate_candidates:
                    payload["delegate_mcp_candidates"] = mcp_delegate_candidates
                profile_id = cls._normalize_string(role_profile_suggestion.get("profile_id"))
                if profile_id:
                    payload["role_profile_id"] = profile_id
                agents.append(payload)

        risk_entries: list[dict[str, Any]] = []

        def append_risk(
            *,
            risk_id: str,
            severity: str,
            summary: str,
            recommended_action: str,
            agent_previews: list[dict[str, Any]],
            payload: dict[str, Any] | None = None,
        ) -> None:
            normalized_previews = cls._normalize_coordination_preview_list(agent_previews)
            entry: dict[str, Any] = {
                "risk_id": risk_id,
                "severity": severity if severity in {"high", "medium", "low"} else "low",
                "summary": summary,
                "recommended_action": recommended_action,
                "agent_previews": normalized_previews,
            }
            if payload:
                entry.update(payload)
            risk_entries.append(entry)

        candidate_coordinator_agents = [
            {
                "agent_id": cls._normalize_string(agent.get("agent_id")),
                "agent_name": cls._normalize_string(agent.get("agent_name") or agent.get("agent_id")),
            }
            for agent in agents
            if bool(agent.get("should_coordinate_parallel_work"))
            or cls._normalize_delegation_role_mode(agent.get("primary_role_mode")) == "coordinator"
            or len(cls._normalize_coordination_preview_list(agent.get("downstream_agents"))) > 0
        ]
        candidate_final_output_agents = [
            {
                "agent_id": cls._normalize_string(agent.get("agent_id")),
                "agent_name": cls._normalize_string(agent.get("agent_name") or agent.get("agent_id")),
            }
            for agent in agents
            if bool(agent.get("should_produce_final_output"))
            or cls._normalize_delegation_work_strategy(agent.get("work_strategy"))
            in {"self_contained_delivery", "verify_and_close"}
            or len(cls._normalize_coordination_preview_list(agent.get("downstream_agents"))) == 0
        ]
        candidate_verification_agents = [
            {
                "agent_id": cls._normalize_string(agent.get("agent_id")),
                "agent_name": cls._normalize_string(agent.get("agent_name") or agent.get("agent_id")),
            }
            for agent in agents
            if cls._normalize_delegation_role_mode(agent.get("primary_role_mode")) == "verification"
            or "verification" in cls._normalize_string_list(agent.get("supporting_role_modes"))
            or cls._normalize_delegation_work_strategy(agent.get("work_strategy")) == "verify_and_close"
            or bool(agent.get("should_produce_final_output"))
        ]

        should_check_structural_coverage = considered_agent_count > 1 or bool(agents)

        if considered_agent_count > 1 and not coordinator_agents:
            append_risk(
                risk_id="missing_coordinator",
                severity="high",
                summary="No node currently owns coordinator mode for this run scope.",
                recommended_action="Promote a planner/coordinator node before the next orchestration pass.",
                agent_previews=candidate_coordinator_agents[:3],
            )
        if should_check_structural_coverage and not final_output_agents:
            append_risk(
                risk_id="missing_final_output_owner",
                severity="high",
                summary="No node currently claims final-output responsibility for this run scope.",
                recommended_action="Assign one node to close the loop with a self-contained final result.",
                agent_previews=candidate_final_output_agents[:3],
            )
        if should_check_structural_coverage and not review_agent_enabled and not verification_agents:
            append_risk(
                risk_id="missing_verification_owner",
                severity="medium",
                summary="No node currently owns verification or close-out review while the review agent is disabled.",
                recommended_action="Add a verifier node or enable a review path before the next run.",
                agent_previews=candidate_verification_agents[:3],
            )
        if len(final_output_agents) > 1:
            append_risk(
                risk_id="multiple_final_output_owners",
                severity="low",
                summary="Multiple nodes claim final-output responsibility in the same run scope.",
                recommended_action="Clarify which node should close the loop to avoid duplicate or conflicting final answers.",
                agent_previews=final_output_agents[:3],
            )
        for agent in agents:
            if not bool(agent.get("should_coordinate_parallel_work")):
                continue
            downstream_agents = cls._normalize_coordination_preview_list(agent.get("downstream_agents"))
            if downstream_agents:
                continue
            append_risk(
                risk_id="coordinator_without_downstream",
                severity="medium",
                summary=(
                    f"{cls._normalize_string(agent.get('agent_name') or agent.get('agent_id'))} is marked to coordinate parallel work but has no downstream handoff targets."
                ),
                recommended_action="Add at least one downstream lane or turn this node into a self-contained finisher instead of a coordinator.",
                agent_previews=[
                    {
                        "agent_id": cls._normalize_string(agent.get("agent_id")),
                        "agent_name": cls._normalize_string(agent.get("agent_name") or agent.get("agent_id")),
                    }
                ],
            )
        for item in coordinator_execution_tool_risks:
            delegate_tool_ids = cls._normalize_string_list(item.get("delegate_tool_ids"))
            delegate_candidates = [
                candidate
                for candidate in item.get("delegate_candidates") or []
                if isinstance(candidate, dict)
            ]
            source_agent_name = cls._normalize_string(
                item.get("source_agent_name") or item.get("source_agent_id")
            ) or "Coordinator"
            append_risk(
                risk_id="coordinator_with_execution_tools",
                severity="low",
                summary=(
                    f"{source_agent_name} still holds direct execution tools: "
                    + cls._format_identifier_preview(delegate_tool_ids, limit=3)
                    + "."
                ),
                recommended_action=build_delegate_action(
                    capability_label="direct execution",
                    candidates=delegate_candidates,
                    fallback=(
                        "If this node should stay in planner mode, narrow its tool policy and move direct execution to worker nodes."
                    ),
                ),
                agent_previews=[
                    dict(preview)
                    for preview in item.get("agent_previews") or []
                    if isinstance(preview, dict)
                ],
                payload={
                    "source_agent_id": cls._normalize_string(item.get("source_agent_id")),
                    "source_agent_name": source_agent_name,
                    "delegate_tool_ids": delegate_tool_ids,
                    "delegate_candidates": delegate_candidates,
                },
            )
        for item in coordinator_mcp_context_risks:
            delegate_mcp_server_ids = cls._normalize_string_list(item.get("delegate_mcp_server_ids"))
            delegate_candidates = [
                candidate
                for candidate in item.get("delegate_candidates") or []
                if isinstance(candidate, dict)
            ]
            source_agent_name = cls._normalize_string(
                item.get("source_agent_name") or item.get("source_agent_id")
            ) or "Coordinator"
            append_risk(
                risk_id="coordinator_with_mcp_context",
                severity="low",
                summary=(
                    f"{source_agent_name} still depends on planning-only MCP context: "
                    + cls._format_identifier_preview(delegate_mcp_server_ids, limit=3)
                    + "."
                ),
                recommended_action=build_delegate_action(
                    capability_label="MCP-backed work",
                    candidates=delegate_candidates,
                    fallback=(
                        "Keep coordinator MCP access scoped to orchestration-safe servers, or hand external-system work to specialist nodes."
                    ),
                ),
                agent_previews=[
                    dict(preview)
                    for preview in item.get("agent_previews") or []
                    if isinstance(preview, dict)
                ],
                payload={
                    "source_agent_id": cls._normalize_string(item.get("source_agent_id")),
                    "source_agent_name": source_agent_name,
                    "delegate_mcp_server_ids": delegate_mcp_server_ids,
                    "delegate_candidates": delegate_candidates,
                },
            )

        role_profile_alignment = build_role_profile_alignment_diagnostics(role_profile_agents)
        role_profile_drift_agents = [
            dict(item)
            for item in role_profile_alignment.get("drift_agents") or []
            if isinstance(item, dict)
        ]
        role_profile_overlap_risks = [
            dict(item)
            for item in role_profile_alignment.get("overlap_risks") or []
            if isinstance(item, dict)
        ]
        role_profile_drift_by_agent_id = {
            cls._normalize_string(item.get("agent_id")): item
            for item in role_profile_drift_agents
            if cls._normalize_string(item.get("agent_id"))
        }
        for agent in agents:
            drift = role_profile_drift_by_agent_id.get(cls._normalize_string(agent.get("agent_id")))
            if not isinstance(drift, dict):
                continue
            if cls._normalize_string(drift.get("profile_id")):
                agent["role_profile_id"] = cls._normalize_string(drift.get("profile_id"))
            if drift.get("missing_skill_ids"):
                agent["missing_role_profile_skill_ids"] = cls._normalize_string_list(
                    drift.get("missing_skill_ids")
                )
            if drift.get("missing_tool_ids"):
                agent["missing_role_profile_tool_ids"] = cls._normalize_string_list(
                    drift.get("missing_tool_ids")
                )
            if drift.get("missing_mcp_server_ids"):
                agent["missing_role_profile_mcp_server_ids"] = cls._normalize_string_list(
                    drift.get("missing_mcp_server_ids")
                )
            if drift.get("outstanding_restrictive_tool_ids"):
                agent["outstanding_role_profile_tool_restriction_ids"] = cls._normalize_string_list(
                    drift.get("outstanding_restrictive_tool_ids")
                )
            if drift.get("outstanding_restrictive_mcp_server_ids"):
                agent["outstanding_role_profile_mcp_server_restriction_ids"] = cls._normalize_string_list(
                    drift.get("outstanding_restrictive_mcp_server_ids")
                )

        role_profile_drift_agent_count = int(role_profile_alignment.get("drift_agent_count") or 0)
        role_profile_overlap_risk_count = int(role_profile_alignment.get("overlap_risk_count") or 0)
        if role_profile_drift_agent_count > 0:
            drift_examples: list[tuple[str, list[str]]] = []
            for item in role_profile_drift_agents[:3]:
                issue_ids = cls._normalize_string_list(item.get("missing_skill_ids"))
                issue_ids += cls._normalize_string_list(item.get("missing_tool_ids"))
                issue_ids += cls._normalize_string_list(item.get("missing_mcp_server_ids"))
                issue_ids += cls._normalize_string_list(item.get("outstanding_restrictive_tool_ids"))
                issue_ids += cls._normalize_string_list(
                    item.get("outstanding_restrictive_mcp_server_ids")
                )
                drift_examples.append(
                    (
                        cls._normalize_string(item.get("agent_name") or item.get("agent_id")) or "Unknown node",
                        issue_ids,
                    )
                )
            append_risk(
                risk_id="role_profile_drift",
                severity="medium",
                summary=(
                    f"Role profile alignment is still drifting for {role_profile_drift_agent_count} node(s): "
                    + cls._format_named_capability_examples(drift_examples)
                    + "."
                ),
                recommended_action="Apply the suggested role profile changes and request any missing profile skills before the next orchestration pass.",
                agent_previews=[
                    {
                        "agent_id": cls._normalize_string(item.get("agent_id")),
                        "agent_name": cls._normalize_string(item.get("agent_name") or item.get("agent_id")),
                    }
                    for item in role_profile_drift_agents[:3]
                ],
            )
        if role_profile_overlap_risk_count > 0:
            overlap_examples: list[str] = []
            overlap_agent_previews: list[dict[str, Any]] = []
            for item in role_profile_overlap_risks[:3]:
                profile_id = cls._normalize_string(item.get("profile_id")) or "generalist"
                shared_lane_ids = cls._normalize_string_list(item.get("shared_lane_ids"))
                agent_previews = cls._normalize_coordination_preview_list(item.get("agent_previews"))
                overlap_examples.append(
                    f"{profile_id}: "
                    + ", ".join(
                        cls._normalize_string(preview.get("agent_name") or preview.get("agent_id"))
                        for preview in agent_previews
                    )
                    + (f" ({cls._format_identifier_preview(shared_lane_ids)})" if shared_lane_ids else "")
                )
                for preview in agent_previews:
                    append_preview_once(overlap_agent_previews, preview)
            append_risk(
                risk_id="role_profile_overlap",
                severity="medium"
                if any(
                    cls._normalize_string(item.get("profile_id")) == "coordinator"
                    for item in role_profile_overlap_risks
                )
                else "low",
                summary=(
                    f"{role_profile_overlap_risk_count} near-duplicate role-profile overlap pair(s) are still active: "
                    + "; ".join(overlap_examples)
                    + "."
                ),
                recommended_action="Differentiate same-profile nodes with distinct lanes, tools, or MCP scope so they complement each other instead of duplicating work.",
                agent_previews=overlap_agent_previews[:4],
            )

        if not agents and not risk_entries:
            return {}
        return {
            "agent_count": considered_agent_count,
            "coordinator_agent_count": len(coordinator_agents),
            "parallel_coordinator_agent_count": len(parallel_coordinator_agents),
            "final_output_agent_count": len(final_output_agents),
            "verification_agent_count": len(verification_agents),
            "role_profile_drift_agent_count": role_profile_drift_agent_count,
            "role_profile_overlap_risk_count": role_profile_overlap_risk_count,
            "risk_count": len(risk_entries),
            "coordinator_agents": coordinator_agents,
            "parallel_coordinator_agents": parallel_coordinator_agents,
            "final_output_agents": final_output_agents,
            "verification_agents": verification_agents,
            "role_profile_drift_agents": role_profile_drift_agents,
            "role_profile_overlap_risks": role_profile_overlap_risks,
            "risks": risk_entries,
            "agents": agents,
        }

    def build_document_ingest_result(
        self,
        *,
        ok: bool,
        stage: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["document_ingest_result"],
            "artifacts": {
                "stage": stage,
                "error_code": error_code,
            },
            "summary": "document ingest succeeded" if ok else (error_message or "document ingest failed"),
        }

    def build_approval_checkpoint_result(
        self,
        *,
        ok: bool,
        session_id: str | None,
        approved: bool,
        interrupted: bool | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["approval_checkpoint_ready"],
            "artifacts": {
                "session_id": session_id,
                "approved": approved,
                "interrupted": interrupted,
                "error_code": error_code,
            },
            "summary": "approval gate ready" if ok else (error_message or "approval gate not ready"),
        }

    def build_session_resume_result(
        self,
        *,
        ok: bool,
        session_id: str | None,
        interrupted: bool | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["session_resume_execution"],
            "artifacts": {
                "session_id": session_id,
                "interrupted": interrupted,
                "error_code": error_code,
            },
            "summary": "session resume succeeded" if ok else (error_message or "session resume failed"),
        }

    def build_agent_orchestration_result(
        self,
        *,
        ok: bool,
        active_agent_ids: list[str],
        blocked_agents: list[dict[str, object]],
        loop_count: int,
        review_agent_enabled: bool,
        error_code: str | None,
        error_message: str | None,
        agent_outputs: dict[str, str] | None = None,
        output_artifacts: dict[str, dict[str, object]] | None = None,
        recovery_mode: str | None = None,
        review_details: dict[str, object] | None = None,
        capability_snapshot: dict[str, object] | None = None,
        handoff_scope: str | None = None,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, object]:
        handoff_diagnostics = self._build_handoff_diagnostics(
            active_agent_ids=active_agent_ids,
            capability_snapshot=capability_snapshot,
        )
        readiness_diagnostics = self._build_readiness_diagnostics(
            active_agent_ids=active_agent_ids,
            capability_snapshot=capability_snapshot,
        )
        availability_diagnostics = self._build_availability_diagnostics(
            active_agent_ids=active_agent_ids,
            capability_snapshot=capability_snapshot,
        )
        execution_contract_diagnostics = self._build_execution_contract_diagnostics(
            active_agent_ids=active_agent_ids,
            capability_snapshot=capability_snapshot,
        )
        collaboration_contract_diagnostics = self._build_collaboration_contract_diagnostics(
            active_agent_ids=active_agent_ids,
            capability_snapshot=capability_snapshot,
            review_agent_enabled=review_agent_enabled,
        )
        checks_run = ["agent_orchestration_cycle"]
        if handoff_diagnostics:
            checks_run.append("handoff_fit_diagnostics")
        if readiness_diagnostics:
            checks_run.append("capability_readiness")
        if availability_diagnostics:
            checks_run.append("capability_availability")
        if execution_contract_diagnostics:
            checks_run.append("capability_execution_contracts")
        if collaboration_contract_diagnostics:
            checks_run.append("collaboration_contracts")
        weak_edge_count = int(handoff_diagnostics.get("weak_edge_count") or 0) if handoff_diagnostics else 0
        blocked_count = int(readiness_diagnostics.get("blocked_count") or 0) if readiness_diagnostics else 0
        limited_count = int(readiness_diagnostics.get("limited_count") or 0) if readiness_diagnostics else 0
        unavailable_count = int(availability_diagnostics.get("unavailable_count") or 0) if availability_diagnostics else 0
        limited_availability_count = int(availability_diagnostics.get("limited_count") or 0) if availability_diagnostics else 0
        planning_only_tool_agent_count = (
            int(execution_contract_diagnostics.get("planning_only_tool_agent_count") or 0)
            if execution_contract_diagnostics
            else 0
        )
        collaboration_risk_count = (
            int(collaboration_contract_diagnostics.get("risk_count") or 0)
            if collaboration_contract_diagnostics
            else 0
        )
        summary = "agent orchestration completed" if ok else (error_message or "agent orchestration failed")
        if ok and weak_edge_count:
            summary = f"{summary}; {weak_edge_count} weak downstream handoff(s) flagged"
        if ok and unavailable_count:
            summary = f"{summary}; {unavailable_count} unavailable capability definition(s) flagged"
        elif ok and limited_availability_count:
            summary = f"{summary}; {limited_availability_count} availability-limited definition(s) flagged"
        if ok and blocked_count:
            summary = f"{summary}; {blocked_count} blocked capability profile(s) flagged"
        elif ok and limited_count:
            summary = f"{summary}; {limited_count} limited capability profile(s) flagged"
        if ok and planning_only_tool_agent_count:
            summary = f"{summary}; {planning_only_tool_agent_count} planning-only tool contract(s) flagged"
        if ok and collaboration_risk_count:
            summary = f"{summary}; {collaboration_risk_count} collaboration contract risk(s) flagged"

        artifacts = {
            "active_agent_ids": active_agent_ids,
            "blocked_agents": blocked_agents,
            "loop_count": loop_count,
            "review_agent_enabled": review_agent_enabled,
            "error_code": error_code,
            "agent_outputs": agent_outputs or {},
            "output_artifacts": output_artifacts or {},
            "recovery_mode": recovery_mode,
            "review_details": review_details or {},
            "capability_snapshot": capability_snapshot or {},
        }
        resolved_scope = self._normalize_string(
            handoff_scope
            or ((capability_snapshot or {}) if isinstance(capability_snapshot, dict) else {}).get("handoff_diagnostic_scope")
            or "all_agents"
        ) or "all_agents"
        scope_artifact: dict[str, Any] = {
            "scope": resolved_scope,
            "active_agent_ids": list(active_agent_ids or []),
        }
        normalized_selected_agent_ids = self._normalize_string_list(selected_agent_ids)
        if normalized_selected_agent_ids:
            scope_artifact["selected_agent_ids"] = normalized_selected_agent_ids
        artifacts["handoff_diagnostics_scope"] = scope_artifact
        if handoff_diagnostics:
            artifacts["handoff_diagnostics"] = handoff_diagnostics
        if availability_diagnostics:
            artifacts["capability_availability"] = availability_diagnostics
        if readiness_diagnostics:
            artifacts["capability_readiness"] = readiness_diagnostics
        if execution_contract_diagnostics:
            artifacts["capability_execution_contracts"] = execution_contract_diagnostics
        if collaboration_contract_diagnostics:
            artifacts["collaboration_contracts"] = collaboration_contract_diagnostics

        return {
            "status": "pass" if ok else "fail",
            "checks_run": checks_run,
            "artifacts": artifacts,
            "summary": summary,
        }
