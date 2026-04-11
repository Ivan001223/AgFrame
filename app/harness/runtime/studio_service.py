from __future__ import annotations

import inspect
import time
import uuid
from pathlib import Path

from app.infrastructure.config.settings import settings
from app.harness.contracts.run import HarnessTaskType
from app.harness.contracts.studio import (
    HarnessAgentCapabilitySummary,
    HarnessAgentDelegationContract,
    HarnessAgentExecutionContract,
    HarnessAgentRoleProfileSuggestion,
    HarnessCapabilityOwnerEntry,
    HarnessCanvasAgent,
    HarnessCanvasEdge,
    HarnessCanvasPosition,
    HarnessCoordinationAgentPreview,
    HarnessDelegationOpportunity,
    HarnessExecutionChecklistItem,
    HarnessMcpServerCatalogItem,
    HarnessOrchestrationAgentRoutingSummary,
    HarnessOrchestrationBriefCapabilityRisk,
    HarnessOrchestrationPhaseSummary,
    HarnessOrchestrationRepairPriority,
    HarnessOrchestrationSummary,
    HarnessReviewAgentSettings,
    HarnessSkillCatalogItem,
    HarnessSkillPoolItem,
    HarnessSkillRequest,
    HarnessStudioGraph,
    HarnessStudioGraphDiagnostics,
    HarnessToolCatalogItem,
)
from app.harness.persistence.stores import HarnessAgentProjectStore
from app.harness.runtime.role_profile_alignment import build_role_profile_alignment_diagnostics
from app.harness.runtime.run_service import HarnessRunService, build_run_service
from app.runtime.llm.provider_registry import infer_tool_calling_support
from app.skills.common.tools import ALL_TOOLS
from app.skills.registry import build_fallback_skill_descriptor, get_skill_descriptor


class HarnessStudioProjectNotFoundError(ValueError):
    pass


class HarnessStudioProjectAccessError(ValueError):
    pass


class HarnessStudioAgentNotFoundError(ValueError):
    pass


_RESEARCH_PHASE_KEYWORDS = (
    "research",
    "researcher",
    "rag",
    "retrieve",
    "retrieval",
    "search",
    "evidence",
    "context",
    "docs",
    "document",
    "study",
    "investig",
    "研究",
    "检索",
    "证据",
    "文档",
    "背景",
)

_SYNTHESIS_PHASE_KEYWORDS = (
    "plan",
    "planner",
    "coordinator",
    "coordination",
    "orchestrat",
    "delegate",
    "delegation",
    "synthes",
    "synthesis",
    "chair",
    "strateg",
    "cluster",
    "编排",
    "协调",
    "规划",
    "综合",
    "主持",
    "策略",
)

_IMPLEMENTATION_PHASE_KEYWORDS = (
    "build",
    "builder",
    "implement",
    "implementation",
    "execute",
    "execution",
    "engineer",
    "code",
    "develop",
    "delivery",
    "执行",
    "实现",
    "开发",
    "编码",
)

_VERIFICATION_PHASE_KEYWORDS = (
    "review",
    "reviewer",
    "verify",
    "verification",
    "qa",
    "test",
    "critic",
    "audit",
    "compliance",
    "审查",
    "验证",
    "测试",
    "质疑",
    "合规",
)


class HarnessStudioService:
    _TOOL_FLAG_BY_ID: dict[str, str] = {
        "calculator": "enable_tools_python_repl",
        "write_file": "enable_tools_write_file",
        "python_executor": "enable_tools_python_executor",
    }
    _COORDINATOR_ALLOWED_DIRECT_TOOL_IDS: set[str] = {"get_current_time"}

    def __init__(
        self,
        *,
        project_store: HarnessAgentProjectStore,
        run_service: HarnessRunService | None = None,
    ):
        self.project_store = project_store
        self.run_service = run_service or build_run_service()
        self.skills_root = Path(__file__).resolve().parents[2] / "skills"

    @staticmethod
    def _normalize_skill_key(value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")

    @staticmethod
    def _humanize_identifier(value: str) -> str:
        parts = [part for part in value.replace("-", "_").split("_") if part]
        return " ".join(part.capitalize() for part in parts) or value

    @staticmethod
    def _format_preview_list(values: list[str], *, limit: int = 5) -> str:
        trimmed = [str(value).strip() for value in values if str(value).strip()]
        if not trimmed:
            return "none"
        preview = trimmed[:limit]
        suffix = f" (+{len(trimmed) - limit} more)" if len(trimmed) > limit else ""
        return ", ".join(preview) + suffix

    def _normalize_identifier_list(self, values: list[object] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            normalized_value = self._normalize_skill_key(str(value))
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            normalized.append(normalized_value)
        return normalized

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _expand_mcp_alias_candidates(self, value: str) -> list[str]:
        raw_value = str(value or "").strip()
        if not raw_value:
            return []

        raw_variants = [raw_value]
        lowered = raw_value.lower()
        claude_prefix = "claude.ai "
        if lowered.startswith(claude_prefix):
            trimmed = raw_value[len(claude_prefix) :].strip()
            if trimmed:
                raw_variants.append(trimmed)

        prefixes = (
            "server_",
            "mcp_server_",
            "modelcontextprotocol_server_",
            "claude_ai_",
            "claude_ai_server_",
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for raw_variant in raw_variants:
            normalized_variants = [self._normalize_skill_key(raw_variant)]
            basename = raw_variant.rsplit("/", 1)[-1].strip()
            if basename and basename != raw_variant:
                normalized_variants.append(self._normalize_skill_key(basename))
            for normalized in normalized_variants:
                if not normalized:
                    continue
                alias_values = [normalized]
                for prefix in prefixes:
                    if normalized.startswith(prefix):
                        alias_values.append(normalized[len(prefix) :])
                if normalized.endswith("_server"):
                    alias_values.append(normalized[: -len("_server")])
                for alias in alias_values:
                    alias = alias.strip("_")
                    if not alias or alias in seen:
                        continue
                    seen.add(alias)
                    candidates.append(alias)
        return candidates

    def _derive_mcp_server_alias_ids(self, payload: dict[str, object], *, server_id: str) -> list[str]:
        alias_values = [server_id]
        alias_values.extend(
            self._expand_mcp_alias_candidates(str(payload.get("title") or ""))
        )
        alias_values.extend(
            self._expand_mcp_alias_candidates(str(payload.get("command") or ""))
        )
        for arg in payload.get("args") or []:
            alias_values.extend(self._expand_mcp_alias_candidates(str(arg)))
        return self._dedupe_preserve_order(alias_values)

    def _build_mcp_server_alias_lookup(self) -> dict[str, str]:
        alias_candidates: dict[str, set[str]] = {}
        for item in self._configured_mcp_servers():
            server_id = self._normalize_skill_key(str(item.get("server_id") or ""))
            if not server_id:
                continue
            for alias in item.get("alias_ids") or []:
                normalized_alias = self._normalize_skill_key(str(alias))
                if not normalized_alias:
                    continue
                alias_candidates.setdefault(normalized_alias, set()).add(server_id)

        return {
            alias: next(iter(server_ids))
            for alias, server_ids in alias_candidates.items()
            if len(server_ids) == 1
        }

    def _canonicalize_mcp_server_ids(
        self,
        values: list[str],
        *,
        alias_lookup: dict[str, str],
    ) -> list[str]:
        canonical_ids: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized_value = self._normalize_skill_key(str(value))
            if not normalized_value:
                continue
            canonical_id = alias_lookup.get(normalized_value, normalized_value)
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            canonical_ids.append(canonical_id)
        return canonical_ids

    def _discover_skill_catalog(self) -> list[dict[str, object]]:
        if not self.skills_root.exists():
            return []
        catalog: list[dict[str, object]] = []
        for child in sorted(self.skills_root.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            skill_id = self._normalize_skill_key(child.name)
            descriptor = get_skill_descriptor(skill_id) or build_fallback_skill_descriptor(skill_id)
            catalog.append(
                HarnessSkillCatalogItem(
                    skill_id=descriptor.skill_id,
                    title=descriptor.title,
                    description=descriptor.description,
                    source=f"app/skills/{child.name}",
                    prompt_hint=descriptor.prompt_hint or None,
                    suggested_tool_ids=list(descriptor.suggested_tool_ids),
                    suggested_mcp_server_ids=list(descriptor.suggested_mcp_server_ids),
                ).model_dump()
            )
        return catalog

    def _discover_tool_catalog(self) -> list[dict[str, object]]:
        catalog: list[dict[str, object]] = []
        feature_flags = settings.feature_flags
        for tool in ALL_TOOLS:
            tool_id = self._normalize_skill_key(str(getattr(tool, "name", "") or ""))
            if not tool_id:
                continue
            requires_flag = self._TOOL_FLAG_BY_ID.get(tool_id)
            enabled = bool(getattr(feature_flags, requires_flag)) if requires_flag else True
            description = str(getattr(tool, "description", "") or inspect.getdoc(tool) or "").strip()
            first_line = description.splitlines()[0].strip() if description else None
            catalog.append(
                HarnessToolCatalogItem(
                    tool_id=tool_id,
                    title=self._humanize_identifier(tool_id),
                    description=first_line or None,
                    status="enabled" if enabled else "disabled",
                    requires_flag=requires_flag,
                ).model_dump()
            )
        return sorted(catalog, key=lambda item: str(item.get("title") or item.get("tool_id") or ""))

    def _configured_mcp_servers(self) -> list[dict[str, object]]:
        raw_config = getattr(settings, "mcp", None)
        raw_servers = getattr(raw_config, "servers", []) if raw_config is not None else []
        catalog: list[dict[str, object]] = []
        for raw_server in raw_servers or []:
            if isinstance(raw_server, dict):
                payload = dict(raw_server)
            elif hasattr(raw_server, "model_dump"):
                payload = dict(raw_server.model_dump())
            else:
                payload = {
                    "server_id": getattr(raw_server, "server_id", ""),
                    "title": getattr(raw_server, "title", ""),
                    "description": getattr(raw_server, "description", ""),
                    "command": getattr(raw_server, "command", ""),
                    "args": list(getattr(raw_server, "args", []) or []),
                    "enabled": getattr(raw_server, "enabled", True),
                }
            server_id = self._normalize_skill_key(str(payload.get("server_id") or ""))
            if not server_id:
                continue
            catalog.append(
                {
                    "server_id": server_id,
                    "title": str(payload.get("title") or self._humanize_identifier(server_id)).strip()
                    or self._humanize_identifier(server_id),
                    "description": str(payload.get("description") or "").strip() or None,
                    "command": str(payload.get("command") or "").strip() or None,
                    "args": [str(arg).strip() for arg in payload.get("args") or [] if str(arg).strip()],
                    "enabled": bool(payload.get("enabled", True)),
                    "alias_ids": self._derive_mcp_server_alias_ids(payload, server_id=server_id),
                }
            )
        return sorted(catalog, key=lambda item: str(item.get("title") or item.get("server_id") or ""))

    def _build_mcp_command_preview(self, command: str | None, args: list[str]) -> str | None:
        parts = [str(command or "").strip(), *[str(arg).strip() for arg in args if str(arg).strip()]]
        compact = " ".join(part for part in parts if part).strip()
        if not compact:
            return None
        if len(compact) <= 96:
            return compact
        return compact[:93].rstrip() + "..."

    def _discover_mcp_server_catalog(self) -> list[dict[str, object]]:
        catalog: list[dict[str, object]] = []
        for item in self._configured_mcp_servers():
            catalog.append(
                HarnessMcpServerCatalogItem(
                    server_id=str(item.get("server_id") or ""),
                    title=str(item.get("title") or item.get("server_id") or ""),
                    description=str(item.get("description") or "") or None,
                    status="enabled" if bool(item.get("enabled", True)) else "disabled",
                    command_preview=self._build_mcp_command_preview(
                        str(item.get("command") or "") or None,
                        [str(arg) for arg in item.get("args") or [] if str(arg).strip()],
                    ),
                ).model_dump()
            )
        return catalog

    def _sync_skill_pool_with_catalog(self, graph: dict[str, object]) -> dict[str, object]:
        catalog = graph.get("skill_catalog") or []
        existing_pool = list(graph.get("skill_pool") or [])
        catalog_by_key: dict[str, dict[str, object]] = {}
        for item in catalog:
            if not isinstance(item, dict):
                continue
            skill_key = self._normalize_skill_key(str(item.get("skill_id") or item.get("title") or ""))
            if not skill_key:
                continue
            catalog_by_key[skill_key] = dict(item)

        pool_by_key: dict[str, dict[str, object]] = {}

        for item in existing_pool:
            if not isinstance(item, dict):
                continue
            skill_key = self._normalize_skill_key(str(item.get("skill_id") or ""))
            if not skill_key:
                continue
            catalog_item = catalog_by_key.get(skill_key, {})
            pool_by_key[skill_key] = HarnessSkillPoolItem(
                skill_id=str(catalog_item.get("skill_id") or item.get("skill_id") or skill_key),
                title=str(catalog_item.get("title") or item.get("title") or skill_key),
                description=str(catalog_item.get("description") or item.get("description") or "") or None,
                source=str(catalog_item.get("source") or item.get("source") or "app/skills"),
                status=str(item.get("status") or "loaded"),
                approved_at=item.get("approved_at") if item.get("approved_at") is not None else None,
            ).model_dump()

        graph["skill_pool"] = sorted(pool_by_key.values(), key=lambda item: str(item.get("title") or item.get("skill_id") or ""))
        return graph

    def _build_provider_route(self, agent: dict[str, object], graph: dict[str, object]) -> str:
        provider_config = graph.get("provider_config") if isinstance(graph.get("provider_config"), dict) else {}
        preferred = str(agent.get("preferred_provider_id") or provider_config.get("preferred_provider_id") or "").strip()
        fallback = str(agent.get("fallback_provider_id") or provider_config.get("fallback_provider_id") or "").strip()
        if preferred and fallback:
            return f"{preferred} -> {fallback}"
        if preferred:
            return preferred
        if fallback:
            return f"project default -> {fallback}"
        return "project default"

    def _build_review_mode(self, agent: dict[str, object], graph: dict[str, object]) -> str:
        review_agent = graph.get("review_agent") if isinstance(graph.get("review_agent"), dict) else {}
        review_agent_enabled = bool(review_agent.get("enabled", True))
        if str(agent.get("node_kind") or "agent") == "cluster":
            if bool(agent.get("cluster_auto_review", True)) and review_agent_enabled:
                return "cluster summary plus team review agent"
            if bool(agent.get("cluster_auto_review", True)):
                return "cluster-local review only"
            if review_agent_enabled:
                return "team review agent only"
            return "direct cluster handoff"
        if review_agent_enabled:
            return "team review agent"
        return "direct handoff"

    def _infer_tool_execution_support(self, agent: dict[str, object], graph: dict[str, object]) -> tuple[str, str]:
        provider_config = graph.get("provider_config") if isinstance(graph.get("provider_config"), dict) else {}
        provider_hint = str(
            agent.get("preferred_provider_id")
            or agent.get("fallback_provider_id")
            or provider_config.get("preferred_provider_id")
            or provider_config.get("fallback_provider_id")
            or ""
        ).strip()
        model = str(agent.get("model") or "gpt-5.2").strip()
        return infer_tool_calling_support(model=model, provider_id=provider_hint)

    def _collect_suggested_tool_ids(
        self,
        *,
        relevant_skill_ids: list[str],
        skill_catalog_by_id: dict[str, dict[str, object]],
        tool_catalog: list[object],
    ) -> tuple[list[str], dict[str, str]]:
        tool_status_by_id: dict[str, str] = {}
        for item in tool_catalog:
            if not isinstance(item, dict):
                continue
            tool_id = self._normalize_skill_key(str(item.get("tool_id") or ""))
            if not tool_id:
                continue
            tool_status_by_id[tool_id] = str(item.get("status") or "enabled")

        suggested_tool_ids: list[str] = []
        seen: set[str] = set()
        for skill_id in relevant_skill_ids:
            skill_item = skill_catalog_by_id.get(self._normalize_skill_key(skill_id), {})
            for tool_id in skill_item.get("suggested_tool_ids") or []:
                normalized_tool_id = self._normalize_skill_key(str(tool_id))
                if not normalized_tool_id or normalized_tool_id not in tool_status_by_id or normalized_tool_id in seen:
                    continue
                seen.add(normalized_tool_id)
                suggested_tool_ids.append(normalized_tool_id)

        return suggested_tool_ids, tool_status_by_id

    def _resolve_agent_tool_access(
        self,
        *,
        agent: dict[str, object],
        relevant_skill_ids: list[str],
        skill_catalog_by_id: dict[str, dict[str, object]],
        tool_catalog: list[object],
    ) -> dict[str, list[str]]:
        suggested_tool_ids, tool_status_by_id = self._collect_suggested_tool_ids(
            relevant_skill_ids=relevant_skill_ids,
            skill_catalog_by_id=skill_catalog_by_id,
            tool_catalog=tool_catalog,
        )
        allowed_tool_ids = self._normalize_identifier_list(agent.get("allowed_tool_ids"))
        denied_tool_ids = self._normalize_identifier_list(agent.get("denied_tool_ids"))
        tool_catalog_ids = set(tool_status_by_id)
        configured_allowed_tool_ids = [tool_id for tool_id in allowed_tool_ids if tool_id in tool_catalog_ids]
        configured_denied_tool_ids = [tool_id for tool_id in denied_tool_ids if tool_id in tool_catalog_ids]
        unknown_allowed_tool_ids = [tool_id for tool_id in allowed_tool_ids if tool_id not in tool_catalog_ids]

        if configured_allowed_tool_ids:
            candidate_tool_ids = list(configured_allowed_tool_ids)
            policy_added_tool_ids = [
                tool_id for tool_id in configured_allowed_tool_ids if tool_id not in suggested_tool_ids
            ]
            allow_blocked_tool_ids = [
                tool_id for tool_id in suggested_tool_ids if tool_id not in configured_allowed_tool_ids
            ]
        else:
            candidate_tool_ids = list(suggested_tool_ids)
            policy_added_tool_ids = []
            allow_blocked_tool_ids = []

        denied_tool_id_set = set(configured_denied_tool_ids)
        deny_blocked_tool_ids = [tool_id for tool_id in candidate_tool_ids if tool_id in denied_tool_id_set]
        candidate_tool_ids = [tool_id for tool_id in candidate_tool_ids if tool_id not in denied_tool_id_set]

        return {
            "configured_allowed_tool_ids": configured_allowed_tool_ids,
            "configured_denied_tool_ids": configured_denied_tool_ids,
            "enabled_tool_ids": [
                tool_id for tool_id in candidate_tool_ids if tool_status_by_id.get(tool_id) == "enabled"
            ],
            "disabled_tool_ids": [
                tool_id for tool_id in candidate_tool_ids if tool_status_by_id.get(tool_id) == "disabled"
            ],
            "policy_added_tool_ids": policy_added_tool_ids,
            "policy_blocked_tool_ids": self._dedupe_preserve_order(
                [*allow_blocked_tool_ids, *deny_blocked_tool_ids]
            ),
            "unknown_allowed_tool_ids": unknown_allowed_tool_ids,
        }

    def _collect_suggested_mcp_server_ids(
        self,
        *,
        relevant_skill_ids: list[str],
        skill_catalog_by_id: dict[str, dict[str, object]],
    ) -> list[str]:
        suggested_server_ids: list[str] = []
        seen: set[str] = set()
        for skill_id in relevant_skill_ids:
            skill_item = skill_catalog_by_id.get(self._normalize_skill_key(skill_id), {})
            for server_id in skill_item.get("suggested_mcp_server_ids") or []:
                normalized_server_id = self._normalize_skill_key(str(server_id))
                if not normalized_server_id or normalized_server_id in seen:
                    continue
                seen.add(normalized_server_id)
                suggested_server_ids.append(normalized_server_id)

        return suggested_server_ids

    def _resolve_agent_mcp_access(
        self,
        *,
        agent: dict[str, object],
        relevant_skill_ids: list[str],
        skill_catalog_by_id: dict[str, dict[str, object]],
        mcp_server_catalog: list[object],
        mcp_alias_to_server_id: dict[str, str],
    ) -> dict[str, list[str]]:
        mcp_status_by_id: dict[str, str] = {}
        for item in mcp_server_catalog:
            if not isinstance(item, dict):
                continue
            server_id = self._normalize_skill_key(str(item.get("server_id") or ""))
            if not server_id:
                continue
            mcp_status_by_id[server_id] = str(item.get("status") or "enabled")

        suggested_server_ids = self._collect_suggested_mcp_server_ids(
            relevant_skill_ids=relevant_skill_ids,
            skill_catalog_by_id=skill_catalog_by_id,
        )
        suggested_server_ids = self._canonicalize_mcp_server_ids(
            suggested_server_ids,
            alias_lookup=mcp_alias_to_server_id,
        )
        raw_allowed_mcp_server_ids = self._normalize_identifier_list(agent.get("allowed_mcp_server_ids"))
        raw_denied_mcp_server_ids = self._normalize_identifier_list(agent.get("denied_mcp_server_ids"))
        allowed_mcp_server_ids = self._canonicalize_mcp_server_ids(
            raw_allowed_mcp_server_ids,
            alias_lookup=mcp_alias_to_server_id,
        )
        denied_mcp_server_ids = self._canonicalize_mcp_server_ids(
            raw_denied_mcp_server_ids,
            alias_lookup=mcp_alias_to_server_id,
        )
        mcp_catalog_ids = set(mcp_status_by_id)
        configured_allowed_mcp_server_ids = [
            server_id for server_id in allowed_mcp_server_ids if server_id in mcp_catalog_ids
        ]
        configured_denied_mcp_server_ids = [
            server_id for server_id in denied_mcp_server_ids if server_id in mcp_catalog_ids
        ]
        unknown_allowed_mcp_server_ids = [
            server_id
            for server_id in raw_allowed_mcp_server_ids
            if mcp_alias_to_server_id.get(server_id, server_id) not in mcp_catalog_ids
        ]

        if allowed_mcp_server_ids:
            candidate_server_ids = list(allowed_mcp_server_ids)
            policy_added_mcp_server_ids = [
                server_id for server_id in allowed_mcp_server_ids if server_id not in suggested_server_ids
            ]
            allow_blocked_mcp_server_ids = [
                server_id for server_id in suggested_server_ids if server_id not in allowed_mcp_server_ids
            ]
        else:
            candidate_server_ids = list(suggested_server_ids)
            policy_added_mcp_server_ids = []
            allow_blocked_mcp_server_ids = []

        denied_mcp_server_id_set = set(denied_mcp_server_ids)
        deny_blocked_mcp_server_ids = [
            server_id for server_id in candidate_server_ids if server_id in denied_mcp_server_id_set
        ]
        candidate_server_ids = [
            server_id for server_id in candidate_server_ids if server_id not in denied_mcp_server_id_set
        ]

        return {
            "configured_allowed_mcp_server_ids": configured_allowed_mcp_server_ids,
            "configured_denied_mcp_server_ids": configured_denied_mcp_server_ids,
            "mcp_server_ids": [
                server_id for server_id in candidate_server_ids if mcp_status_by_id.get(server_id) == "enabled"
            ],
            "missing_mcp_server_ids": [
                server_id for server_id in candidate_server_ids if mcp_status_by_id.get(server_id) != "enabled"
            ],
            "policy_added_mcp_server_ids": policy_added_mcp_server_ids,
            "policy_blocked_mcp_server_ids": self._dedupe_preserve_order(
                [*allow_blocked_mcp_server_ids, *deny_blocked_mcp_server_ids]
            ),
            "unknown_allowed_mcp_server_ids": unknown_allowed_mcp_server_ids,
        }

    def _match_skills_for_intents(self, intents: list[str], graph: dict[str, object]) -> list[str]:
        catalog = graph.get("skill_catalog") or []
        matches: list[str] = []
        normalized_intents = [self._normalize_skill_key(intent) for intent in intents if self._normalize_skill_key(intent)]
        for item in catalog:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("skill_id") or "").strip()
            if not skill_id:
                continue
            search_space = " ".join(
                [
                    self._normalize_skill_key(skill_id),
                    self._normalize_skill_key(str(item.get("title") or "")),
                    self._normalize_skill_key(str(item.get("source") or "")),
                    self._normalize_skill_key(str(item.get("description") or "")),
                    self._normalize_skill_key(str(item.get("prompt_hint") or "")),
                ]
            )
            if any(intent and intent in search_space for intent in normalized_intents):
                matches.append(skill_id)
        return sorted(dict.fromkeys(matches))

    def _build_skill_catalog_detail(
        self,
        skill_id: str,
        *,
        skill_catalog_by_id: dict[str, dict[str, object]],
    ) -> dict[str, object] | None:
        normalized_skill_id = self._normalize_skill_key(skill_id)
        if not normalized_skill_id:
            return None
        catalog_item = dict(skill_catalog_by_id.get(normalized_skill_id) or {})
        if catalog_item:
            return HarnessSkillCatalogItem(
                skill_id=normalized_skill_id,
                title=str(catalog_item.get("title") or normalized_skill_id).strip() or normalized_skill_id,
                description=str(catalog_item.get("description") or "").strip() or None,
                source=str(catalog_item.get("source") or f"app/skills/{normalized_skill_id}").strip()
                or f"app/skills/{normalized_skill_id}",
                status=str(catalog_item.get("status") or "available").strip() or "available",
                prompt_hint=str(catalog_item.get("prompt_hint") or "").strip() or None,
                suggested_tool_ids=self._normalize_identifier_list(catalog_item.get("suggested_tool_ids")),
                suggested_mcp_server_ids=self._normalize_identifier_list(
                    catalog_item.get("suggested_mcp_server_ids")
                ),
            ).model_dump()

        descriptor = get_skill_descriptor(normalized_skill_id) or build_fallback_skill_descriptor(normalized_skill_id)
        return HarnessSkillCatalogItem(
            skill_id=descriptor.skill_id,
            title=descriptor.title,
            description=descriptor.description or None,
            source=f"app/skills/{descriptor.skill_id}",
            status="available",
            prompt_hint=descriptor.prompt_hint or None,
            suggested_tool_ids=list(descriptor.suggested_tool_ids),
            suggested_mcp_server_ids=list(descriptor.suggested_mcp_server_ids),
        ).model_dump()

    def _build_missing_skill_details(
        self,
        skill_ids: list[str],
        *,
        skill_catalog_by_id: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        seen: set[str] = set()
        for skill_id in skill_ids:
            normalized_skill_id = self._normalize_skill_key(skill_id)
            if not normalized_skill_id or normalized_skill_id in seen:
                continue
            detail = self._build_skill_catalog_detail(
                normalized_skill_id,
                skill_catalog_by_id=skill_catalog_by_id,
            )
            if detail is None:
                continue
            seen.add(normalized_skill_id)
            details.append(detail)
        return details

    def _build_missing_mcp_server_details(
        self,
        server_ids: list[str],
        *,
        mcp_server_catalog_by_id: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        seen: set[str] = set()
        for server_id in server_ids:
            normalized_server_id = self._normalize_skill_key(server_id)
            if not normalized_server_id or normalized_server_id in seen:
                continue
            seen.add(normalized_server_id)
            catalog_item = dict(mcp_server_catalog_by_id.get(normalized_server_id) or {})
            if catalog_item:
                details.append(
                    HarnessMcpServerCatalogItem(
                        server_id=normalized_server_id,
                        title=str(catalog_item.get("title") or normalized_server_id).strip()
                        or self._humanize_identifier(normalized_server_id),
                        description=str(catalog_item.get("description") or "").strip() or None,
                        status=str(catalog_item.get("status") or "disabled").strip() or "disabled",
                        command_preview=str(catalog_item.get("command_preview") or "").strip() or None,
                    ).model_dump()
                )
                continue
            details.append(
                HarnessMcpServerCatalogItem(
                    server_id=normalized_server_id,
                    title=self._humanize_identifier(normalized_server_id),
                    description="No project MCP server inventory entry was found for this identifier.",
                    status="disabled",
                    command_preview=None,
                ).model_dump()
            )
        return details

    def _build_agent_readiness(self, summary: dict[str, object]) -> dict[str, object]:
        loaded_skill_ids = [str(item).strip() for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
        missing_skill_ids = [str(item).strip() for item in summary.get("missing_skill_ids") or [] if str(item).strip()]
        configured_allowed_tool_ids = [
            str(item).strip() for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
        ]
        disabled_tool_ids = [str(item).strip() for item in summary.get("disabled_tool_ids") or [] if str(item).strip()]
        provider_limited_tool_ids = [
            str(item).strip() for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        ]
        configured_allowed_mcp_server_ids = [
            str(item).strip() for item in summary.get("configured_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        missing_mcp_server_ids = [str(item).strip() for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
        unknown_allowed_tool_ids = [
            str(item).strip() for item in summary.get("unknown_allowed_tool_ids") or [] if str(item).strip()
        ]
        unknown_allowed_mcp_server_ids = [
            str(item).strip() for item in summary.get("unknown_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()

        blockers: list[str] = []
        warnings: list[str] = []

        if missing_skill_ids:
            blockers.append(
                "Missing approved skills before this node can run: "
                f"{self._format_preview_list(missing_skill_ids)}"
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
                f"{self._format_preview_list(missing_mcp_server_ids)}"
            )
        if has_explicit_capability_requirements and provider_limited_tool_ids:
            line = (
                "Current provider route cannot execute these tools directly: "
                f"{self._format_preview_list(provider_limited_tool_ids)}"
            )
            if tool_execution_support_reason:
                line += f" ({tool_execution_support_reason})"
            warnings.append(line)
        if has_explicit_capability_requirements and disabled_tool_ids:
            warnings.append(
                "Some relevant tools stay disabled until feature flags change: "
                f"{self._format_preview_list(disabled_tool_ids)}"
            )
        if unknown_allowed_tool_ids:
            warnings.append(
                "Node policy references unknown tool ids: "
                f"{self._format_preview_list(unknown_allowed_tool_ids)}"
            )
        if unknown_allowed_mcp_server_ids:
            warnings.append(
                "Node policy references unknown MCP ids: "
                f"{self._format_preview_list(unknown_allowed_mcp_server_ids)}"
            )

        if blockers:
            status = "blocked"
        elif warnings:
            status = "limited"
        else:
            status = "ready"

        return {
            "readiness_status": status,
            "readiness_blockers": blockers,
            "readiness_warnings": warnings,
        }

    def _build_agent_availability(
        self,
        agent: dict[str, object],
        *,
        summary: dict[str, object],
        approved_skill_ids: set[str],
        tool_catalog_by_id: dict[str, dict[str, object]],
        mcp_status_by_id: dict[str, str],
        mcp_alias_to_server_id: dict[str, str],
    ) -> dict[str, object]:
        required_skill_ids = self._normalize_identifier_list(agent.get("required_skill_ids"))
        missing_required_skill_ids = [
            skill_id for skill_id in required_skill_ids if skill_id not in approved_skill_ids
        ]
        required_tool_ids = self._normalize_identifier_list(agent.get("required_tool_ids"))
        missing_required_tool_ids: list[str] = []
        required_mcp_server_ids = self._canonicalize_mcp_server_ids(
            self._normalize_identifier_list(agent.get("required_mcp_server_ids")),
            alias_lookup=mcp_alias_to_server_id,
        )
        missing_required_mcp_server_ids = [
            server_id for server_id in required_mcp_server_ids if mcp_status_by_id.get(server_id) != "enabled"
        ]
        enabled_tool_ids = {
            str(item).strip() for item in summary.get("enabled_tool_ids") or [] if str(item).strip()
        }
        disabled_tool_ids = {
            str(item).strip() for item in summary.get("disabled_tool_ids") or [] if str(item).strip()
        }
        provider_limited_tool_ids = {
            str(item).strip() for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        }
        configured_allowed_tool_ids = {
            str(item).strip() for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
        }
        configured_denied_tool_ids = {
            str(item).strip() for item in summary.get("configured_denied_tool_ids") or [] if str(item).strip()
        }
        tool_catalog_ids = set(tool_catalog_by_id)
        tool_execution_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()
        requires_tool_calling = bool(agent.get("requires_tool_calling", False))
        requires_direct_tool_support = requires_tool_calling or bool(required_tool_ids)

        blockers: list[str] = []
        warnings: list[str] = []

        if missing_required_skill_ids:
            blockers.append(
                "Definition requires approved skills that are not yet in the project pool: "
                f"{self._format_preview_list(missing_required_skill_ids)}"
            )
        if required_tool_ids:
            missing_required_tool_ids = [
                tool_id for tool_id in required_tool_ids if tool_id not in enabled_tool_ids
            ]
            unknown_required_tool_ids = [
                tool_id for tool_id in required_tool_ids if tool_id not in tool_catalog_ids
            ]
            disabled_required_tool_ids = [
                tool_id for tool_id in required_tool_ids if tool_id in disabled_tool_ids
            ]
            provider_limited_required_tool_ids = [
                tool_id for tool_id in required_tool_ids if tool_id in provider_limited_tool_ids
            ]
            policy_missing_required_tool_ids = [
                tool_id
                for tool_id in missing_required_tool_ids
                if tool_id not in unknown_required_tool_ids
                and tool_id not in disabled_required_tool_ids
                and tool_id not in provider_limited_required_tool_ids
            ]
            if unknown_required_tool_ids:
                blockers.append(
                    "Definition requires tool ids that are not present in project inventory: "
                    f"{self._format_preview_list(unknown_required_tool_ids)}"
                )
            if disabled_required_tool_ids:
                blockers.append(
                    "Definition requires tools that are currently disabled in project inventory: "
                    f"{self._format_preview_list(disabled_required_tool_ids)}"
                )
            if policy_missing_required_tool_ids:
                if configured_allowed_tool_ids or configured_denied_tool_ids:
                    blockers.append(
                        "Definition requires tools that the current node tool policy does not enable: "
                        f"{self._format_preview_list(policy_missing_required_tool_ids)}"
                    )
                else:
                    blockers.append(
                        "Definition requires tools that are not currently enabled for this node; "
                        "add the relevant skills or an allow-only tool policy: "
                        f"{self._format_preview_list(policy_missing_required_tool_ids)}"
                    )
            if provider_limited_required_tool_ids:
                line = (
                    "Definition requires tools that the current provider route cannot execute directly: "
                    f"{self._format_preview_list(provider_limited_required_tool_ids)}"
                )
                if tool_execution_support_reason:
                    line += f" ({tool_execution_support_reason})"
                blockers.append(line)
        if missing_required_mcp_server_ids:
            blockers.append(
                "Definition requires enabled MCP servers that are not currently available: "
                f"{self._format_preview_list(missing_required_mcp_server_ids)}"
            )
        if requires_direct_tool_support:
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

        if blockers:
            status = "unavailable"
        elif warnings:
            status = "limited"
        else:
            status = "available"

        return {
            "required_skill_ids": required_skill_ids,
            "missing_required_skill_ids": missing_required_skill_ids,
            "required_tool_ids": required_tool_ids,
            "missing_required_tool_ids": missing_required_tool_ids,
            "requires_tool_calling": requires_tool_calling,
            "required_mcp_server_ids": required_mcp_server_ids,
            "missing_required_mcp_server_ids": missing_required_mcp_server_ids,
            "availability_status": status,
            "availability_blockers": blockers,
            "availability_warnings": warnings,
        }

    def _summarize_agent_readiness_counts(
        self,
        summaries: list[dict[str, object]],
        *,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, int]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        counts = {
            "total_agent_count": 0,
            "ready_agent_count": 0,
            "limited_agent_count": 0,
            "blocked_agent_count": 0,
        }
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id or (selected and agent_id not in selected):
                continue
            counts["total_agent_count"] += 1
            status = str(summary.get("readiness_status") or "ready").strip() or "ready"
            if status == "blocked":
                counts["blocked_agent_count"] += 1
            elif status == "limited":
                counts["limited_agent_count"] += 1
            else:
                counts["ready_agent_count"] += 1
        return counts

    def _summarize_agent_availability_counts(
        self,
        summaries: list[dict[str, object]],
        *,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, int]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        counts = {
            "total_agent_count": 0,
            "available_agent_count": 0,
            "limited_agent_count": 0,
            "unavailable_agent_count": 0,
        }
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id or (selected and agent_id not in selected):
                continue
            counts["total_agent_count"] += 1
            status = str(summary.get("availability_status") or "available").strip() or "available"
            if status == "unavailable":
                counts["unavailable_agent_count"] += 1
            elif status == "limited":
                counts["limited_agent_count"] += 1
            else:
                counts["available_agent_count"] += 1
        return counts

    def _summarize_agent_execution_contract_counts(
        self,
        summaries: list[dict[str, object]],
        *,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, int]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        counts = {
            "total_agent_count": 0,
            "direct_execution_agent_count": 0,
            "planning_only_tool_agent_count": 0,
            "planning_only_mcp_agent_count": 0,
        }
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id or (selected and agent_id not in selected):
                continue
            counts["total_agent_count"] += 1
            execution_contract = (
                dict(summary.get("execution_contract") or {})
                if isinstance(summary.get("execution_contract"), dict)
                else self._build_agent_execution_contract(summary)
            )
            tool_access_mode = str(execution_contract.get("tool_access_mode") or "").strip() or "none"
            planning_only_tool_ids = [
                str(item)
                for item in execution_contract.get("planning_only_tool_ids") or []
                if str(item).strip()
            ]
            planning_only_mcp_server_ids = [
                str(item)
                for item in execution_contract.get("planning_only_mcp_server_ids") or []
                if str(item).strip()
            ]
            if tool_access_mode in {"direct_execution", "mixed"}:
                counts["direct_execution_agent_count"] += 1
            if planning_only_tool_ids:
                counts["planning_only_tool_agent_count"] += 1
            if planning_only_mcp_server_ids:
                counts["planning_only_mcp_agent_count"] += 1
        return counts

    def _summarize_agent_delegation_contract_counts(
        self,
        summaries: list[dict[str, object]],
        *,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, int]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        counts = {
            "total_agent_count": 0,
            "coordinator_agent_count": 0,
            "parallel_coordinator_agent_count": 0,
            "final_output_agent_count": 0,
            "verification_agent_count": 0,
        }
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id or (selected and agent_id not in selected):
                continue
            counts["total_agent_count"] += 1
            delegation_contract = (
                dict(summary.get("delegation_contract") or {})
                if isinstance(summary.get("delegation_contract"), dict)
                else {}
            )
            primary_role_mode = str(delegation_contract.get("primary_role_mode") or "").strip() or "generalist"
            supporting_role_modes = {
                str(item).strip()
                for item in delegation_contract.get("supporting_role_modes") or []
                if str(item).strip()
            }
            work_strategy = str(delegation_contract.get("work_strategy") or "").strip()
            if primary_role_mode == "coordinator":
                counts["coordinator_agent_count"] += 1
            if bool(delegation_contract.get("should_coordinate_parallel_work")):
                counts["parallel_coordinator_agent_count"] += 1
            if bool(delegation_contract.get("should_produce_final_output")):
                counts["final_output_agent_count"] += 1
            if (
                primary_role_mode == "verification"
                or "verification" in supporting_role_modes
                or work_strategy == "verify_and_close"
            ):
                counts["verification_agent_count"] += 1
        return counts

    @staticmethod
    def _normalize_orchestration_text(value: object) -> str:
        return str(value or "").strip().lower()

    def _matches_orchestration_keywords(
        self,
        values: list[object],
        keywords: tuple[str, ...],
    ) -> bool:
        haystack = " ".join(
            self._normalize_orchestration_text(value)
            for value in values
            if str(value or "").strip()
        )
        if not haystack:
            return False
        return any(keyword in haystack for keyword in keywords)

    @staticmethod
    def _coordination_preview(*, agent_id: str, agent_name: str) -> dict[str, object]:
        return HarnessCoordinationAgentPreview(
            agent_id=agent_id,
            agent_name=agent_name,
        ).model_dump()

    def _pick_top_coordination_anchors(
        self,
        entries: list[dict[str, object]],
        *,
        max_count: int = 3,
    ) -> list[dict[str, object]]:
        ranked = [
            {
                "agent_id": str(entry.get("agent_id") or "").strip(),
                "agent_name": str(entry.get("agent_name") or entry.get("agent_id") or "").strip(),
                "score": int(entry.get("score") or 0),
            }
            for entry in entries
            if isinstance(entry, dict)
            and str(entry.get("agent_id") or "").strip()
            and str(entry.get("agent_name") or entry.get("agent_id") or "").strip()
            and int(entry.get("score") or 0) > 0
        ]
        ranked.sort(
            key=lambda item: (
                -int(item["score"]),
                str(item["agent_name"]),
            )
        )
        return [
            self._coordination_preview(
                agent_id=str(entry["agent_id"]),
                agent_name=str(entry["agent_name"]),
            )
            for entry in ranked[:max_count]
        ]

    def _compute_actionable_tool_policy_suggestion_ids(
        self,
        agent: dict[str, object] | None,
        blocked_tool_ids: list[str],
    ) -> list[str]:
        if not isinstance(agent, dict):
            return []
        normalized_blocked_tool_ids = self._normalize_identifier_list(blocked_tool_ids)
        if not normalized_blocked_tool_ids:
            return []
        allowed_tool_ids = set(self._normalize_identifier_list(agent.get("allowed_tool_ids")))
        denied_tool_ids = set(self._normalize_identifier_list(agent.get("denied_tool_ids")))
        has_explicit_allowed_tools = bool(allowed_tool_ids)
        return [
            tool_id
            for tool_id in normalized_blocked_tool_ids
            if (has_explicit_allowed_tools and tool_id not in allowed_tool_ids) or tool_id in denied_tool_ids
        ]

    def _compute_actionable_mcp_policy_suggestion_ids(
        self,
        agent: dict[str, object] | None,
        blocked_mcp_server_ids: list[str],
    ) -> list[str]:
        if not isinstance(agent, dict):
            return []
        normalized_blocked_mcp_server_ids = self._normalize_identifier_list(blocked_mcp_server_ids)
        if not normalized_blocked_mcp_server_ids:
            return []
        allowed_mcp_server_ids = set(self._normalize_identifier_list(agent.get("allowed_mcp_server_ids")))
        denied_mcp_server_ids = set(self._normalize_identifier_list(agent.get("denied_mcp_server_ids")))
        has_explicit_allowed_mcp_servers = bool(allowed_mcp_server_ids)
        return [
            server_id
            for server_id in normalized_blocked_mcp_server_ids
            if (has_explicit_allowed_mcp_servers and server_id not in allowed_mcp_server_ids)
            or server_id in denied_mcp_server_ids
        ]

    def _compute_actionable_coordinator_tool_policy_restriction_ids(
        self,
        agent: dict[str, object] | None,
        summary: dict[str, object] | None,
    ) -> list[str]:
        if not isinstance(agent, dict) or not isinstance(summary, dict):
            return []
        delegation_contract = (
            dict(summary.get("delegation_contract") or {})
            if isinstance(summary.get("delegation_contract"), dict)
            else self._build_agent_delegation_contract(
                agent=agent,
                summary=summary,
                agents_by_id={str(agent.get("agent_id") or "").strip(): agent},
                edges=[],
                orchestration_summary=None,
            )
        )
        role_hint = self._normalize_skill_key(str(agent.get("role") or summary.get("role") or ""))
        primary_role_mode = str(delegation_contract.get("primary_role_mode") or "generalist").strip() or "generalist"
        should_coordinate_parallel_work = bool(delegation_contract.get("should_coordinate_parallel_work"))
        should_produce_final_output = bool(delegation_contract.get("should_produce_final_output"))
        if should_produce_final_output or (
            primary_role_mode != "coordinator"
            and role_hint != "coordinator"
            and not should_coordinate_parallel_work
        ):
            return []
        execution_contract = (
            dict(summary.get("execution_contract") or {})
            if isinstance(summary.get("execution_contract"), dict)
            else self._build_agent_execution_contract(summary)
        )
        tool_access_mode = str(execution_contract.get("tool_access_mode") or "none").strip() or "none"
        executable_tool_ids = self._normalize_identifier_list(execution_contract.get("executable_tool_ids"))
        if not executable_tool_ids and tool_access_mode in {"direct_execution", "mixed"}:
            executable_tool_ids = self._normalize_identifier_list(summary.get("enabled_tool_ids"))
        denied_tool_ids = set(self._normalize_identifier_list(agent.get("denied_tool_ids")))
        return [
            tool_id
            for tool_id in executable_tool_ids
            if tool_id not in self._COORDINATOR_ALLOWED_DIRECT_TOOL_IDS and tool_id not in denied_tool_ids
        ]

    def _compute_actionable_coordinator_mcp_policy_restriction_ids(
        self,
        agent: dict[str, object] | None,
        summary: dict[str, object] | None,
    ) -> list[str]:
        if not isinstance(agent, dict) or not isinstance(summary, dict):
            return []
        delegation_contract = (
            dict(summary.get("delegation_contract") or {})
            if isinstance(summary.get("delegation_contract"), dict)
            else self._build_agent_delegation_contract(
                agent=agent,
                summary=summary,
                agents_by_id={str(agent.get("agent_id") or "").strip(): agent},
                edges=[],
                orchestration_summary=None,
            )
        )
        role_hint = self._normalize_skill_key(str(agent.get("role") or summary.get("role") or ""))
        primary_role_mode = str(delegation_contract.get("primary_role_mode") or "generalist").strip() or "generalist"
        should_coordinate_parallel_work = bool(delegation_contract.get("should_coordinate_parallel_work"))
        should_produce_final_output = bool(delegation_contract.get("should_produce_final_output"))
        if should_produce_final_output or (
            primary_role_mode != "coordinator"
            and role_hint != "coordinator"
            and not should_coordinate_parallel_work
        ):
            return []
        execution_contract = (
            dict(summary.get("execution_contract") or {})
            if isinstance(summary.get("execution_contract"), dict)
            else self._build_agent_execution_contract(summary)
        )
        mcp_access_mode = str(execution_contract.get("mcp_access_mode") or "none").strip() or "none"
        planning_only_mcp_server_ids = self._normalize_identifier_list(
            execution_contract.get("planning_only_mcp_server_ids")
        )
        if not planning_only_mcp_server_ids and mcp_access_mode == "planning_only":
            planning_only_mcp_server_ids = self._normalize_identifier_list(summary.get("mcp_server_ids"))
        denied_mcp_server_ids = set(self._normalize_identifier_list(agent.get("denied_mcp_server_ids")))
        return [
            server_id
            for server_id in planning_only_mcp_server_ids
            if server_id not in denied_mcp_server_ids
        ]

    def _build_orchestration_topology_summary(
        self,
        *,
        agents: list[dict[str, object]],
        edges: list[dict[str, object]],
        summaries: list[dict[str, object]],
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, object]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        filtered_agents = [
            agent
            for agent in agents
            if isinstance(agent, dict)
            and str(agent.get("agent_id") or "").strip()
            and (not selected or str(agent.get("agent_id") or "").strip() in selected)
        ]
        if not filtered_agents:
            return {
                "total_agent_count": 0,
                "total_lane_count": 0,
                "shared_lane_count": 0,
                "single_owner_lane_count": 0,
                "isolated_agent_count": 0,
                "underconnected_agent_count": 0,
                "isolated_agents": [],
                "underconnected_agents": [],
                "shared_lane_ids": [],
                "single_owner_lane_ids": [],
            }

        included_agent_ids = {
            str(agent.get("agent_id") or "").strip()
            for agent in filtered_agents
        }
        summary_by_agent_id = {
            str(summary.get("agent_id") or "").strip(): dict(summary)
            for summary in summaries
            if isinstance(summary, dict)
            and str(summary.get("agent_id") or "").strip() in included_agent_ids
        }
        inbound_count_by_agent_id = {
            str(agent.get("agent_id") or "").strip(): 0
            for agent in filtered_agents
        }
        outbound_count_by_agent_id = {
            str(agent.get("agent_id") or "").strip(): 0
            for agent in filtered_agents
        }
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source_agent_id = str(edge.get("source_agent_id") or "").strip()
            target_agent_id = str(edge.get("target_agent_id") or "").strip()
            if source_agent_id not in included_agent_ids or target_agent_id not in included_agent_ids:
                continue
            outbound_count_by_agent_id[source_agent_id] = outbound_count_by_agent_id.get(source_agent_id, 0) + 1
            inbound_count_by_agent_id[target_agent_id] = inbound_count_by_agent_id.get(target_agent_id, 0) + 1

        lane_owners: dict[str, list[dict[str, object]]] = {}
        isolated_agents: list[dict[str, object]] = []
        underconnected_agents: list[dict[str, object]] = []

        for agent in filtered_agents:
            agent_id = str(agent.get("agent_id") or "").strip()
            agent_name = str(agent.get("name") or agent_id).strip() or agent_id
            summary = summary_by_agent_id.get(agent_id, {})
            lane_ids = [
                str(item).strip()
                for item in summary.get("delegation_lane_ids") or []
                if str(item).strip()
            ]
            for lane_id in lane_ids:
                owners = lane_owners.setdefault(lane_id, [])
                if not any(str(owner.get("agent_id") or "").strip() == agent_id for owner in owners):
                    owners.append(self._coordination_preview(agent_id=agent_id, agent_name=agent_name))

            inbound_count = inbound_count_by_agent_id.get(agent_id, 0)
            outbound_count = outbound_count_by_agent_id.get(agent_id, 0)
            if inbound_count == 0 and outbound_count == 0:
                isolated_agents.append(
                    self._coordination_preview(agent_id=agent_id, agent_name=agent_name)
                )
                continue

            recommended_collaborators = [
                dict(item)
                for item in summary.get("recommended_collaborators") or []
                if isinstance(item, dict)
            ]
            downstream_handoff_scores = [
                dict(item)
                for item in summary.get("downstream_handoff_scores") or []
                if isinstance(item, dict)
            ]
            has_bridge_opportunity = any(
                not bool(item.get("edge_present"))
                and str(item.get("fit") or "").strip() in {"strong", "good"}
                for item in recommended_collaborators
            )
            has_weak_downstream = any(
                bool(item.get("edge_present"))
                and str(item.get("fit") or "").strip() in {"", "weak"}
                for item in downstream_handoff_scores
            )
            if has_bridge_opportunity or has_weak_downstream:
                underconnected_agents.append(
                    self._coordination_preview(agent_id=agent_id, agent_name=agent_name)
                )

        for owners in lane_owners.values():
            owners.sort(key=lambda item: str(item.get("agent_name") or ""))

        shared_lane_ids = sorted(lane_id for lane_id, owners in lane_owners.items() if len(owners) > 1)
        single_owner_lane_ids = sorted(lane_id for lane_id, owners in lane_owners.items() if len(owners) == 1)
        isolated_agents.sort(key=lambda item: str(item.get("agent_name") or ""))
        underconnected_agents.sort(key=lambda item: str(item.get("agent_name") or ""))

        return {
            "total_agent_count": len(filtered_agents),
            "total_lane_count": len(lane_owners),
            "shared_lane_count": len(shared_lane_ids),
            "single_owner_lane_count": len(single_owner_lane_ids),
            "isolated_agent_count": len(isolated_agents),
            "underconnected_agent_count": len(underconnected_agents),
            "isolated_agents": isolated_agents,
            "underconnected_agents": underconnected_agents,
            "shared_lane_ids": shared_lane_ids,
            "single_owner_lane_ids": single_owner_lane_ids,
        }

    @staticmethod
    def _append_capability_owner(
        owner_map: dict[str, list[dict[str, object]]],
        *,
        capability_id: str,
        owner: dict[str, object],
    ) -> None:
        owners = owner_map.setdefault(capability_id, [])
        owner_agent_id = str(owner.get("agent_id") or "").strip()
        if owner_agent_id and not any(str(item.get("agent_id") or "").strip() == owner_agent_id for item in owners):
            owners.append(owner)
            owners.sort(key=lambda item: str(item.get("agent_name") or ""))

    def _build_orchestration_capability_coverage_summary(
        self,
        *,
        agents: list[dict[str, object]],
        summaries: list[dict[str, object]],
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, object]:
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        filtered_agents = [
            agent
            for agent in agents
            if isinstance(agent, dict)
            and str(agent.get("agent_id") or "").strip()
            and (not selected or str(agent.get("agent_id") or "").strip() in selected)
        ]
        if not filtered_agents:
            return {
                "total_skill_count": 0,
                "total_tool_count": 0,
                "total_mcp_count": 0,
                "shared_skill_ids": [],
                "single_owner_skills": [],
                "shared_tool_ids": [],
                "single_owner_tools": [],
                "shared_mcp_server_ids": [],
                "single_owner_mcp_servers": [],
                "missing_skill_ids": [],
                "blocked_tool_ids": [],
                "missing_mcp_server_ids": [],
            }

        agent_name_by_id = {
            str(agent.get("agent_id") or "").strip(): str(agent.get("name") or agent.get("agent_id") or "").strip()
            or str(agent.get("agent_id") or "").strip()
            for agent in filtered_agents
        }
        included_agent_ids = set(agent_name_by_id)
        capability_owner_map = {
            "skills": {},
            "tools": {},
            "mcp": {},
        }
        missing_skill_ids: set[str] = set()
        blocked_tool_ids: set[str] = set()
        missing_mcp_server_ids: set[str] = set()

        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id or agent_id not in included_agent_ids:
                continue
            owner = self._coordination_preview(
                agent_id=agent_id,
                agent_name=agent_name_by_id.get(agent_id, agent_id),
            )
            for skill_id in [str(item).strip() for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]:
                self._append_capability_owner(capability_owner_map["skills"], capability_id=skill_id, owner=owner)
            for tool_id in [str(item).strip() for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]:
                self._append_capability_owner(capability_owner_map["tools"], capability_id=tool_id, owner=owner)
            for server_id in [str(item).strip() for item in summary.get("mcp_server_ids") or [] if str(item).strip()]:
                self._append_capability_owner(capability_owner_map["mcp"], capability_id=server_id, owner=owner)
            for skill_id in [
                *[str(item).strip() for item in summary.get("missing_skill_ids") or [] if str(item).strip()],
                *[str(item).strip() for item in summary.get("missing_required_skill_ids") or [] if str(item).strip()],
            ]:
                missing_skill_ids.add(skill_id)
            for tool_id in [
                *[str(item).strip() for item in summary.get("policy_blocked_tool_ids") or [] if str(item).strip()],
                *[str(item).strip() for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()],
                *[str(item).strip() for item in summary.get("missing_required_tool_ids") or [] if str(item).strip()],
            ]:
                blocked_tool_ids.add(tool_id)
            for server_id in [
                *[str(item).strip() for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()],
                *[
                    str(item).strip()
                    for item in summary.get("missing_required_mcp_server_ids") or []
                    if str(item).strip()
                ],
                *[
                    str(item).strip()
                    for item in summary.get("policy_blocked_mcp_server_ids") or []
                    if str(item).strip()
                ],
            ]:
                missing_mcp_server_ids.add(server_id)

        def split_coverage(owner_map: dict[str, list[dict[str, object]]]) -> tuple[list[str], list[dict[str, object]]]:
            shared_ids: list[str] = []
            single_owner_entries: list[dict[str, object]] = []
            for capability_id, owners in owner_map.items():
                if len(owners) > 1:
                    shared_ids.append(capability_id)
                else:
                    single_owner_entries.append(
                        HarnessCapabilityOwnerEntry(
                            capability_id=capability_id,
                            owner_agents=owners,
                        ).model_dump()
                    )
            shared_ids.sort()
            single_owner_entries.sort(key=lambda item: str(item.get("capability_id") or ""))
            return shared_ids, single_owner_entries

        shared_skill_ids, single_owner_skills = split_coverage(capability_owner_map["skills"])
        shared_tool_ids, single_owner_tools = split_coverage(capability_owner_map["tools"])
        shared_mcp_server_ids, single_owner_mcp_servers = split_coverage(capability_owner_map["mcp"])

        return {
            "total_skill_count": len(capability_owner_map["skills"]),
            "total_tool_count": len(capability_owner_map["tools"]),
            "total_mcp_count": len(capability_owner_map["mcp"]),
            "shared_skill_ids": shared_skill_ids,
            "single_owner_skills": single_owner_skills,
            "shared_tool_ids": shared_tool_ids,
            "single_owner_tools": single_owner_tools,
            "shared_mcp_server_ids": shared_mcp_server_ids,
            "single_owner_mcp_servers": single_owner_mcp_servers,
            "missing_skill_ids": sorted(missing_skill_ids),
            "blocked_tool_ids": sorted(blocked_tool_ids),
            "missing_mcp_server_ids": sorted(missing_mcp_server_ids),
        }

    def _build_orchestration_summary(
        self,
        graph: dict[str, object],
        *,
        selected_agent_ids: list[str] | None = None,
    ) -> dict[str, object]:
        agents = [
            dict(agent)
            for agent in graph.get("agents") or []
            if isinstance(agent, dict)
        ]
        edges = [
            dict(edge)
            for edge in graph.get("edges") or []
            if isinstance(edge, dict)
        ]
        summaries = [
            dict(summary)
            for summary in graph.get("agent_capability_summaries") or []
            if isinstance(summary, dict)
        ]
        review_enabled = bool((graph.get("review_agent") or {}).get("enabled", True))
        execution_step_count = len(graph.get("execution_checklist") or [])
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids or []
            if str(agent_id).strip()
        }
        filtered_agents = [
            agent
            for agent in agents
            if str(agent.get("agent_id") or "").strip()
            and (not selected or str(agent.get("agent_id") or "").strip() in selected)
        ]
        if not filtered_agents:
            return HarnessOrchestrationSummary(
                execution_step_count=execution_step_count,
                review_enabled=review_enabled,
            ).model_dump()

        included_agent_ids = {
            str(agent.get("agent_id") or "").strip()
            for agent in filtered_agents
        }
        summary_by_agent_id = {
            str(summary.get("agent_id") or "").strip(): dict(summary)
            for summary in summaries
            if str(summary.get("agent_id") or "").strip() in included_agent_ids
        }
        inbound_count_by_agent_id = {
            str(agent.get("agent_id") or "").strip(): 0
            for agent in filtered_agents
        }
        outbound_count_by_agent_id = {
            str(agent.get("agent_id") or "").strip(): 0
            for agent in filtered_agents
        }
        for edge in edges:
            source_agent_id = str(edge.get("source_agent_id") or "").strip()
            target_agent_id = str(edge.get("target_agent_id") or "").strip()
            if source_agent_id not in included_agent_ids or target_agent_id not in included_agent_ids:
                continue
            outbound_count_by_agent_id[source_agent_id] = outbound_count_by_agent_id.get(source_agent_id, 0) + 1
            inbound_count_by_agent_id[target_agent_id] = inbound_count_by_agent_id.get(target_agent_id, 0) + 1

        start_agents = sorted(
            [
                self._coordination_preview(
                    agent_id=str(agent.get("agent_id") or "").strip(),
                    agent_name=str(agent.get("name") or agent.get("agent_id") or "").strip()
                    or str(agent.get("agent_id") or "").strip(),
                )
                for agent in filtered_agents
                if inbound_count_by_agent_id.get(str(agent.get("agent_id") or "").strip(), 0) == 0
            ],
            key=lambda item: str(item.get("agent_name") or ""),
        )
        terminal_agents = sorted(
            [
                self._coordination_preview(
                    agent_id=str(agent.get("agent_id") or "").strip(),
                    agent_name=str(agent.get("name") or agent.get("agent_id") or "").strip()
                    or str(agent.get("agent_id") or "").strip(),
                )
                for agent in filtered_agents
                if outbound_count_by_agent_id.get(str(agent.get("agent_id") or "").strip(), 0) == 0
            ],
            key=lambda item: str(item.get("agent_name") or ""),
        )

        selection_scope = sorted(included_agent_ids) if selected else None
        topology_summary = self._build_orchestration_topology_summary(
            agents=agents,
            edges=edges,
            summaries=summaries,
            selected_agent_ids=selection_scope,
        )
        capability_coverage_summary = self._build_orchestration_capability_coverage_summary(
            agents=agents,
            summaries=summaries,
            selected_agent_ids=selection_scope,
        )
        collaboration_diagnostics = (
            self._filter_graph_diagnostics_for_agent_scope(
                graph.get("graph_diagnostics") if isinstance(graph.get("graph_diagnostics"), dict) else {},
                selected_agent_ids=selection_scope or [],
            )
            if selection_scope
            else (
                graph.get("graph_diagnostics")
                if isinstance(graph.get("graph_diagnostics"), dict)
                else HarnessStudioGraphDiagnostics().model_dump()
            )
        )
        availability_counts = self._summarize_agent_availability_counts(
            summaries,
            selected_agent_ids=selection_scope,
        )
        unavailable_count = int(availability_counts.get("unavailable_agent_count") or 0)
        limited_availability_count = int(availability_counts.get("limited_agent_count") or 0)

        agent_by_id = {
            str(agent.get("agent_id") or "").strip(): dict(agent)
            for agent in filtered_agents
        }
        policy_repair_agent_count = 0
        for agent_id, agent in agent_by_id.items():
            summary = summary_by_agent_id.get(agent_id, {})
            actionable_tool_ids = self._compute_actionable_tool_policy_suggestion_ids(
                agent,
                [str(item).strip() for item in summary.get("policy_blocked_tool_ids") or [] if str(item).strip()],
            )
            actionable_mcp_server_ids = self._compute_actionable_mcp_policy_suggestion_ids(
                agent,
                [
                    str(item).strip()
                    for item in summary.get("policy_blocked_mcp_server_ids") or []
                    if str(item).strip()
                ],
            )
            restrictive_tool_ids = self._compute_actionable_coordinator_tool_policy_restriction_ids(
                agent,
                summary,
            )
            restrictive_mcp_server_ids = self._compute_actionable_coordinator_mcp_policy_restriction_ids(
                agent,
                summary,
            )
            if (
                actionable_tool_ids
                or actionable_mcp_server_ids
                or restrictive_tool_ids
                or restrictive_mcp_server_ids
            ):
                policy_repair_agent_count += 1

        single_owner_capability_risks = sorted(
            [
                HarnessOrchestrationBriefCapabilityRisk(
                    kind="skill",
                    capability_id=str(entry.get("capability_id") or "").strip(),
                    owner_agents=list(entry.get("owner_agents") or []),
                ).model_dump()
                for entry in capability_coverage_summary.get("single_owner_skills") or []
                if str(entry.get("capability_id") or "").strip()
            ]
            + [
                HarnessOrchestrationBriefCapabilityRisk(
                    kind="tool",
                    capability_id=str(entry.get("capability_id") or "").strip(),
                    owner_agents=list(entry.get("owner_agents") or []),
                ).model_dump()
                for entry in capability_coverage_summary.get("single_owner_tools") or []
                if str(entry.get("capability_id") or "").strip()
            ]
            + [
                HarnessOrchestrationBriefCapabilityRisk(
                    kind="mcp",
                    capability_id=str(entry.get("capability_id") or "").strip(),
                    owner_agents=list(entry.get("owner_agents") or []),
                ).model_dump()
                for entry in capability_coverage_summary.get("single_owner_mcp_servers") or []
                if str(entry.get("capability_id") or "").strip()
            ],
            key=lambda item: (
                {"skill": 0, "tool": 1, "mcp": 2}.get(str(item.get("kind") or ""), 9),
                str(item.get("capability_id") or ""),
            ),
        )
        capability_gap_count = (
            len(capability_coverage_summary.get("missing_skill_ids") or [])
            + len(capability_coverage_summary.get("blocked_tool_ids") or [])
            + len(capability_coverage_summary.get("missing_mcp_server_ids") or [])
        )
        role_profile_alignment = build_role_profile_alignment_diagnostics(
            [
                {
                    "agent_id": agent_id,
                    "agent_name": str(agent.get("name") or agent_id).strip() or agent_id,
                    "loaded_skill_ids": summary.get("loaded_skill_ids"),
                    "enabled_tool_ids": summary.get("enabled_tool_ids"),
                    "provider_limited_tool_ids": summary.get("provider_limited_tool_ids"),
                    "mcp_server_ids": summary.get("mcp_server_ids"),
                    "configured_denied_tool_ids": summary.get("configured_denied_tool_ids"),
                    "configured_denied_mcp_server_ids": summary.get("configured_denied_mcp_server_ids"),
                    "delegation_lane_ids": summary.get("delegation_lane_ids"),
                    "role_profile_suggestion": summary.get("role_profile_suggestion"),
                }
                for agent_id, agent in agent_by_id.items()
                if isinstance(summary_by_agent_id.get(agent_id), dict)
                for summary in [summary_by_agent_id.get(agent_id, {})]
            ]
        )
        role_profile_drift_agent_count = int(role_profile_alignment.get("drift_agent_count") or 0)
        role_profile_overlap_risk_count = int(role_profile_alignment.get("overlap_risk_count") or 0)

        phase_entries: dict[str, list[dict[str, object]]] = {
            "research": [],
            "synthesis": [],
            "implementation": [],
            "verification": [],
        }
        routing_entries: dict[str, list[dict[str, object]]] = {
            "coordinator": [],
            "research": [],
            "implementation": [],
            "verification": [],
            "skill": [],
            "tool": [],
            "mcp": [],
        }
        for agent in filtered_agents:
            agent_id = str(agent.get("agent_id") or "").strip()
            agent_name = str(agent.get("name") or agent_id).strip() or agent_id
            summary = summary_by_agent_id.get(agent_id, {})
            role_hint = self._normalize_skill_key(str(agent.get("role") or ""))
            inbound_count = inbound_count_by_agent_id.get(agent_id, 0)
            outbound_count = outbound_count_by_agent_id.get(agent_id, 0)
            lane_ids = [str(item).strip() for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
            lane_id_set = set(lane_ids)
            lane_count = len(lane_ids)
            loaded_skill_ids = {
                str(item).strip()
                for item in summary.get("loaded_skill_ids") or []
                if str(item).strip()
            }
            strong_collaborator_count = len(
                [
                    item
                    for item in summary.get("recommended_collaborators") or []
                    if isinstance(item, dict)
                    and str(item.get("fit") or "").strip() in {"strong", "good"}
                ]
            )
            enabled_tool_count = len(
                [str(item).strip() for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
            )
            mcp_server_count = len(
                [str(item).strip() for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
            )
            loaded_skill_count = len(
                [str(item).strip() for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
            )
            signals: list[object] = [
                agent.get("name"),
                agent.get("role"),
                agent.get("description"),
                agent.get("system_prompt"),
                summary.get("delegation_focus"),
                summary.get("review_mode"),
                *list(agent.get("skill_intents") or []),
                *lane_ids,
            ]

            research_score = 2 if self._matches_orchestration_keywords(signals, _RESEARCH_PHASE_KEYWORDS) else 0
            synthesis_score = 2 if self._matches_orchestration_keywords(signals, _SYNTHESIS_PHASE_KEYWORDS) else 0
            implementation_score = (
                2 if self._matches_orchestration_keywords(signals, _IMPLEMENTATION_PHASE_KEYWORDS) else 0
            )
            verification_score = (
                2 if self._matches_orchestration_keywords(signals, _VERIFICATION_PHASE_KEYWORDS) else 0
            )
            if role_hint in {"research", "researcher"} or "research" in lane_id_set or "research" in loaded_skill_ids:
                research_score += 1
            if role_hint in {"coordinator", "planner", "lead", "manager"} or "coordination" in lane_id_set:
                synthesis_score += 1
            if role_hint in {"implementation", "builder", "engineer", "developer"} or (
                {"implementation", "execution"} & lane_id_set
            ):
                implementation_score += 1
            if role_hint in {"review", "reviewer", "qa", "critic", "verification"}:
                verification_score += 1

            if bool(agent.get("cluster_auto_research")):
                research_score += 2
            if str(agent.get("node_kind") or "agent").strip() == "cluster":
                synthesis_score += 2
            if outbound_count > 1:
                synthesis_score += 1
            if inbound_count == 0 and outbound_count > 0:
                synthesis_score += 1
            if enabled_tool_count > 0 or bool(agent.get("requires_tool_calling")):
                implementation_score += 1
            if outbound_count == 0 and inbound_count > 0:
                implementation_score += 1
            if bool(agent.get("cluster_auto_review")):
                verification_score += 1
            if str(summary.get("review_mode") or "").strip():
                verification_score += 1

            if verification_score >= max(3, research_score, synthesis_score, implementation_score + 1):
                phase_id = "verification"
            elif synthesis_score >= max(3, research_score, implementation_score, verification_score):
                phase_id = "synthesis"
            elif implementation_score >= max(2, research_score, verification_score):
                phase_id = "implementation"
            elif research_score >= 2:
                phase_id = "research"
            elif bool(agent.get("cluster_auto_research")):
                phase_id = "research"
            elif (
                str(agent.get("node_kind") or "agent").strip() == "cluster"
                or outbound_count > 1
                or (inbound_count == 0 and outbound_count > 0)
            ):
                phase_id = "synthesis"
            elif verification_score > 0 or bool(agent.get("cluster_auto_review")):
                phase_id = "verification"
            elif inbound_count == 0:
                phase_id = "research"
            else:
                phase_id = "implementation"

            base_score_by_phase = {
                "research": research_score,
                "synthesis": synthesis_score,
                "implementation": implementation_score,
                "verification": verification_score,
            }
            preview = {
                "agent_id": agent_id,
                "agent_name": agent_name,
            }
            phase_entries[phase_id].append(
                {
                    **preview,
                    "score": base_score_by_phase[phase_id],
                }
            )

            coordinator_score = (
                synthesis_score * 10
                + outbound_count * 3
                + lane_count * 2
                + strong_collaborator_count
                + (2 if str(agent.get("node_kind") or "agent").strip() == "cluster" else 0)
            )
            research_routing_score = research_score * 10 + (3 if bool(agent.get("cluster_auto_research")) else 0) + (
                1 if inbound_count == 0 else 0
            )
            implementation_routing_score = (
                implementation_score * 10
                + enabled_tool_count * 2
                + mcp_server_count
                + (1 if outbound_count == 0 and inbound_count > 0 else 0)
            )
            verification_routing_score = verification_score * 10 + (
                2 if bool(agent.get("cluster_auto_review")) else 0
            ) + (1 if str(summary.get("review_mode") or "").strip() else 0)
            skill_routing_score = loaded_skill_count * 10 + research_score + synthesis_score
            tool_routing_score = enabled_tool_count * 10 + (
                3 if bool(agent.get("requires_tool_calling")) else 0
            ) + implementation_score
            mcp_routing_score = mcp_server_count * 10 + implementation_score + synthesis_score

            routing_entries["coordinator"].append({**preview, "score": coordinator_score})
            routing_entries["research"].append({**preview, "score": research_routing_score})
            routing_entries["implementation"].append({**preview, "score": implementation_routing_score})
            routing_entries["verification"].append({**preview, "score": verification_routing_score})
            routing_entries["skill"].append({**preview, "score": skill_routing_score})
            routing_entries["tool"].append({**preview, "score": tool_routing_score})
            routing_entries["mcp"].append({**preview, "score": mcp_routing_score})

        def to_phase_summary(
            phase_id: str,
            fallback_agents: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            explicit_entries = sorted(
                phase_entries.get(phase_id, []),
                key=lambda item: (
                    -int(item.get("score") or 0),
                    str(item.get("agent_name") or ""),
                ),
            )
            phase_agents = [
                self._coordination_preview(
                    agent_id=str(item.get("agent_id") or "").strip(),
                    agent_name=str(item.get("agent_name") or item.get("agent_id") or "").strip(),
                )
                for item in explicit_entries
                if str(item.get("agent_id") or "").strip()
            ]
            if not phase_agents:
                phase_agents = list(fallback_agents or [])
            return HarnessOrchestrationPhaseSummary(
                phase_id=phase_id,  # type: ignore[arg-type]
                agent_count=len(phase_agents),
                agents=phase_agents,
            ).model_dump()

        implementation_fallback_agents = (
            list(terminal_agents)
            if terminal_agents
            else [
                self._coordination_preview(
                    agent_id=str(agent.get("agent_id") or "").strip(),
                    agent_name=str(agent.get("name") or agent.get("agent_id") or "").strip()
                    or str(agent.get("agent_id") or "").strip(),
                )
                for agent in filtered_agents[:2]
            ]
        )
        phases = [
            to_phase_summary("research", start_agents),
            to_phase_summary(
                "synthesis",
                [
                    self._coordination_preview(
                        agent_id=str(agent.get("agent_id") or "").strip(),
                        agent_name=str(agent.get("name") or agent.get("agent_id") or "").strip()
                        or str(agent.get("agent_id") or "").strip(),
                    )
                    for agent in filtered_agents
                    if outbound_count_by_agent_id.get(str(agent.get("agent_id") or "").strip(), 0) > 0
                ][:3],
            ),
            to_phase_summary("implementation", implementation_fallback_agents),
            to_phase_summary("verification", terminal_agents),
        ]
        agent_routing = HarnessOrchestrationAgentRoutingSummary(
            coordinator_anchors=self._pick_top_coordination_anchors(routing_entries["coordinator"]),
            research_anchors=self._pick_top_coordination_anchors(routing_entries["research"]),
            implementation_anchors=self._pick_top_coordination_anchors(routing_entries["implementation"]),
            verification_anchors=self._pick_top_coordination_anchors(routing_entries["verification"]),
            skill_capable_anchors=self._pick_top_coordination_anchors(routing_entries["skill"]),
            tool_capable_anchors=self._pick_top_coordination_anchors(routing_entries["tool"]),
            mcp_capable_anchors=self._pick_top_coordination_anchors(routing_entries["mcp"]),
        ).model_dump()

        repair_priorities: list[dict[str, object]] = []
        if unavailable_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="availability",
                    severity="high",
                    count=unavailable_count,
                ).model_dump()
            )
        elif limited_availability_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="availability",
                    severity="medium",
                    count=limited_availability_count,
                ).model_dump()
            )
        if capability_gap_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="capability_gaps",
                    severity="high",
                    count=capability_gap_count,
                ).model_dump()
            )
        if policy_repair_agent_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="policy_repair",
                    severity="medium",
                    count=policy_repair_agent_count,
                ).model_dump()
            )
        if role_profile_drift_agent_count > 0 or role_profile_overlap_risk_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="role_profile_alignment",
                    severity="medium" if role_profile_drift_agent_count > 0 else "low",
                    count=role_profile_drift_agent_count + role_profile_overlap_risk_count,
                ).model_dump()
            )
        weak_edge_count = int(collaboration_diagnostics.get("weak_edge_count") or 0)
        best_next_count = int(collaboration_diagnostics.get("best_next_count") or 0)
        if weak_edge_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="weak_handoffs",
                    severity="medium",
                    count=weak_edge_count,
                ).model_dump()
            )
        if best_next_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="best_next_handoffs",
                    severity="low",
                    count=best_next_count,
                ).model_dump()
            )
        connectivity_count = int(topology_summary.get("isolated_agent_count") or 0) + int(
            topology_summary.get("underconnected_agent_count") or 0
        )
        if connectivity_count > 0:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="connectivity",
                    severity="low",
                    count=connectivity_count,
                ).model_dump()
            )
        if single_owner_capability_risks:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="single_owner_capabilities",
                    severity="low",
                    count=len(single_owner_capability_risks),
                ).model_dump()
            )
        if not review_enabled:
            repair_priorities.append(
                HarnessOrchestrationRepairPriority(
                    priority_id="review_path",
                    severity="low",
                    count=1,
                ).model_dump()
            )
        repair_priorities.sort(
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(str(item.get("severity") or "low"), 9),
                -int(item.get("count") or 0),
            )
        )

        if unavailable_count > 0:
            readiness = "blocked"
        elif (
            weak_edge_count > 0
            or policy_repair_agent_count > 0
            or capability_gap_count > 0
            or role_profile_drift_agent_count > 0
        ):
            readiness = "repair"
        elif (
            role_profile_overlap_risk_count > 0
            or
            int(topology_summary.get("isolated_agent_count") or 0) > 0
            or int(topology_summary.get("underconnected_agent_count") or 0) > 0
            or bool(single_owner_capability_risks)
            or not review_enabled
            or not start_agents
            or not terminal_agents
        ):
            readiness = "watch"
        else:
            readiness = "ready"

        return HarnessOrchestrationSummary(
            total_agent_count=len(filtered_agents),
            execution_step_count=execution_step_count,
            review_enabled=review_enabled,
            readiness=readiness,
            start_agents=start_agents,
            terminal_agents=terminal_agents,
            shared_lane_count=int(topology_summary.get("shared_lane_count") or 0),
            single_owner_capability_count=len(single_owner_capability_risks),
            single_owner_capability_risks=single_owner_capability_risks,
            unavailable_count=unavailable_count,
            limited_availability_count=limited_availability_count,
            policy_repair_agent_count=policy_repair_agent_count,
            role_profile_drift_agent_count=role_profile_drift_agent_count,
            role_profile_overlap_risk_count=role_profile_overlap_risk_count,
            weak_edge_count=weak_edge_count,
            best_next_count=best_next_count,
            capability_gap_count=capability_gap_count,
            isolated_agent_count=int(topology_summary.get("isolated_agent_count") or 0),
            underconnected_agent_count=int(topology_summary.get("underconnected_agent_count") or 0),
            phases=phases,
            repair_priorities=repair_priorities,
            agent_routing=agent_routing,
        ).model_dump()

    def _build_agent_capability_brief(self, summary: dict[str, object]) -> str:
        loaded_skill_ids = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
        missing_skill_ids = [str(item) for item in summary.get("missing_skill_ids") or [] if str(item).strip()]
        suggested_skill_ids = [str(item) for item in summary.get("suggested_skill_ids") or [] if str(item).strip()]
        required_skill_ids = [str(item) for item in summary.get("required_skill_ids") or [] if str(item).strip()]
        missing_required_skill_ids = [
            str(item) for item in summary.get("missing_required_skill_ids") or [] if str(item).strip()
        ]
        required_tool_ids = [str(item) for item in summary.get("required_tool_ids") or [] if str(item).strip()]
        missing_required_tool_ids = [
            str(item) for item in summary.get("missing_required_tool_ids") or [] if str(item).strip()
        ]
        configured_allowed_tool_ids = [
            str(item) for item in summary.get("configured_allowed_tool_ids") or [] if str(item).strip()
        ]
        configured_denied_tool_ids = [
            str(item) for item in summary.get("configured_denied_tool_ids") or [] if str(item).strip()
        ]
        enabled_tool_ids = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
        disabled_tool_ids = [str(item) for item in summary.get("disabled_tool_ids") or [] if str(item).strip()]
        policy_added_tool_ids = [str(item) for item in summary.get("policy_added_tool_ids") or [] if str(item).strip()]
        policy_blocked_tool_ids = [
            str(item) for item in summary.get("policy_blocked_tool_ids") or [] if str(item).strip()
        ]
        unknown_allowed_tool_ids = [
            str(item) for item in summary.get("unknown_allowed_tool_ids") or [] if str(item).strip()
        ]
        requires_tool_calling = bool(summary.get("requires_tool_calling", False))
        provider_limited_tool_ids = [
            str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        ]
        tool_execution_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()
        required_mcp_server_ids = [
            str(item) for item in summary.get("required_mcp_server_ids") or [] if str(item).strip()
        ]
        missing_required_mcp_server_ids = [
            str(item) for item in summary.get("missing_required_mcp_server_ids") or [] if str(item).strip()
        ]
        configured_allowed_mcp_server_ids = [
            str(item) for item in summary.get("configured_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        configured_denied_mcp_server_ids = [
            str(item) for item in summary.get("configured_denied_mcp_server_ids") or [] if str(item).strip()
        ]
        mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
        missing_mcp_server_ids = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
        policy_added_mcp_server_ids = [
            str(item) for item in summary.get("policy_added_mcp_server_ids") or [] if str(item).strip()
        ]
        policy_blocked_mcp_server_ids = [
            str(item) for item in summary.get("policy_blocked_mcp_server_ids") or [] if str(item).strip()
        ]
        unknown_allowed_mcp_server_ids = [
            str(item) for item in summary.get("unknown_allowed_mcp_server_ids") or [] if str(item).strip()
        ]
        delegation_lane_ids = [str(item) for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
        recommended_collaborators = [
            dict(item) for item in summary.get("recommended_collaborators") or [] if isinstance(item, dict)
        ]
        downstream_handoff_scores = [
            dict(item) for item in summary.get("downstream_handoff_scores") or [] if isinstance(item, dict)
        ]
        delegation_focus = str(summary.get("delegation_focus") or "").strip()
        availability_status = str(summary.get("availability_status") or "available").strip() or "available"
        availability_blockers = [str(item) for item in summary.get("availability_blockers") or [] if str(item).strip()]
        availability_warnings = [str(item) for item in summary.get("availability_warnings") or [] if str(item).strip()]
        readiness_status = str(summary.get("readiness_status") or "ready").strip() or "ready"
        readiness_blockers = [str(item) for item in summary.get("readiness_blockers") or [] if str(item).strip()]
        readiness_warnings = [str(item) for item in summary.get("readiness_warnings") or [] if str(item).strip()]
        provider_route = str(summary.get("provider_route") or "project default").strip() or "project default"
        review_mode = str(summary.get("review_mode") or "direct handoff").strip() or "direct handoff"

        parts = [
            f"Availability: {availability_status}",
            f"Readiness: {readiness_status}",
            f"Approved skills: {self._format_preview_list(loaded_skill_ids)}",
            f"Missing approved skills: {self._format_preview_list(missing_skill_ids)}",
            f"Suggested source skills from intents: {self._format_preview_list(suggested_skill_ids)}",
        ]
        if required_skill_ids:
            parts.append(f"Required approved skills: {self._format_preview_list(required_skill_ids)}")
        if missing_required_skill_ids:
            parts.append(
                "Missing required approved skills: "
                f"{self._format_preview_list(missing_required_skill_ids)}"
            )
        if required_tool_ids:
            parts.append(f"Required tools: {self._format_preview_list(required_tool_ids)}")
        if missing_required_tool_ids:
            parts.append(
                "Missing required tools: "
                f"{self._format_preview_list(missing_required_tool_ids)}"
            )
        if required_mcp_server_ids:
            parts.append(f"Required MCP servers: {self._format_preview_list(required_mcp_server_ids)}")
        if missing_required_mcp_server_ids:
            parts.append(
                "Missing required MCP servers: "
                f"{self._format_preview_list(missing_required_mcp_server_ids)}"
            )
        if requires_tool_calling:
            parts.append("Definition requires direct tool-calling support")
        if availability_blockers:
            parts.append(f"Availability blockers: {self._format_preview_list(availability_blockers, limit=3)}")
        if availability_warnings:
            parts.append(f"Availability warnings: {self._format_preview_list(availability_warnings, limit=3)}")
        if readiness_blockers:
            parts.append(f"Readiness blockers: {self._format_preview_list(readiness_blockers, limit=3)}")
        if readiness_warnings:
            parts.append(f"Readiness warnings: {self._format_preview_list(readiness_warnings, limit=3)}")
        if configured_allowed_tool_ids or configured_denied_tool_ids:
            parts.append(
                "Tool policy: "
                f"allow only {self._format_preview_list(configured_allowed_tool_ids)}; "
                f"deny {self._format_preview_list(configured_denied_tool_ids)}"
            )
        if enabled_tool_ids or disabled_tool_ids:
            parts.append(f"Enabled tools: {self._format_preview_list(enabled_tool_ids)}")
            parts.append(f"Disabled tools unless flags change: {self._format_preview_list(disabled_tool_ids)}")
        else:
            parts.append("No direct tool access is currently advertised for this node")
        if policy_added_tool_ids:
            parts.append(f"Tools added by node policy: {self._format_preview_list(policy_added_tool_ids)}")
        if policy_blocked_tool_ids:
            parts.append(f"Tools blocked by node policy: {self._format_preview_list(policy_blocked_tool_ids)}")
        if unknown_allowed_tool_ids:
            parts.append(f"Unknown tool ids in node policy: {self._format_preview_list(unknown_allowed_tool_ids)}")
        if provider_limited_tool_ids:
            parts.append(f"Provider-limited tools: {self._format_preview_list(provider_limited_tool_ids)}")
        support_line = f"Tool calling support: {tool_execution_support}"
        if tool_execution_support_reason:
            support_line += f" ({tool_execution_support_reason})"
        parts.append(support_line)
        if configured_allowed_mcp_server_ids or configured_denied_mcp_server_ids:
            parts.append(
                "MCP policy: "
                f"allow only {self._format_preview_list(configured_allowed_mcp_server_ids)}; "
                f"deny {self._format_preview_list(configured_denied_mcp_server_ids)}"
            )
        if mcp_server_ids:
            parts.append(f"MCP servers: {self._format_preview_list(mcp_server_ids)}")
        else:
            parts.append("MCP servers: none configured in this project")
        if missing_mcp_server_ids:
            parts.append(
                "Relevant MCP servers not enabled in this project: "
                f"{self._format_preview_list(missing_mcp_server_ids)}"
            )
        if policy_added_mcp_server_ids:
            parts.append(f"MCP servers added by node policy: {self._format_preview_list(policy_added_mcp_server_ids)}")
        if policy_blocked_mcp_server_ids:
            parts.append(
                "MCP servers blocked by node policy: "
                f"{self._format_preview_list(policy_blocked_mcp_server_ids)}"
            )
        if unknown_allowed_mcp_server_ids:
            parts.append(
                "Unknown MCP server ids in node policy: "
                f"{self._format_preview_list(unknown_allowed_mcp_server_ids)}"
            )
        if delegation_lane_ids:
            parts.append(f"Delegation lanes: {self._format_preview_list(delegation_lane_ids)}")
        if delegation_focus:
            parts.append(f"Delegation focus: {delegation_focus}")
        recommended_preview = self._build_delegation_partner_preview(recommended_collaborators)
        if recommended_preview:
            parts.append(f"Recommended collaborators: {recommended_preview}")
        weak_downstream = [
            item for item in downstream_handoff_scores if str(item.get("fit") or "weak").strip() == "weak"
        ]
        weak_downstream_preview = self._build_delegation_partner_preview(weak_downstream)
        if weak_downstream_preview:
            parts.append(f"Current downstream handoff warnings: {weak_downstream_preview}")
        parts.append(f"Provider route: {provider_route}")
        parts.append(f"Review path: {review_mode}")
        return ". ".join(parts) + "."

    def _build_agent_execution_contract(self, summary: dict[str, object]) -> dict[str, object]:
        approved_skill_ids = [str(item) for item in summary.get("loaded_skill_ids") or [] if str(item).strip()]
        suggested_skill_ids = [str(item) for item in summary.get("suggested_skill_ids") or [] if str(item).strip()]
        executable_tool_ids = [str(item) for item in summary.get("enabled_tool_ids") or [] if str(item).strip()]
        planning_only_tool_ids = [
            str(item) for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        ]
        disabled_tool_ids = [str(item) for item in summary.get("disabled_tool_ids") or [] if str(item).strip()]
        planning_only_mcp_server_ids = [str(item) for item in summary.get("mcp_server_ids") or [] if str(item).strip()]
        missing_mcp_server_ids = [str(item) for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()]
        tool_execution_support = str(summary.get("tool_execution_support") or "").strip()

        if tool_execution_support == "unsupported" and executable_tool_ids:
            planning_only_tool_ids = self._dedupe_preserve_order([*planning_only_tool_ids, *executable_tool_ids])
            executable_tool_ids = []

        if executable_tool_ids and planning_only_tool_ids:
            tool_access_mode = "mixed"
        elif executable_tool_ids:
            tool_access_mode = "direct_execution"
        elif planning_only_tool_ids:
            tool_access_mode = "planning_only"
        else:
            tool_access_mode = "none"

        return HarnessAgentExecutionContract(
            skill_execution_mode="guidance_only",
            approved_skill_ids=approved_skill_ids,
            suggested_skill_ids=suggested_skill_ids,
            tool_access_mode=tool_access_mode,
            executable_tool_ids=executable_tool_ids,
            planning_only_tool_ids=planning_only_tool_ids,
            disabled_tool_ids=disabled_tool_ids,
            mcp_access_mode="planning_only" if planning_only_mcp_server_ids else "none",
            planning_only_mcp_server_ids=planning_only_mcp_server_ids,
            missing_mcp_server_ids=missing_mcp_server_ids,
        ).model_dump()

    @staticmethod
    def _extract_coordination_preview_list(value: object) -> list[dict[str, object]]:
        previews: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in value or []:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("agent_id") or "").strip()
            agent_name = str(item.get("agent_name") or item.get("agent_id") or "").strip()
            if not agent_id or not agent_name or agent_id in seen:
                continue
            seen.add(agent_id)
            previews.append(
                HarnessCoordinationAgentPreview(
                    agent_id=agent_id,
                    agent_name=agent_name,
                ).model_dump()
            )
        return previews

    @classmethod
    def _extract_coordination_preview_ids(cls, value: object) -> set[str]:
        return {
            str(item.get("agent_id") or "").strip()
            for item in cls._extract_coordination_preview_list(value)
            if str(item.get("agent_id") or "").strip()
        }

    def _build_agent_delegation_contract(
        self,
        *,
        agent: dict[str, object],
        summary: dict[str, object],
        agents_by_id: dict[str, dict[str, object]],
        edges: list[dict[str, object]],
        orchestration_summary: dict[str, object] | None,
    ) -> dict[str, object]:
        agent_id = str(agent.get("agent_id") or "").strip()
        if not agent_id:
            return HarnessAgentDelegationContract().model_dump()

        def append_preview(
            previews: list[dict[str, object]],
            seen: set[str],
            *,
            other_agent_id: str,
            fallback_name: str | None = None,
        ) -> None:
            normalized_agent_id = str(other_agent_id or "").strip()
            if not normalized_agent_id or normalized_agent_id in seen:
                return
            other_agent = agents_by_id.get(normalized_agent_id, {})
            other_agent_name = (
                str(other_agent.get("name") or fallback_name or normalized_agent_id).strip() or normalized_agent_id
            )
            seen.add(normalized_agent_id)
            previews.append(
                self._coordination_preview(
                    agent_id=normalized_agent_id,
                    agent_name=other_agent_name,
                )
            )

        def preview_list_from_targets(targets: list[dict[str, object]]) -> list[dict[str, object]]:
            previews: list[dict[str, object]] = []
            seen: set[str] = set()
            for item in targets:
                if not isinstance(item, dict):
                    continue
                append_preview(
                    previews,
                    seen,
                    other_agent_id=str(item.get("agent_id") or "").strip(),
                    fallback_name=str(item.get("agent_name") or item.get("agent_id") or "").strip() or None,
                )
            return previews

        role_hint = self._normalize_skill_key(str(agent.get("role") or ""))
        node_kind = str(agent.get("node_kind") or "agent").strip() or "agent"
        lane_ids = [str(item).strip() for item in summary.get("delegation_lane_ids") or [] if str(item).strip()]
        lane_id_set = set(lane_ids)
        loaded_skill_ids = {
            str(item).strip()
            for item in summary.get("loaded_skill_ids") or []
            if str(item).strip()
        }
        enabled_tool_ids = {
            str(item).strip()
            for item in summary.get("enabled_tool_ids") or []
            if str(item).strip()
        }
        provider_limited_tool_ids = [
            str(item).strip()
            for item in summary.get("provider_limited_tool_ids") or []
            if str(item).strip()
        ]
        mcp_server_ids = {
            str(item).strip()
            for item in summary.get("mcp_server_ids") or []
            if str(item).strip()
        }
        missing_mcp_server_ids = [
            str(item).strip()
            for item in summary.get("missing_mcp_server_ids") or []
            if str(item).strip()
        ]
        missing_skill_ids = self._dedupe_preserve_order(
            [
                *[str(item).strip() for item in summary.get("missing_skill_ids") or [] if str(item).strip()],
                *[
                    str(item).strip()
                    for item in summary.get("missing_required_skill_ids") or []
                    if str(item).strip()
                ],
            ]
        )
        requires_tool_calling = bool(summary.get("requires_tool_calling"))
        tool_execution_support = str(summary.get("tool_execution_support") or "").strip() or "unknown"
        tool_execution_support_reason = str(summary.get("tool_execution_support_reason") or "").strip()
        delegation_focus = str(summary.get("delegation_focus") or "").strip() or None
        recommended_collaborators = [
            dict(item)
            for item in summary.get("recommended_collaborators") or []
            if isinstance(item, dict)
        ]
        downstream_handoff_scores = [
            dict(item)
            for item in summary.get("downstream_handoff_scores") or []
            if isinstance(item, dict)
        ]

        upstream_agents: list[dict[str, object]] = []
        downstream_agents: list[dict[str, object]] = []
        upstream_seen: set[str] = set()
        downstream_seen: set[str] = set()
        inbound_count = 0
        outbound_count = 0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source_agent_id = str(edge.get("source_agent_id") or "").strip()
            target_agent_id = str(edge.get("target_agent_id") or "").strip()
            if target_agent_id == agent_id and source_agent_id:
                inbound_count += 1
                append_preview(upstream_agents, upstream_seen, other_agent_id=source_agent_id)
            if source_agent_id == agent_id and target_agent_id:
                outbound_count += 1
                append_preview(downstream_agents, downstream_seen, other_agent_id=target_agent_id)

        routing_summary = (
            dict(orchestration_summary.get("agent_routing") or {})
            if isinstance(orchestration_summary, dict) and isinstance(orchestration_summary.get("agent_routing"), dict)
            else {}
        )
        phases = (
            [dict(item) for item in orchestration_summary.get("phases") or [] if isinstance(item, dict)]
            if isinstance(orchestration_summary, dict)
            else []
        )
        phase_members: dict[str, set[str]] = {}
        for phase in phases:
            phase_id = str(phase.get("phase_id") or "").strip()
            if not phase_id:
                continue
            phase_members[phase_id] = self._extract_coordination_preview_ids(phase.get("agents"))

        coordinator_anchor_ids = self._extract_coordination_preview_ids(routing_summary.get("coordinator_anchors"))
        research_anchor_ids = self._extract_coordination_preview_ids(routing_summary.get("research_anchors"))
        implementation_anchor_ids = self._extract_coordination_preview_ids(
            routing_summary.get("implementation_anchors")
        )
        verification_anchor_ids = self._extract_coordination_preview_ids(routing_summary.get("verification_anchors"))
        terminal_agent_ids = (
            self._extract_coordination_preview_ids(orchestration_summary.get("terminal_agents"))
            if isinstance(orchestration_summary, dict)
            else set()
        )
        start_agent_ids = (
            self._extract_coordination_preview_ids(orchestration_summary.get("start_agents"))
            if isinstance(orchestration_summary, dict)
            else set()
        )

        coordinator_score = 0
        if agent_id in coordinator_anchor_ids:
            coordinator_score += 4
        if agent_id in phase_members.get("synthesis", set()):
            coordinator_score += 3
        if node_kind == "cluster":
            coordinator_score += 3
        if role_hint in {"coordinator", "planner", "lead", "manager"}:
            coordinator_score += 3
        if "coordination" in lane_id_set:
            coordinator_score += 2
        if outbound_count > 1:
            coordinator_score += 2
        elif outbound_count > 0 and inbound_count == 0:
            coordinator_score += 1

        research_score = 0
        if agent_id in research_anchor_ids:
            research_score += 4
        if agent_id in phase_members.get("research", set()):
            research_score += 3
        if role_hint in {"research", "researcher"}:
            research_score += 3
        if "research" in lane_id_set:
            research_score += 2
        if "grounding" in lane_id_set:
            research_score += 1
        if {"research", "rag", "ocr"} & loaded_skill_ids:
            research_score += 1
        if {"web_search", "knowledge_retriever", "read_document"} & enabled_tool_ids:
            research_score += 1
        if {"fetch", "browser", "filesystem"} & mcp_server_ids:
            research_score += 1
        if agent_id in start_agent_ids:
            research_score += 1

        implementation_score = 0
        if agent_id in implementation_anchor_ids:
            implementation_score += 4
        if agent_id in phase_members.get("implementation", set()):
            implementation_score += 3
        if role_hint in {"implementation", "builder", "engineer", "developer"}:
            implementation_score += 3
        if {"implementation", "execution"} & lane_id_set:
            implementation_score += 2
        if enabled_tool_ids or provider_limited_tool_ids or requires_tool_calling:
            implementation_score += 1
        if outbound_count == 0 and inbound_count > 0:
            implementation_score += 1

        verification_score = 0
        if agent_id in verification_anchor_ids:
            verification_score += 4
        if agent_id in phase_members.get("verification", set()):
            verification_score += 3
        if role_hint in {"review", "reviewer", "qa", "critic", "verification", "validator"}:
            verification_score += 3
        if bool(agent.get("cluster_auto_review")):
            verification_score += 2
        if outbound_count == 0 and inbound_count > 0 and not enabled_tool_ids:
            verification_score += 1

        generalist_score = 1
        if not lane_ids:
            generalist_score += 1
        if not loaded_skill_ids and not enabled_tool_ids and not mcp_server_ids:
            generalist_score += 1

        role_scores = {
            "coordinator": coordinator_score,
            "research": research_score,
            "implementation": implementation_score,
            "verification": verification_score,
            "generalist": generalist_score,
        }
        role_priority = {
            "coordinator": 0,
            "research": 1,
            "implementation": 2,
            "verification": 3,
            "generalist": 4,
        }
        ranked_role_modes = sorted(
            role_scores.items(),
            key=lambda item: (-int(item[1]), role_priority.get(str(item[0]), 9)),
        )
        primary_role_mode = str(ranked_role_modes[0][0]) if ranked_role_modes else "generalist"
        primary_role_score = int(ranked_role_modes[0][1]) if ranked_role_modes else 0
        supporting_role_modes = [
            str(role_mode)
            for role_mode, score in ranked_role_modes[1:]
            if int(score) >= 4 and int(score) >= max(primary_role_score - 3, 4)
        ]

        preferred_collaborator_candidates = [
            item
            for item in recommended_collaborators
            if str(item.get("fit") or "").strip() in {"strong", "good"}
        ] or recommended_collaborators
        preferred_collaborators = preview_list_from_targets(preferred_collaborator_candidates[:3])
        weak_handoff_targets = preview_list_from_targets(
            [
                item
                for item in downstream_handoff_scores
                if bool(item.get("edge_present")) and str(item.get("fit") or "weak").strip() == "weak"
            ][:3]
        )

        should_coordinate_parallel_work = primary_role_mode == "coordinator" or outbound_count > 1
        should_produce_final_output = not downstream_agents or agent_id in terminal_agent_ids

        if primary_role_mode == "coordinator":
            work_strategy = "synthesize_and_route"
        elif primary_role_mode == "research":
            work_strategy = "self_contained_delivery" if should_produce_final_output else "gather_then_handoff"
        elif primary_role_mode == "implementation":
            work_strategy = "self_contained_delivery" if should_produce_final_output else "implement_then_handoff"
        elif primary_role_mode == "verification":
            work_strategy = "verify_and_close"
        else:
            work_strategy = "self_contained_delivery" if should_produce_final_output else "flexible"

        single_owner_dependencies: list[str] = []
        if isinstance(orchestration_summary, dict):
            for risk in orchestration_summary.get("single_owner_capability_risks") or []:
                if not isinstance(risk, dict):
                    continue
                owners = self._extract_coordination_preview_ids(risk.get("owner_agents"))
                if agent_id not in owners:
                    continue
                capability_kind = str(risk.get("kind") or "capability").strip() or "capability"
                capability_id = str(risk.get("capability_id") or "").strip()
                if capability_id:
                    single_owner_dependencies.append(f"{capability_kind}:{capability_id}")

        watchouts: list[str] = []
        if weak_handoff_targets:
            weak_target_names = [
                str(item.get("agent_name") or item.get("agent_id") or "").strip()
                for item in weak_handoff_targets
                if str(item.get("agent_name") or item.get("agent_id") or "").strip()
            ]
            watchouts.append(
                "Current downstream handoffs are weak for: "
                + self._format_preview_list(weak_target_names, limit=3)
            )
        if missing_skill_ids:
            watchouts.append(
                "Missing approved skills may block this role: "
                + self._format_preview_list(missing_skill_ids, limit=3)
            )
        if requires_tool_calling and tool_execution_support != "supported":
            reason = tool_execution_support_reason or "provider route does not expose direct tool calling"
            watchouts.append(
                "This node expects direct tool execution, but the runtime contract is degraded: "
                + reason
            )
        elif provider_limited_tool_ids:
            watchouts.append(
                "Some tools are planning-only under the current provider route: "
                + self._format_preview_list(provider_limited_tool_ids, limit=3)
            )
        if missing_mcp_server_ids:
            watchouts.append(
                "Relevant MCP inventory is missing: "
                + self._format_preview_list(missing_mcp_server_ids, limit=3)
            )
        if single_owner_dependencies:
            watchouts.append(
                "This node is the single owner for capabilities that need backup coverage: "
                + self._format_preview_list(single_owner_dependencies, limit=3)
            )
        if primary_role_mode == "coordinator" and not downstream_agents:
            watchouts.append(
                "No explicit downstream edge is configured, so coordination and final delivery currently collapse into one node"
            )

        return HarnessAgentDelegationContract(
            primary_role_mode=primary_role_mode,  # type: ignore[arg-type]
            supporting_role_modes=self._dedupe_preserve_order(supporting_role_modes)[:3],
            work_strategy=work_strategy,  # type: ignore[arg-type]
            should_coordinate_parallel_work=should_coordinate_parallel_work,
            should_produce_final_output=should_produce_final_output,
            primary_focus=delegation_focus,
            upstream_agents=upstream_agents,
            downstream_agents=downstream_agents,
            preferred_collaborators=preferred_collaborators,
            weak_handoff_targets=weak_handoff_targets,
            watchouts=self._dedupe_preserve_order(watchouts)[:4],
        ).model_dump()

    def _resolve_agent_role_profile_id(
        self,
        *,
        agent: dict[str, object],
        summary: dict[str, object],
        delegation_contract: dict[str, object] | None,
    ) -> str:
        primary_role_mode = self._normalize_skill_key(
            str((delegation_contract or {}).get("primary_role_mode") or "")
        )
        if primary_role_mode in {"coordinator", "research", "implementation", "verification", "generalist"}:
            return primary_role_mode

        role_hint = self._normalize_skill_key(str(agent.get("role") or summary.get("role") or ""))
        if role_hint in {"coordinator", "planner", "lead", "manager"}:
            return "coordinator"
        if role_hint in {"research", "researcher"}:
            return "research"
        if role_hint in {"implementation", "builder", "engineer", "developer"}:
            return "implementation"
        if role_hint in {"verification", "review", "reviewer", "qa", "critic", "validator"}:
            return "verification"
        return "generalist"

    def _build_agent_role_profile_suggestion(
        self,
        *,
        agent: dict[str, object],
        summary: dict[str, object],
        graph: dict[str, object],
        delegation_contract: dict[str, object] | None,
        approved_skill_ids: set[str],
        tool_catalog_by_id: dict[str, dict[str, object]],
        mcp_server_catalog_by_id: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        profile_id = self._resolve_agent_role_profile_id(
            agent=agent,
            summary=summary,
            delegation_contract=delegation_contract,
        )
        has_knowledge_bases = bool(graph.get("knowledge_base_ids") or [])
        available_tool_catalog_ids = {
            tool_id
            for tool_id, item in tool_catalog_by_id.items()
            if str(item.get("status") or "enabled").strip() == "enabled"
        }
        available_mcp_server_ids = {
            server_id
            for server_id, item in mcp_server_catalog_by_id.items()
            if str(item.get("status") or "enabled").strip() == "enabled"
        }

        suggested_skill_ids: list[str] = []
        suggested_tool_ids: list[str] = []
        suggested_mcp_server_ids: list[str] = []
        restrictive_tool_ids: list[str] = []
        restrictive_mcp_server_ids: list[str] = []

        if profile_id == "coordinator":
            suggested_skill_ids = ["memory"]
            suggested_tool_ids = ["get_current_time"]
            restrictive_tool_ids = self._compute_actionable_coordinator_tool_policy_restriction_ids(
                agent,
                summary,
            )
            restrictive_mcp_server_ids = self._compute_actionable_coordinator_mcp_policy_restriction_ids(
                agent,
                summary,
            )
        elif profile_id == "research":
            suggested_skill_ids = ["research"]
            suggested_tool_ids = ["web_search", "read_document", "get_current_time"]
            suggested_mcp_server_ids = ["fetch", "browser"]
            if has_knowledge_bases:
                suggested_skill_ids.append("rag")
                suggested_tool_ids.append("knowledge_retriever")
                suggested_mcp_server_ids.append("filesystem")
        elif profile_id == "implementation":
            suggested_skill_ids = ["tools"]
            suggested_tool_ids = ["write_file", "python_executor", "read_document", "get_current_time"]
            suggested_mcp_server_ids = ["filesystem", "github"]
        elif profile_id == "verification":
            suggested_skill_ids = ["tools"]
            suggested_tool_ids = ["read_document", "get_current_time"]
            suggested_mcp_server_ids = ["filesystem", "github"]
        else:
            suggested_skill_ids = ["profile"]
            suggested_tool_ids = ["get_current_time"]

        normalized_suggested_skill_ids = self._normalize_identifier_list(suggested_skill_ids)
        normalized_suggested_tool_ids = [
            tool_id
            for tool_id in self._normalize_identifier_list(suggested_tool_ids)
            if tool_id in available_tool_catalog_ids
        ]
        normalized_suggested_mcp_server_ids = [
            server_id
            for server_id in self._normalize_identifier_list(suggested_mcp_server_ids)
            if server_id in available_mcp_server_ids
        ]

        available_skill_ids = [
            skill_id for skill_id in normalized_suggested_skill_ids if skill_id in approved_skill_ids
        ]
        missing_skill_ids = [
            skill_id for skill_id in normalized_suggested_skill_ids if skill_id not in approved_skill_ids
        ]

        return HarnessAgentRoleProfileSuggestion(
            profile_id=profile_id,  # type: ignore[arg-type]
            suggested_skill_ids=normalized_suggested_skill_ids,
            available_skill_ids=available_skill_ids,
            missing_skill_ids=missing_skill_ids,
            suggested_tool_ids=normalized_suggested_tool_ids,
            suggested_mcp_server_ids=normalized_suggested_mcp_server_ids,
            restrictive_tool_ids=self._normalize_identifier_list(restrictive_tool_ids),
            restrictive_mcp_server_ids=self._normalize_identifier_list(restrictive_mcp_server_ids),
        ).model_dump()

    def _build_agent_delegation_focus(self, summary: dict[str, object]) -> str:
        loaded_skill_ids = {str(item).strip() for item in summary.get("loaded_skill_ids") or [] if str(item).strip()}
        enabled_tool_ids = {str(item).strip() for item in summary.get("enabled_tool_ids") or [] if str(item).strip()}
        provider_limited_tool_ids = {
            str(item).strip() for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        }
        mcp_server_ids = {str(item).strip() for item in summary.get("mcp_server_ids") or [] if str(item).strip()}
        missing_mcp_server_ids = {
            str(item).strip() for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()
        }
        tool_execution_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"

        lanes: list[str] = []
        if (
            "research" in loaded_skill_ids
            or "web_search" in enabled_tool_ids
            or {"fetch", "browser"} & mcp_server_ids
        ):
            lanes.append("external research, source comparison, and evidence gathering")
        if (
            "rag" in loaded_skill_ids
            or "ocr" in loaded_skill_ids
            or {"knowledge_retriever", "read_document"} & enabled_tool_ids
            or "filesystem" in mcp_server_ids
        ):
            lanes.append("project grounding, document synthesis, and file-backed analysis")
        if "memory" in loaded_skill_ids or "profile" in loaded_skill_ids:
            lanes.append("consistency with prior user and project decisions")
        if "github" in mcp_server_ids:
            lanes.append("repository and issue workflows through GitHub MCP")
        elif "github" in missing_mcp_server_ids:
            lanes.append("repository workflows once GitHub MCP is enabled")
        if {"write_file", "python_executor"} & enabled_tool_ids:
            lanes.append("tool-assisted implementation and execution")
        elif "tools" in loaded_skill_ids:
            lanes.append("implementation planning and structured execution")
        if tool_execution_support == "unsupported" and provider_limited_tool_ids and not enabled_tool_ids:
            lanes.append("reasoning-only synthesis until live tool execution is available")
        if not lanes:
            lanes.append("reasoning, synthesis, and clean downstream handoffs")

        deduped = list(dict.fromkeys(lanes))
        return "; ".join(deduped[:3]).strip()

    def _build_agent_delegation_lane_ids(
        self,
        *,
        agent: dict[str, object],
        summary: dict[str, object],
    ) -> list[str]:
        loaded_skill_ids = {str(item).strip() for item in summary.get("loaded_skill_ids") or [] if str(item).strip()}
        enabled_tool_ids = {str(item).strip() for item in summary.get("enabled_tool_ids") or [] if str(item).strip()}
        provider_limited_tool_ids = {
            str(item).strip() for item in summary.get("provider_limited_tool_ids") or [] if str(item).strip()
        }
        mcp_server_ids = {str(item).strip() for item in summary.get("mcp_server_ids") or [] if str(item).strip()}
        missing_mcp_server_ids = {
            str(item).strip() for item in summary.get("missing_mcp_server_ids") or [] if str(item).strip()
        }
        role_hint = self._normalize_skill_key(str(agent.get("role") or ""))
        node_kind = str(agent.get("node_kind") or "agent").strip()
        tool_execution_support = str(summary.get("tool_execution_support") or "unknown").strip() or "unknown"

        lanes: list[str] = []
        if node_kind == "cluster" or role_hint in {"coordinator", "planner", "lead", "manager"}:
            lanes.append("coordination")
        if (
            "research" in loaded_skill_ids
            or "web_search" in enabled_tool_ids
            or {"fetch", "browser"} & mcp_server_ids
        ):
            lanes.append("research")
        if (
            "rag" in loaded_skill_ids
            or "ocr" in loaded_skill_ids
            or {"knowledge_retriever", "read_document"} & enabled_tool_ids
            or "filesystem" in mcp_server_ids
        ):
            lanes.append("grounding")
        if "memory" in loaded_skill_ids or "profile" in loaded_skill_ids:
            lanes.append("memory")
        if "github" in mcp_server_ids or "github" in missing_mcp_server_ids:
            lanes.append("repository")
        if {"write_file", "python_executor"} & enabled_tool_ids:
            lanes.append("implementation")
        elif "tools" in loaded_skill_ids:
            lanes.append("execution")
        if tool_execution_support == "unsupported" and provider_limited_tool_ids and not enabled_tool_ids:
            lanes.append("reasoning_only")
        if not lanes:
            lanes.append("generalist")
        return self._dedupe_preserve_order(lanes)

    @staticmethod
    def _delegation_fit_bucket(score: int) -> str:
        if score >= 60:
            return "strong"
        if score >= 40:
            return "good"
        return "weak"

    def _score_delegation_target(
        self,
        *,
        source_agent: dict[str, object],
        source_summary: dict[str, object],
        target_agent: dict[str, object],
        target_summary: dict[str, object],
        interaction: str | None = None,
        edge_present: bool = False,
    ) -> dict[str, object]:
        source_lane_ids = [
            str(item).strip() for item in source_summary.get("delegation_lane_ids") or [] if str(item).strip()
        ]
        target_lane_ids = [
            str(item).strip() for item in target_summary.get("delegation_lane_ids") or [] if str(item).strip()
        ]
        source_lane_set = set(source_lane_ids)
        target_lane_set = set(target_lane_ids)
        complementary_lane_ids = [
            lane_id for lane_id in target_lane_ids if lane_id not in source_lane_set and lane_id != "generalist"
        ]
        overlap_lane_ids = [
            lane_id for lane_id in target_lane_ids if lane_id in source_lane_set and lane_id != "generalist"
        ]
        source_loaded_skill_ids = [
            str(item).strip() for item in source_summary.get("loaded_skill_ids") or [] if str(item).strip()
        ]
        source_loaded_skill_id_set = set(source_loaded_skill_ids)
        target_loaded_skill_ids = [
            str(item).strip() for item in target_summary.get("loaded_skill_ids") or [] if str(item).strip()
        ]
        new_skill_ids = [skill_id for skill_id in target_loaded_skill_ids if skill_id not in source_loaded_skill_id_set]
        source_enabled_tool_ids = [
            str(item).strip() for item in source_summary.get("enabled_tool_ids") or [] if str(item).strip()
        ]
        source_enabled_tool_id_set = set(source_enabled_tool_ids)
        target_enabled_tool_ids = [
            str(item).strip() for item in target_summary.get("enabled_tool_ids") or [] if str(item).strip()
        ]
        target_enabled_tool_id_set = set(target_enabled_tool_ids)
        source_mcp_server_ids = [
            str(item).strip() for item in source_summary.get("mcp_server_ids") or [] if str(item).strip()
        ]
        source_mcp_server_id_set = set(source_mcp_server_ids)
        target_mcp_server_ids = [
            str(item).strip() for item in target_summary.get("mcp_server_ids") or [] if str(item).strip()
        ]
        source_missing_mcp_server_ids = {
            str(item).strip() for item in source_summary.get("missing_mcp_server_ids") or [] if str(item).strip()
        }
        new_tool_ids = [tool_id for tool_id in target_enabled_tool_ids if tool_id not in source_enabled_tool_id_set]
        new_mcp_server_ids = [server_id for server_id in target_mcp_server_ids if server_id not in source_mcp_server_id_set]
        gap_cover_mcp_server_ids = [
            server_id for server_id in target_mcp_server_ids if server_id in source_missing_mcp_server_ids
        ]
        source_profile_id = self._resolve_agent_role_profile_id(
            agent=source_agent,
            summary=source_summary,
            delegation_contract=None,
        )
        target_profile_id = self._resolve_agent_role_profile_id(
            agent=target_agent,
            summary=target_summary,
            delegation_contract=None,
        )
        same_role_profile = source_profile_id == target_profile_id
        same_role_profile_overlap_risk = bool(
            same_role_profile
            and overlap_lane_ids
            and not new_tool_ids
            and not new_mcp_server_ids
            and not gap_cover_mcp_server_ids
            and len(complementary_lane_ids) <= 1
        )

        score = 0
        if complementary_lane_ids:
            score += 32 + min(18, 9 * len(complementary_lane_ids))
        if overlap_lane_ids:
            score += min(16, 8 * len(overlap_lane_ids))
        if new_tool_ids:
            score += min(18, 6 * len(new_tool_ids))
        if new_mcp_server_ids:
            score += min(12, 4 * len(new_mcp_server_ids))
        if gap_cover_mcp_server_ids:
            score += min(12, 6 * len(gap_cover_mcp_server_ids))
        if (
            str(source_summary.get("tool_execution_support") or "unknown").strip() == "unsupported"
            and str(target_summary.get("tool_execution_support") or "unknown").strip() == "supported"
            and target_enabled_tool_ids
        ):
            score += 10
        if same_role_profile:
            score -= 6
        if same_role_profile_overlap_risk:
            score -= 18
        if same_role_profile_overlap_risk and target_profile_id == "coordinator":
            score -= 8
        if edge_present:
            score += 5
        if not complementary_lane_ids and not new_tool_ids and not new_mcp_server_ids and not overlap_lane_ids:
            score = max(score - 10, 0)

        fit = self._delegation_fit_bucket(score)
        rationale_parts: list[str] = []
        if complementary_lane_ids:
            rationale_parts.append(
                "adds "
                + ", ".join(self._humanize_identifier(lane_id) for lane_id in complementary_lane_ids[:3])
                + " lane coverage"
            )
        elif new_skill_ids:
            rationale_parts.append(
                "adds "
                + ", ".join(self._humanize_identifier(skill_id) for skill_id in new_skill_ids[:3])
                + " skill coverage"
            )
        if new_tool_ids:
            rationale_parts.append(
                "brings "
                + ", ".join(self._humanize_identifier(tool_id) for tool_id in new_tool_ids[:3])
                + " tool access"
            )
        if gap_cover_mcp_server_ids:
            rationale_parts.append(
                "covers missing MCP like "
                + ", ".join(self._humanize_identifier(server_id) for server_id in gap_cover_mcp_server_ids[:3])
            )
        if same_role_profile_overlap_risk and overlap_lane_ids:
            rationale_parts.append(
                "shares the same "
                + self._humanize_identifier(target_profile_id)
                + " profile and overlaps on "
                + ", ".join(self._humanize_identifier(lane_id) for lane_id in overlap_lane_ids[:3])
                + " lanes"
            )
        elif same_role_profile:
            rationale_parts.append(
                "stays in the same " + self._humanize_identifier(target_profile_id) + " profile lane"
            )
        if not rationale_parts and overlap_lane_ids:
            rationale_parts.append(
                "shares "
                + ", ".join(self._humanize_identifier(lane_id) for lane_id in overlap_lane_ids[:3])
                + " lanes for reinforcement"
            )
        if not rationale_parts:
            rationale_parts.append("mostly overlaps the current node and adds limited new capacity")

        return {
            "agent_id": str(target_agent.get("agent_id") or "").strip(),
            "agent_name": str(target_agent.get("name") or target_agent.get("agent_id") or "agent").strip(),
            "score": max(0, min(score, 100)),
            "fit": fit,
            "rationale": "; ".join(rationale_parts),
            "new_skill_ids": new_skill_ids,
            "overlap_lane_ids": overlap_lane_ids,
            "complementary_lane_ids": complementary_lane_ids,
            "new_tool_ids": new_tool_ids,
            "new_mcp_server_ids": new_mcp_server_ids,
            "gap_cover_mcp_server_ids": gap_cover_mcp_server_ids,
            "source_profile_id": source_profile_id,
            "target_profile_id": target_profile_id,
            "same_role_profile": same_role_profile,
            "same_role_profile_overlap_risk": same_role_profile_overlap_risk,
            "edge_present": edge_present,
            "interaction": interaction or None,
        }

    def _build_delegation_partner_preview(self, recommendations: list[dict[str, object]]) -> str | None:
        parts: list[str] = []
        for recommendation in recommendations[:3]:
            agent_name = str(recommendation.get("agent_name") or recommendation.get("agent_id") or "").strip()
            fit = str(recommendation.get("fit") or "weak").strip() or "weak"
            rationale = str(recommendation.get("rationale") or "").strip()
            if not agent_name:
                continue
            segment = f"{agent_name} ({fit})"
            if rationale:
                segment += f": {rationale}"
            parts.append(segment)
        if not parts:
            return None
        return "; ".join(parts)

    def _build_graph_diagnostics(
        self,
        *,
        summaries: list[dict[str, object]],
        recommendations_by_source: dict[str, list[dict[str, object]]],
        agent_by_id: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        weak_downstream_edges: list[dict[str, object]] = []
        best_next_handoffs: list[dict[str, object]] = []

        for summary in summaries:
            source_agent_id = str(summary.get("agent_id") or "").strip()
            source_agent_name = (
                str((agent_by_id.get(source_agent_id) or {}).get("name") or source_agent_id).strip() or source_agent_id
            )
            if not source_agent_id:
                continue
            source_lane_ids = [
                str(item).strip() for item in summary.get("delegation_lane_ids") or [] if str(item).strip()
            ]
            delegation_focus = str(summary.get("delegation_focus") or "").strip() or None
            recommendations = [
                dict(item)
                for item in recommendations_by_source.get(source_agent_id, [])
                if isinstance(item, dict)
            ]
            non_edge_candidates = [
                item
                for item in recommendations
                if not bool(item.get("edge_present")) and str(item.get("fit") or "weak").strip() in {"strong", "good"}
            ]
            if non_edge_candidates:
                best_next_handoffs.append(
                    HarnessDelegationOpportunity(
                        source_agent_id=source_agent_id,
                        source_agent_name=source_agent_name,
                        source_lane_ids=source_lane_ids,
                        delegation_focus=delegation_focus,
                        target=non_edge_candidates[0],
                    ).model_dump()
                )

            for downstream in summary.get("downstream_handoff_scores") or []:
                if not isinstance(downstream, dict):
                    continue
                if not bool(downstream.get("edge_present")):
                    continue
                if str(downstream.get("fit") or "weak").strip() != "weak":
                    continue
                weak_downstream_edges.append(
                    HarnessDelegationOpportunity(
                        source_agent_id=source_agent_id,
                        source_agent_name=source_agent_name,
                        source_lane_ids=source_lane_ids,
                        delegation_focus=delegation_focus,
                        target=downstream,
                        suggested_replacements=[
                            item
                            for item in non_edge_candidates
                            if str(item.get("agent_id") or "").strip()
                            != str(downstream.get("agent_id") or "").strip()
                        ][:2],
                    ).model_dump()
                )

        weak_downstream_edges.sort(
            key=lambda item: (
                int(((item.get("target") or {}) if isinstance(item, dict) else {}).get("score") or 0),
                str(item.get("source_agent_name") or ""),
                str((((item.get("target") or {}) if isinstance(item, dict) else {})).get("agent_name") or ""),
            )
        )
        best_next_handoffs.sort(
            key=lambda item: (
                0
                if str((((item.get("target") or {}) if isinstance(item, dict) else {})).get("fit") or "weak").strip()
                == "strong"
                else 1,
                -int((((item.get("target") or {}) if isinstance(item, dict) else {})).get("score") or 0),
                str(item.get("source_agent_name") or ""),
            )
        )

        return HarnessStudioGraphDiagnostics(
            weak_downstream_edges=weak_downstream_edges,
            best_next_handoffs=best_next_handoffs,
            weak_edge_count=len(weak_downstream_edges),
            best_next_count=len(best_next_handoffs),
        ).model_dump()

    def _filter_graph_diagnostics_for_agent_scope(
        self,
        diagnostics: dict[str, object] | None,
        *,
        selected_agent_ids: list[str],
    ) -> dict[str, object]:
        if not isinstance(diagnostics, dict):
            return HarnessStudioGraphDiagnostics().model_dump()
        selected = {
            str(agent_id).strip()
            for agent_id in selected_agent_ids
            if str(agent_id).strip()
        }
        if not selected:
            return HarnessStudioGraphDiagnostics().model_dump()

        weak_downstream_edges: list[dict[str, object]] = []
        for item in diagnostics.get("weak_downstream_edges") or []:
            if not isinstance(item, dict):
                continue
            source_agent_id = str(item.get("source_agent_id") or "").strip()
            target = dict(item.get("target") or {}) if isinstance(item.get("target"), dict) else {}
            target_agent_id = str(target.get("agent_id") or "").strip()
            if source_agent_id not in selected or target_agent_id not in selected:
                continue
            filtered_item = dict(item)
            filtered_item["suggested_replacements"] = [
                dict(replacement)
                for replacement in item.get("suggested_replacements") or []
                if isinstance(replacement, dict)
                and str(replacement.get("agent_id") or "").strip() in selected
            ]
            weak_downstream_edges.append(filtered_item)

        best_next_handoffs: list[dict[str, object]] = []
        for item in diagnostics.get("best_next_handoffs") or []:
            if not isinstance(item, dict):
                continue
            source_agent_id = str(item.get("source_agent_id") or "").strip()
            target = dict(item.get("target") or {}) if isinstance(item.get("target"), dict) else {}
            target_agent_id = str(target.get("agent_id") or "").strip()
            if source_agent_id not in selected or target_agent_id not in selected:
                continue
            best_next_handoffs.append(dict(item))

        return HarnessStudioGraphDiagnostics(
            weak_downstream_edges=weak_downstream_edges,
            best_next_handoffs=best_next_handoffs,
            weak_edge_count=len(weak_downstream_edges),
            best_next_count=len(best_next_handoffs),
        ).model_dump()

    def _sync_capability_catalogs(self, graph: dict[str, object]) -> dict[str, object]:
        graph["tool_catalog"] = self._discover_tool_catalog()
        graph["mcp_server_catalog"] = self._discover_mcp_server_catalog()
        loaded_skill_ids = {
            self._normalize_skill_key(str(item.get("skill_id") or ""))
            for item in graph.get("skill_pool") or []
            if isinstance(item, dict)
        }
        tool_catalog = graph.get("tool_catalog") or []
        mcp_server_catalog = graph.get("mcp_server_catalog") or []
        skill_catalog_by_id = {
            self._normalize_skill_key(str(item.get("skill_id") or "")): dict(item)
            for item in graph.get("skill_catalog") or []
            if isinstance(item, dict) and self._normalize_skill_key(str(item.get("skill_id") or ""))
        }
        mcp_server_catalog_by_id = {
            self._normalize_skill_key(str(item.get("server_id") or "")): dict(item)
            for item in mcp_server_catalog
            if isinstance(item, dict) and self._normalize_skill_key(str(item.get("server_id") or ""))
        }
        tool_catalog_by_id = {
            self._normalize_skill_key(str(item.get("tool_id") or "")): dict(item)
            for item in tool_catalog
            if isinstance(item, dict) and self._normalize_skill_key(str(item.get("tool_id") or ""))
        }
        mcp_status_by_id = {
            server_id: str(item.get("status") or "enabled")
            for server_id, item in mcp_server_catalog_by_id.items()
        }
        mcp_alias_to_server_id = self._build_mcp_server_alias_lookup()
        summaries: list[dict[str, object]] = []
        for agent in graph.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            agent_id = str(agent.get("agent_id") or "").strip()
            if not agent_id:
                continue
            skill_ids = [
                self._normalize_skill_key(str(skill_id))
                for skill_id in agent.get("skill_ids") or []
                if self._normalize_skill_key(str(skill_id))
            ]
            loaded_for_agent = sorted(skill_id for skill_id in skill_ids if skill_id in loaded_skill_ids)
            missing_for_agent = sorted(skill_id for skill_id in skill_ids if skill_id not in loaded_skill_ids)
            loaded_skill_hints = [
                str(skill_catalog_by_id.get(skill_id, {}).get("prompt_hint") or "").strip()
                for skill_id in loaded_for_agent
                if str(skill_catalog_by_id.get(skill_id, {}).get("prompt_hint") or "").strip()
            ]
            intent_matches = [
                skill_id
                for skill_id in self._match_skills_for_intents(
                    [str(intent) for intent in agent.get("skill_intents") or []],
                    graph,
                )
                if skill_id not in loaded_for_agent and skill_id not in missing_for_agent
            ]
            relevant_skill_ids = list(dict.fromkeys([*loaded_for_agent, *missing_for_agent, *intent_matches]))
            resolved_tool_access = self._resolve_agent_tool_access(
                agent=agent,
                relevant_skill_ids=relevant_skill_ids,
                skill_catalog_by_id=skill_catalog_by_id,
                tool_catalog=tool_catalog,
            )
            resolved_mcp_access = self._resolve_agent_mcp_access(
                agent=agent,
                relevant_skill_ids=relevant_skill_ids,
                skill_catalog_by_id=skill_catalog_by_id,
                mcp_server_catalog=mcp_server_catalog,
                mcp_alias_to_server_id=mcp_alias_to_server_id,
            )
            tool_execution_support, tool_execution_support_reason = self._infer_tool_execution_support(agent, graph)
            enabled_tool_ids = list(resolved_tool_access["enabled_tool_ids"])
            provider_limited_tool_ids = enabled_tool_ids if tool_execution_support == "unsupported" else []
            if tool_execution_support == "unsupported":
                enabled_tool_ids = []

            summary = HarnessAgentCapabilitySummary(
                agent_id=agent_id,
                loaded_skill_ids=loaded_for_agent,
                missing_skill_ids=missing_for_agent,
                missing_skill_details=[],
                suggested_skill_ids=intent_matches,
                loaded_skill_hints=loaded_skill_hints,
                required_skill_ids=[],
                missing_required_skill_ids=[],
                required_tool_ids=[],
                missing_required_tool_ids=[],
                configured_allowed_tool_ids=resolved_tool_access["configured_allowed_tool_ids"],
                configured_denied_tool_ids=resolved_tool_access["configured_denied_tool_ids"],
                enabled_tool_ids=enabled_tool_ids,
                disabled_tool_ids=resolved_tool_access["disabled_tool_ids"],
                policy_added_tool_ids=resolved_tool_access["policy_added_tool_ids"],
                policy_blocked_tool_ids=resolved_tool_access["policy_blocked_tool_ids"],
                unknown_allowed_tool_ids=resolved_tool_access["unknown_allowed_tool_ids"],
                requires_tool_calling=bool(agent.get("requires_tool_calling", False)),
                provider_limited_tool_ids=provider_limited_tool_ids,
                tool_execution_support=tool_execution_support,
                tool_execution_support_reason=tool_execution_support_reason or None,
                required_mcp_server_ids=[],
                missing_required_mcp_server_ids=[],
                configured_allowed_mcp_server_ids=resolved_mcp_access["configured_allowed_mcp_server_ids"],
                configured_denied_mcp_server_ids=resolved_mcp_access["configured_denied_mcp_server_ids"],
                mcp_server_ids=resolved_mcp_access["mcp_server_ids"],
                missing_mcp_server_ids=resolved_mcp_access["missing_mcp_server_ids"],
                missing_mcp_server_details=[],
                policy_added_mcp_server_ids=resolved_mcp_access["policy_added_mcp_server_ids"],
                policy_blocked_mcp_server_ids=resolved_mcp_access["policy_blocked_mcp_server_ids"],
                unknown_allowed_mcp_server_ids=resolved_mcp_access["unknown_allowed_mcp_server_ids"],
                delegation_lane_ids=[],
                recommended_collaborators=[],
                downstream_handoff_scores=[],
                delegation_focus=None,
                availability_status="available",
                availability_blockers=[],
                availability_warnings=[],
                readiness_status="ready",
                readiness_blockers=[],
                readiness_warnings=[],
                provider_route=self._build_provider_route(agent, graph),
                review_mode=self._build_review_mode(agent, graph),
            ).model_dump()
            summary["missing_skill_details"] = self._build_missing_skill_details(
                missing_for_agent,
                skill_catalog_by_id=skill_catalog_by_id,
            )
            summary["missing_mcp_server_details"] = self._build_missing_mcp_server_details(
                resolved_mcp_access["missing_mcp_server_ids"],
                mcp_server_catalog_by_id=mcp_server_catalog_by_id,
            )
            summary["delegation_lane_ids"] = self._build_agent_delegation_lane_ids(agent=agent, summary=summary)
            summary["delegation_focus"] = self._build_agent_delegation_focus(summary)
            summary.update(
                self._build_agent_availability(
                    agent,
                    summary=summary,
                    approved_skill_ids=loaded_skill_ids,
                    tool_catalog_by_id=tool_catalog_by_id,
                    mcp_status_by_id=mcp_status_by_id,
                    mcp_alias_to_server_id=mcp_alias_to_server_id,
                )
            )
            summary.update(self._build_agent_readiness(summary))
            summary["execution_contract"] = self._build_agent_execution_contract(summary)
            summaries.append(summary)

        summary_by_id = {
            str(summary.get("agent_id") or "").strip(): summary
            for summary in summaries
            if str(summary.get("agent_id") or "").strip()
        }
        agent_by_id = {
            str(agent.get("agent_id") or "").strip(): dict(agent)
            for agent in graph.get("agents") or []
            if isinstance(agent, dict) and str(agent.get("agent_id") or "").strip()
        }
        outgoing_edges_by_source: dict[str, list[dict[str, object]]] = {}
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source_agent_id = str(edge.get("source_agent_id") or "").strip()
            target_agent_id = str(edge.get("target_agent_id") or "").strip()
            if not source_agent_id or not target_agent_id:
                continue
            outgoing_edges_by_source.setdefault(source_agent_id, []).append(dict(edge))

        recommendations_by_source: dict[str, list[dict[str, object]]] = {}
        for summary in summaries:
            agent_id = str(summary.get("agent_id") or "").strip()
            source_agent = agent_by_id.get(agent_id, {})
            source_summary = summary_by_id.get(agent_id, {})
            existing_edges = {
                str(edge.get("target_agent_id") or "").strip(): dict(edge)
                for edge in outgoing_edges_by_source.get(agent_id, [])
                if str(edge.get("target_agent_id") or "").strip()
            }
            recommendations: list[dict[str, object]] = []
            for target_agent_id, target_summary in summary_by_id.items():
                if target_agent_id == agent_id:
                    continue
                target_agent = agent_by_id.get(target_agent_id, {})
                recommendation = self._score_delegation_target(
                    source_agent=source_agent,
                    source_summary=source_summary,
                    target_agent=target_agent,
                    target_summary=target_summary,
                    interaction=str(existing_edges.get(target_agent_id, {}).get("interaction") or "").strip() or None,
                    edge_present=target_agent_id in existing_edges,
                )
                if int(recommendation.get("score") or 0) <= 0 and not bool(recommendation.get("edge_present")):
                    continue
                recommendations.append(recommendation)

            recommendations.sort(
                key=lambda item: (
                    -int(item.get("score") or 0),
                    0 if bool(item.get("edge_present")) else 1,
                    str(item.get("agent_name") or item.get("agent_id") or ""),
                )
            )
            recommendations_by_source[agent_id] = recommendations
            summary["recommended_collaborators"] = recommendations[:3]
            summary["downstream_handoff_scores"] = [
                recommendation for recommendation in recommendations if bool(recommendation.get("edge_present"))
            ]
            summary["capability_brief"] = self._build_agent_capability_brief(summary)

        graph["agent_capability_summaries"] = summaries
        graph["graph_diagnostics"] = self._build_graph_diagnostics(
            summaries=summaries,
            recommendations_by_source=recommendations_by_source,
            agent_by_id=agent_by_id,
        )
        orchestration_summary = self._build_orchestration_summary(graph)
        graph["orchestration_summary"] = orchestration_summary
        for summary in summaries:
            agent_id = str(summary.get("agent_id") or "").strip()
            if not agent_id:
                continue
            delegation_contract = self._build_agent_delegation_contract(
                agent=agent_by_id.get(agent_id, {}),
                summary=summary,
                agents_by_id=agent_by_id,
                edges=[dict(edge) for edge in graph.get("edges") or [] if isinstance(edge, dict)],
                orchestration_summary=orchestration_summary,
            )
            summary["delegation_contract"] = delegation_contract
            summary["role_profile_suggestion"] = self._build_agent_role_profile_suggestion(
                agent=agent_by_id.get(agent_id, {}),
                summary=summary,
                graph=graph,
                delegation_contract=delegation_contract,
                approved_skill_ids=loaded_skill_ids,
                tool_catalog_by_id=tool_catalog_by_id,
                mcp_server_catalog_by_id=mcp_server_catalog_by_id,
            )
        graph["orchestration_summary"] = self._build_orchestration_summary(graph)
        return graph

    def _normalize_execution_checklist(self, items: list[object] | None) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            active_form = str(item.get("active_form") or "").strip() or None
            normalized.append(
                HarnessExecutionChecklistItem(
                    item_id=str(item.get("item_id") or f"check_{index + 1}"),
                    content=content,
                    status=str(item.get("status") or "pending"),
                    active_form=active_form,
                ).model_dump()
            )
        return normalized

    def _default_graph(self) -> dict[str, object]:
        graph = HarnessStudioGraph(
            agents=[
                HarnessCanvasAgent(
                    agent_id="agent_planner",
                    name="Planner",
                    role="coordinator",
                    description="Breaks the task into delegable work and routes to specialists.",
                    system_prompt="Plan the collaboration loop, decide handoffs, and keep the swarm aligned.",
                    model="gpt-5.2",
                    max_iterations=4,
                    position=HarnessCanvasPosition(x=96, y=96),
                    skill_intents=["research", "memory"],
                ),
                HarnessCanvasAgent(
                    agent_id="agent_researcher",
                    name="Researcher",
                    role="research",
                    description="Looks up context, documents, and supporting evidence.",
                    system_prompt="Gather evidence, summarize findings, and hand off clean context.",
                    model="gpt-5.2",
                    max_iterations=3,
                    position=HarnessCanvasPosition(x=420, y=84),
                    skill_intents=["research", "rag"],
                ),
                HarnessCanvasAgent(
                    agent_id="agent_builder",
                    name="Builder",
                    role="implementation",
                    description="Turns the plan into code or structured output.",
                    system_prompt="Implement the agreed solution and keep outputs production-oriented.",
                    model="gpt-5.2",
                    max_iterations=4,
                    position=HarnessCanvasPosition(x=420, y=280),
                    skill_intents=["tools", "memory"],
                ),
            ],
            edges=[
                HarnessCanvasEdge(
                    edge_id="edge_planner_researcher",
                    source_agent_id="agent_planner",
                    target_agent_id="agent_researcher",
                    interaction="delegate",
                ),
                HarnessCanvasEdge(
                    edge_id="edge_researcher_builder",
                    source_agent_id="agent_researcher",
                    target_agent_id="agent_builder",
                    interaction="handoff",
                ),
                HarnessCanvasEdge(
                    edge_id="edge_planner_builder",
                    source_agent_id="agent_planner",
                    target_agent_id="agent_builder",
                    interaction="review",
                ),
            ],
            review_agent=HarnessReviewAgentSettings(),
        )
        payload = graph.model_dump()
        payload["skill_catalog"] = self._discover_skill_catalog()
        payload = self._sync_skill_pool_with_catalog(payload)
        return self._sync_capability_catalogs(payload)

    def _normalize_graph(self, graph_json: dict[str, object] | None) -> dict[str, object]:
        base = dict(graph_json or {})
        payload = HarnessStudioGraph.model_validate(
            {
                "version": base.get("version", 1),
                "agents": base.get("agents", []),
                "edges": base.get("edges", []),
                "graph_diagnostics": base.get("graph_diagnostics", {}),
                "knowledge_base_ids": base.get("knowledge_base_ids", []),
                "execution_checklist": self._normalize_execution_checklist(base.get("execution_checklist")),
                "skill_pool": base.get("skill_pool", []),
                "pending_skill_requests": base.get("pending_skill_requests", []),
                "tool_catalog": [],
                "mcp_server_catalog": [],
                "agent_capability_summaries": [],
                "orchestration_summary": base.get("orchestration_summary", {}),
                "review_agent": base.get("review_agent", {}),
                "canvas": base.get("canvas", {}),
                "skill_catalog": [],
                "provider_config": base.get("provider_config", {}),
            }
        ).model_dump()
        payload["skill_catalog"] = self._discover_skill_catalog()
        payload = self._sync_skill_pool_with_catalog(payload)
        return self._sync_capability_catalogs(payload)

    def _hydrate_project(self, project: dict[str, object]) -> dict[str, object]:
        hydrated = dict(project)
        hydrated["graph_json"] = self._normalize_graph(project.get("graph_json") if isinstance(project, dict) else None)
        graph = hydrated["graph_json"]
        agents = graph.get("agents") or []
        edges = graph.get("edges") or []
        hydrated["agent_count"] = len(agents)
        hydrated["edge_count"] = len(edges)
        hydrated["checklist_count"] = len(graph.get("execution_checklist") or [])
        hydrated["open_checklist_count"] = len(
            [
                item
                for item in graph.get("execution_checklist") or []
                if isinstance(item, dict) and str(item.get("status") or "pending") != "completed"
            ]
        )
        hydrated["loaded_skill_count"] = len(graph.get("skill_pool") or [])
        hydrated["pending_skill_request_count"] = len(
            [item for item in graph.get("pending_skill_requests") or [] if item.get("status") == "pending"]
        )
        return hydrated

    def _ensure_access(self, *, project_id: str, user_id: str) -> dict[str, object]:
        project = self.project_store.get_project(project_id)
        if project is None:
            raise HarnessStudioProjectNotFoundError("Studio project not found")
        if str(project.get("user_id") or "") != user_id:
            raise HarnessStudioProjectAccessError("Not authorized to access this studio project")
        return project

    def _catalog_lookup(self, graph_json: dict[str, object]) -> dict[str, dict[str, object]]:
        catalog = graph_json.get("skill_catalog") or []
        lookup: dict[str, dict[str, object]] = {}
        for item in catalog:
            if not isinstance(item, dict):
                continue
            skill_id = self._normalize_skill_key(str(item.get("skill_id") or item.get("title") or ""))
            title_key = self._normalize_skill_key(str(item.get("title") or ""))
            if skill_id:
                lookup[skill_id] = dict(item)
            if title_key:
                lookup[title_key] = dict(item)
        return lookup

    def _match_catalog_item(self, requested_skill: str, graph_json: dict[str, object]) -> dict[str, object]:
        lookup = self._catalog_lookup(graph_json)
        key = self._normalize_skill_key(requested_skill)
        if key in lookup:
            return lookup[key]
        for value in lookup.values():
            skill_id = self._normalize_skill_key(str(value.get("skill_id") or ""))
            title = self._normalize_skill_key(str(value.get("title") or ""))
            if key and (key in skill_id or key in title):
                return dict(value)
        return {
            "skill_id": key or requested_skill.strip(),
            "title": requested_skill.strip() or "Unknown skill",
            "description": "No matching local skill source was discovered automatically.",
            "source": "unresolved",
            "status": "missing",
        }

    def list_projects(self, *, user_id: str) -> list[dict[str, object]]:
        projects = self.project_store.list_projects(user_id=user_id)
        if not projects:
            return [self.get_current_project(user_id=user_id)]
        return [self._hydrate_project(project) for project in projects]

    def get_current_project(self, *, user_id: str) -> dict[str, object]:
        existing = self.project_store.get_latest_project_for_user(user_id)
        if existing is not None:
            return self._hydrate_project(existing)
        created = self.project_store.create_project(
            project_id=f"hp_{uuid.uuid4()}",
            user_id=user_id,
            name="Default agent studio",
            description="Canvas workspace for composing and running multi-agent collaborations.",
            graph_json=self._default_graph(),
        )
        return self._hydrate_project(created)

    def get_project(self, *, project_id: str, user_id: str) -> dict[str, object]:
        return self._hydrate_project(self._ensure_access(project_id=project_id, user_id=user_id))

    def create_project(self, *, user_id: str, name: str, description: str | None = None) -> dict[str, object]:
        created = self.project_store.create_project(
            project_id=f"hp_{uuid.uuid4()}",
            user_id=user_id,
            name=name.strip(),
            description=(description or "").strip() or None,
            graph_json=self._default_graph(),
        )
        return self._hydrate_project(created)

    def update_project(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        graph_json: dict[str, object] | None = None,
    ) -> dict[str, object]:
        project = self._ensure_access(project_id=project_id, user_id=user_id)
        changes: dict[str, object] = {}
        if name is not None:
            changes["name"] = name.strip()
        if description is not None:
            changes["description"] = description.strip() or None
        if graph_json is not None:
            normalized = self._normalize_graph(graph_json)
            changes["graph_json"] = normalized
        updated = self.project_store.update_project(str(project.get("project_id") or project_id), **changes)
        if updated is None:
            raise HarnessStudioProjectNotFoundError("Studio project not found")
        return self._hydrate_project(updated)

    def request_skills(
        self,
        *,
        project_id: str,
        user_id: str,
        agent_id: str,
        requested_skills: list[str],
    ) -> dict[str, object]:
        project = self._ensure_access(project_id=project_id, user_id=user_id)
        graph = self._normalize_graph(project.get("graph_json") if isinstance(project, dict) else None)
        agents = graph.get("agents") or []
        if not any(str(agent.get("agent_id") or "") == agent_id for agent in agents if isinstance(agent, dict)):
            raise HarnessStudioAgentNotFoundError("Agent not found in studio project")

        graph = self._sync_skill_pool_with_catalog(graph)
        loaded_skill_ids = {
            self._normalize_skill_key(str(item.get("skill_id") or ""))
            for item in graph.get("skill_pool") or []
            if isinstance(item, dict)
        }
        pending_requests = list(graph.get("pending_skill_requests") or [])
        pending_skill_ids = {
            self._normalize_skill_key(str(item.get("skill_id") or ""))
            for item in pending_requests
            if isinstance(item, dict) and str(item.get("status") or "pending") == "pending"
        }
        available: list[str] = []
        created_requests: list[dict[str, object]] = []
        now = int(time.time())
        for raw_skill in requested_skills:
            normalized_request = self._normalize_skill_key(raw_skill)
            if not normalized_request:
                continue
            if normalized_request in loaded_skill_ids:
                available.append(normalized_request)
                continue
            if normalized_request in pending_skill_ids:
                continue
            catalog_item = self._match_catalog_item(raw_skill, graph)
            created_request = HarnessSkillRequest(
                request_id=f"hsr_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                skill_id=str(catalog_item.get("skill_id") or normalized_request),
                title=str(catalog_item.get("title") or raw_skill.strip() or normalized_request),
                source=str(catalog_item.get("source") or "unresolved"),
                reason=(
                    "Skill was not found in the local catalog."
                    if str(catalog_item.get("status") or "") == "missing"
                    else None
                ),
                discovered_at=now,
            ).model_dump()
            pending_requests.append(created_request)
            pending_skill_ids.add(normalized_request)
            created_requests.append(created_request)

        graph["pending_skill_requests"] = pending_requests
        updated = self.project_store.update_project(str(project.get("project_id") or project_id), graph_json=graph)
        if updated is None:
            raise HarnessStudioProjectNotFoundError("Studio project not found")
        hydrated = self._hydrate_project(updated)
        hydrated["skill_request_result"] = {
            "available_skill_ids": available,
            "created_requests": created_requests,
        }
        return hydrated

    def resolve_skill_request(
        self,
        *,
        project_id: str,
        user_id: str,
        request_id: str,
        approved: bool,
    ) -> dict[str, object]:
        project = self._ensure_access(project_id=project_id, user_id=user_id)
        graph = self._normalize_graph(project.get("graph_json") if isinstance(project, dict) else None)
        pending_requests = list(graph.get("pending_skill_requests") or [])
        skill_pool = list(graph.get("skill_pool") or [])
        catalog_lookup = self._catalog_lookup(graph)
        now = int(time.time())

        updated_request: dict[str, object] | None = None
        for index, item in enumerate(pending_requests):
            if not isinstance(item, dict) or str(item.get("request_id") or "") != request_id:
                continue
            resolved = dict(item)
            resolved["status"] = "approved" if approved else "rejected"
            resolved["resolved_at"] = now
            pending_requests[index] = resolved
            updated_request = resolved
            if approved:
                skill_key = self._normalize_skill_key(str(resolved.get("skill_id") or ""))
                if skill_key and not any(
                    self._normalize_skill_key(str(pool_item.get("skill_id") or "")) == skill_key
                    for pool_item in skill_pool
                    if isinstance(pool_item, dict)
                ):
                    catalog_item = catalog_lookup.get(skill_key) or {
                        "skill_id": resolved.get("skill_id"),
                        "title": resolved.get("title"),
                        "description": None,
                        "source": resolved.get("source") or "approved_request",
                    }
                    skill_pool.append(
                        HarnessSkillPoolItem(
                            skill_id=str(catalog_item.get("skill_id") or skill_key),
                            title=str(catalog_item.get("title") or resolved.get("title") or skill_key),
                            description=str(catalog_item.get("description") or "") or None,
                            source=str(catalog_item.get("source") or resolved.get("source") or "approved_request"),
                            approved_at=now,
                        ).model_dump()
                    )
            break

        if updated_request is None:
            raise HarnessStudioProjectNotFoundError("Skill request not found")

        graph["pending_skill_requests"] = pending_requests
        graph["skill_pool"] = skill_pool
        updated = self.project_store.update_project(str(project.get("project_id") or project_id), graph_json=graph)
        if updated is None:
            raise HarnessStudioProjectNotFoundError("Studio project not found")
        hydrated = self._hydrate_project(updated)
        hydrated["resolved_skill_request"] = updated_request
        return hydrated

    def create_orchestration_run(
        self,
        *,
        project_id: str,
        user_id: str,
        run_scope: str,
        agent_ids: list[str],
        loop_count: int,
        task: str = "",
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        project = self._ensure_access(project_id=project_id, user_id=user_id)
        hydrated = self._hydrate_project(project)
        graph = hydrated["graph_json"]
        agents = graph.get("agents") or []
        if run_scope == "selected":
            selected_ids = [agent_id for agent_id in agent_ids if any(str(agent.get("agent_id") or "") == agent_id for agent in agents)]
        else:
            selected_ids = [str(agent.get("agent_id") or "") for agent in agents if isinstance(agent, dict)]
        execution_checklist = self._normalize_execution_checklist(graph.get("execution_checklist"))
        open_checklist = [
            item
            for item in execution_checklist
            if str(item.get("status") or "pending") != "completed"
        ]
        graph_diagnostics = graph.get("graph_diagnostics") if isinstance(graph.get("graph_diagnostics"), dict) else {}
        agent_capability_summaries = (
            graph.get("agent_capability_summaries")
            if isinstance(graph.get("agent_capability_summaries"), list)
            else []
        )
        weak_edge_count = int(graph_diagnostics.get("weak_edge_count") or 0)
        best_next_count = int(graph_diagnostics.get("best_next_count") or 0)
        graph_readiness_counts = self._summarize_agent_readiness_counts(agent_capability_summaries)
        graph_availability_counts = self._summarize_agent_availability_counts(agent_capability_summaries)
        graph_execution_contract_counts = self._summarize_agent_execution_contract_counts(
            agent_capability_summaries
        )
        graph_delegation_contract_counts = self._summarize_agent_delegation_contract_counts(
            agent_capability_summaries
        )
        scoped_readiness_counts = self._summarize_agent_readiness_counts(
            agent_capability_summaries,
            selected_agent_ids=selected_ids if run_scope == "selected" else None,
        )
        scoped_availability_counts = self._summarize_agent_availability_counts(
            agent_capability_summaries,
            selected_agent_ids=selected_ids if run_scope == "selected" else None,
        )
        scoped_execution_contract_counts = self._summarize_agent_execution_contract_counts(
            agent_capability_summaries,
            selected_agent_ids=selected_ids if run_scope == "selected" else None,
        )
        scoped_delegation_contract_counts = self._summarize_agent_delegation_contract_counts(
            agent_capability_summaries,
            selected_agent_ids=selected_ids if run_scope == "selected" else None,
        )
        scoped_graph_diagnostics = (
            self._filter_graph_diagnostics_for_agent_scope(
                graph_diagnostics,
                selected_agent_ids=selected_ids,
            )
            if run_scope == "selected"
            else graph_diagnostics
        )
        scoped_weak_edge_count = int(scoped_graph_diagnostics.get("weak_edge_count") or 0)
        scoped_best_next_count = int(scoped_graph_diagnostics.get("best_next_count") or 0)
        run_graph = dict(graph)
        if run_scope == "selected":
            run_graph["selected_scope_orchestration_summary"] = self._build_orchestration_summary(
                graph,
                selected_agent_ids=selected_ids,
            )

        return self.run_service.create_run(
            user_id=user_id,
            task_type=HarnessTaskType.AGENT_ORCHESTRATION.value,
            input_json={
                "project_id": project_id,
                "project_name": hydrated.get("name"),
                "run_scope": run_scope,
                "selected_agent_ids": selected_ids,
                "loop_count": loop_count,
                "task": task or hydrated.get("name", ""),
                "timeout_seconds": timeout_seconds,
                "knowledge_base_ids": list(graph.get("knowledge_base_ids") or []),
                "task_checklist": execution_checklist,
                "graph": run_graph,
            },
            session_id=None,
            metadata_json={
                "source": "harness_studio",
                "project_id": project_id,
                "project_name": hydrated.get("name"),
                "selected_agent_ids": selected_ids,
                "loop_count": loop_count,
                "knowledge_base_ids": list(graph.get("knowledge_base_ids") or []),
                "checklist_count": len(execution_checklist),
                "open_checklist_count": len(open_checklist),
                "review_agent_enabled": bool((graph.get("review_agent") or {}).get("enabled", True)),
                "pending_skill_request_count": hydrated.get("pending_skill_request_count", 0),
                "graph_weak_edge_count": weak_edge_count,
                "graph_best_next_count": best_next_count,
                "graph_total_agent_count": graph_readiness_counts["total_agent_count"],
                "graph_ready_agent_count": graph_readiness_counts["ready_agent_count"],
                "graph_limited_agent_count": graph_readiness_counts["limited_agent_count"],
                "graph_blocked_agent_count": graph_readiness_counts["blocked_agent_count"],
                "graph_available_agent_count": graph_availability_counts["available_agent_count"],
                "graph_availability_limited_agent_count": graph_availability_counts["limited_agent_count"],
                "graph_unavailable_agent_count": graph_availability_counts["unavailable_agent_count"],
                "graph_direct_execution_agent_count": graph_execution_contract_counts["direct_execution_agent_count"],
                "graph_planning_only_tool_agent_count": graph_execution_contract_counts["planning_only_tool_agent_count"],
                "graph_planning_only_mcp_agent_count": graph_execution_contract_counts["planning_only_mcp_agent_count"],
                "graph_coordinator_agent_count": graph_delegation_contract_counts["coordinator_agent_count"],
                "graph_parallel_coordinator_agent_count": graph_delegation_contract_counts["parallel_coordinator_agent_count"],
                "graph_final_output_agent_count": graph_delegation_contract_counts["final_output_agent_count"],
                "graph_verification_agent_count": graph_delegation_contract_counts["verification_agent_count"],
                "handoff_diagnostic_scope": "selected_agents" if run_scope == "selected" else "all_agents",
                "handoff_scope_weak_edge_count": scoped_weak_edge_count,
                "handoff_scope_best_next_count": scoped_best_next_count,
                "scope_total_agent_count": scoped_readiness_counts["total_agent_count"],
                "scope_ready_agent_count": scoped_readiness_counts["ready_agent_count"],
                "scope_limited_agent_count": scoped_readiness_counts["limited_agent_count"],
                "scope_blocked_agent_count": scoped_readiness_counts["blocked_agent_count"],
                "scope_available_agent_count": scoped_availability_counts["available_agent_count"],
                "scope_availability_limited_agent_count": scoped_availability_counts["limited_agent_count"],
                "scope_unavailable_agent_count": scoped_availability_counts["unavailable_agent_count"],
                "scope_direct_execution_agent_count": scoped_execution_contract_counts["direct_execution_agent_count"],
                "scope_planning_only_tool_agent_count": scoped_execution_contract_counts["planning_only_tool_agent_count"],
                "scope_planning_only_mcp_agent_count": scoped_execution_contract_counts["planning_only_mcp_agent_count"],
                "scope_coordinator_agent_count": scoped_delegation_contract_counts["coordinator_agent_count"],
                "scope_parallel_coordinator_agent_count": scoped_delegation_contract_counts["parallel_coordinator_agent_count"],
                "scope_final_output_agent_count": scoped_delegation_contract_counts["final_output_agent_count"],
                "scope_verification_agent_count": scoped_delegation_contract_counts["verification_agent_count"],
            },
        )


def build_studio_service() -> HarnessStudioService:
    return HarnessStudioService(project_store=HarnessAgentProjectStore())
