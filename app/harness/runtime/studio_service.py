from __future__ import annotations

import time
import uuid
from pathlib import Path

from app.harness.contracts.run import HarnessTaskType
from app.harness.contracts.studio import (
    HarnessCanvasAgent,
    HarnessCanvasEdge,
    HarnessCanvasPosition,
    HarnessReviewAgentSettings,
    HarnessSkillCatalogItem,
    HarnessSkillPoolItem,
    HarnessSkillRequest,
    HarnessStudioGraph,
)
from app.harness.persistence.stores import HarnessAgentProjectStore
from app.harness.runtime.run_service import HarnessRunService, build_run_service


class HarnessStudioProjectNotFoundError(ValueError):
    pass


class HarnessStudioProjectAccessError(ValueError):
    pass


class HarnessStudioAgentNotFoundError(ValueError):
    pass


class HarnessStudioService:
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

    def _discover_skill_catalog(self) -> list[dict[str, object]]:
        if not self.skills_root.exists():
            return []
        catalog: list[dict[str, object]] = []
        for child in sorted(self.skills_root.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            title = child.name.replace("_", " ").strip().title()
            description = f"Local skill package from app/skills/{child.name}"
            catalog.append(
                HarnessSkillCatalogItem(
                    skill_id=self._normalize_skill_key(child.name),
                    title=title,
                    description=description,
                    source=f"app/skills/{child.name}",
                ).model_dump()
            )
        return catalog

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
        return payload

    def _normalize_graph(self, graph_json: dict[str, object] | None) -> dict[str, object]:
        base = dict(graph_json or {})
        payload = HarnessStudioGraph.model_validate(
            {
                "version": base.get("version", 1),
                "agents": base.get("agents", []),
                "edges": base.get("edges", []),
                "skill_pool": base.get("skill_pool", []),
                "pending_skill_requests": base.get("pending_skill_requests", []),
                "review_agent": base.get("review_agent", {}),
                "canvas": base.get("canvas", {}),
                "skill_catalog": [],
            }
        ).model_dump()
        payload["skill_catalog"] = self._discover_skill_catalog()
        return payload

    def _hydrate_project(self, project: dict[str, object]) -> dict[str, object]:
        hydrated = dict(project)
        hydrated["graph_json"] = self._normalize_graph(project.get("graph_json") if isinstance(project, dict) else None)
        graph = hydrated["graph_json"]
        agents = graph.get("agents") or []
        edges = graph.get("edges") or []
        hydrated["agent_count"] = len(agents)
        hydrated["edge_count"] = len(edges)
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

        loaded_skill_ids = {
            self._normalize_skill_key(str(item.get("skill_id") or ""))
            for item in graph.get("skill_pool") or []
            if isinstance(item, dict)
        }
        pending_requests = list(graph.get("pending_skill_requests") or [])
        pending_keys = {
            self._normalize_skill_key(str(item.get("skill_id") or ""))
            for item in pending_requests
            if isinstance(item, dict) and str(item.get("status") or "") == "pending"
        }

        now = int(time.time())
        available: list[str] = []
        created_requests: list[dict[str, object]] = []
        for raw_skill in requested_skills:
            normalized_request = self._normalize_skill_key(raw_skill)
            if not normalized_request:
                continue
            if normalized_request in loaded_skill_ids:
                available.append(normalized_request)
                continue
            if normalized_request in pending_keys:
                continue
            catalog_item = self._match_catalog_item(raw_skill, graph)
            request_payload = HarnessSkillRequest(
                request_id=f"hsr_{uuid.uuid4()}",
                agent_id=agent_id,
                skill_id=str(catalog_item.get("skill_id") or normalized_request),
                title=str(catalog_item.get("title") or raw_skill),
                source=str(catalog_item.get("source") or "unresolved"),
                status="pending",
                reason=f"{agent_id} requested skill '{raw_skill.strip()}' from discovered source.",
                discovered_at=now,
            ).model_dump()
            pending_requests.append(request_payload)
            pending_keys.add(normalized_request)
            created_requests.append(request_payload)

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
                "graph": graph,
            },
            session_id=None,
            metadata_json={
                "source": "harness_studio",
                "project_name": hydrated.get("name"),
                "selected_agent_ids": selected_ids,
                "loop_count": loop_count,
                "review_agent_enabled": bool((graph.get("review_agent") or {}).get("enabled", True)),
                "pending_skill_request_count": hydrated.get("pending_skill_request_count", 0),
            },
        )


def build_studio_service() -> HarnessStudioService:
    return HarnessStudioService(project_store=HarnessAgentProjectStore())
