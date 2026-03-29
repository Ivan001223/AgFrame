from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HarnessCanvasPosition(BaseModel):
    x: float = 0
    y: float = 0


class HarnessCanvasViewport(BaseModel):
    x: float = 0
    y: float = 0
    zoom: float = 1


class HarnessClusterMember(BaseModel):
    member_id: str
    name: str
    role: str = "specialist"
    system_prompt: str = ""
    model: str = "gpt-5.2"
    preferred_provider_id: str | None = None
    fallback_provider_id: str | None = None
    temperature: float = 0.2
    timeout_seconds: int | None = None


class HarnessReviewAgentSettings(BaseModel):
    enabled: bool = True
    hidden: bool = True
    name: str = "Compliance reviewer"
    model: str = "gpt-5.1-codex-mini"
    preferred_provider_id: str | None = None
    fallback_provider_id: str | None = None
    system_prompt: str = (
        "Review every agent action for policy compliance, tool safety, and workflow hygiene. "
        "This is a lightweight placeholder reviewer until a stronger policy runtime is added."
    )


class HarnessCanvasAgent(BaseModel):
    agent_id: str
    name: str
    node_kind: Literal["agent", "cluster"] = "agent"
    cluster_strategy: Literal["brainstorm", "custom"] | None = None
    role: str = "specialist"
    description: str | None = None
    system_prompt: str = ""
    model: str = "gpt-5.2"
    preferred_provider_id: str | None = None
    fallback_provider_id: str | None = None
    temperature: float = 0.2
    max_iterations: int = 3
    timeout_seconds: int | None = None
    position: HarnessCanvasPosition = Field(default_factory=HarnessCanvasPosition)
    skill_ids: list[str] = Field(default_factory=list)
    skill_intents: list[str] = Field(default_factory=list)
    cluster_members: list[HarnessClusterMember] = Field(default_factory=list)
    brainstorm_rounds: int = Field(default=3, ge=1, le=5)
    cluster_auto_research: bool = False
    cluster_auto_review: bool = True


class HarnessCanvasEdge(BaseModel):
    edge_id: str
    source_agent_id: str
    target_agent_id: str
    interaction: str = "handoff"
    condition: str | None = None


class HarnessSkillCatalogItem(BaseModel):
    skill_id: str
    title: str
    description: str | None = None
    source: str
    status: str = "available"


class HarnessSkillPoolItem(BaseModel):
    skill_id: str
    title: str
    description: str | None = None
    source: str
    status: str = "loaded"
    approved_at: int | None = None


class HarnessSkillRequest(BaseModel):
    request_id: str
    agent_id: str
    skill_id: str
    title: str
    source: str
    status: str = "pending"
    reason: str | None = None
    discovered_at: int
    resolved_at: int | None = None


class HarnessStudioProviderConfig(BaseModel):
    preferred_provider_id: str | None = None
    fallback_provider_id: str | None = None


class HarnessStudioGraph(BaseModel):
    version: int = 1
    agents: list[HarnessCanvasAgent] = Field(default_factory=list)
    edges: list[HarnessCanvasEdge] = Field(default_factory=list)
    skill_pool: list[HarnessSkillPoolItem] = Field(default_factory=list)
    pending_skill_requests: list[HarnessSkillRequest] = Field(default_factory=list)
    skill_catalog: list[HarnessSkillCatalogItem] = Field(default_factory=list)
    review_agent: HarnessReviewAgentSettings = Field(default_factory=HarnessReviewAgentSettings)
    canvas: HarnessCanvasViewport = Field(default_factory=HarnessCanvasViewport)
    provider_config: HarnessStudioProviderConfig = Field(default_factory=HarnessStudioProviderConfig)


class HarnessStudioProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)


class HarnessStudioProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    graph_json: HarnessStudioGraph | None = None


class HarnessStudioSkillRequestCreate(BaseModel):
    agent_id: str
    requested_skills: list[str] = Field(default_factory=list)


class HarnessStudioSkillDecision(BaseModel):
    approved: bool


class HarnessStudioRunCreate(BaseModel):
    run_scope: Literal["all", "selected"] = "all"
    agent_ids: list[str] = Field(default_factory=list)
    loop_count: int = Field(default=1, ge=1, le=10)
    task: str = Field(default="", max_length=2000)
    timeout_seconds: int | None = Field(default=None, ge=5, le=600)
