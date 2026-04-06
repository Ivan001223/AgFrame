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
        "Review every agent output for policy compliance, unsafe instructions, secret leakage, unsupported claims, "
        "and workflow hygiene. Return PASS only when it is safe for downstream use, otherwise return BLOCK with a concise reason."
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
    required_skill_ids: list[str] = Field(default_factory=list)
    required_tool_ids: list[str] = Field(default_factory=list)
    allowed_tool_ids: list[str] = Field(default_factory=list)
    denied_tool_ids: list[str] = Field(default_factory=list)
    requires_tool_calling: bool = False
    required_mcp_server_ids: list[str] = Field(default_factory=list)
    allowed_mcp_server_ids: list[str] = Field(default_factory=list)
    denied_mcp_server_ids: list[str] = Field(default_factory=list)
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
    prompt_hint: str | None = None
    suggested_tool_ids: list[str] = Field(default_factory=list)
    suggested_mcp_server_ids: list[str] = Field(default_factory=list)


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


class HarnessToolCatalogItem(BaseModel):
    tool_id: str
    title: str
    description: str | None = None
    status: Literal["enabled", "disabled"] = "enabled"
    requires_flag: str | None = None


class HarnessMcpServerCatalogItem(BaseModel):
    server_id: str
    title: str
    description: str | None = None
    status: Literal["enabled", "disabled"] = "enabled"
    command_preview: str | None = None


class HarnessDelegationTargetFit(BaseModel):
    agent_id: str
    agent_name: str
    score: int = Field(default=0, ge=0, le=100)
    fit: Literal["strong", "good", "weak"] = "weak"
    rationale: str | None = None
    new_skill_ids: list[str] = Field(default_factory=list)
    overlap_lane_ids: list[str] = Field(default_factory=list)
    complementary_lane_ids: list[str] = Field(default_factory=list)
    new_tool_ids: list[str] = Field(default_factory=list)
    new_mcp_server_ids: list[str] = Field(default_factory=list)
    gap_cover_mcp_server_ids: list[str] = Field(default_factory=list)
    source_profile_id: Literal["coordinator", "research", "implementation", "verification", "generalist"] | None = None
    target_profile_id: Literal["coordinator", "research", "implementation", "verification", "generalist"] | None = None
    same_role_profile: bool | None = None
    same_role_profile_overlap_risk: bool | None = None
    edge_present: bool = False
    interaction: str | None = None


class HarnessDelegationOpportunity(BaseModel):
    source_agent_id: str
    source_agent_name: str
    source_lane_ids: list[str] = Field(default_factory=list)
    delegation_focus: str | None = None
    target: HarnessDelegationTargetFit
    suggested_replacements: list[HarnessDelegationTargetFit] = Field(default_factory=list)


class HarnessStudioGraphDiagnostics(BaseModel):
    weak_downstream_edges: list[HarnessDelegationOpportunity] = Field(default_factory=list)
    best_next_handoffs: list[HarnessDelegationOpportunity] = Field(default_factory=list)
    weak_edge_count: int = Field(default=0, ge=0)
    best_next_count: int = Field(default=0, ge=0)


class HarnessAgentExecutionContract(BaseModel):
    skill_execution_mode: Literal["guidance_only"] = "guidance_only"
    approved_skill_ids: list[str] = Field(default_factory=list)
    suggested_skill_ids: list[str] = Field(default_factory=list)
    tool_access_mode: Literal["direct_execution", "planning_only", "mixed", "none"] = "none"
    executable_tool_ids: list[str] = Field(default_factory=list)
    planning_only_tool_ids: list[str] = Field(default_factory=list)
    disabled_tool_ids: list[str] = Field(default_factory=list)
    mcp_access_mode: Literal["planning_only", "none"] = "none"
    planning_only_mcp_server_ids: list[str] = Field(default_factory=list)
    missing_mcp_server_ids: list[str] = Field(default_factory=list)


class HarnessAgentDelegationContract(BaseModel):
    primary_role_mode: Literal["coordinator", "research", "implementation", "verification", "generalist"] = (
        "generalist"
    )
    supporting_role_modes: list[str] = Field(default_factory=list)
    work_strategy: Literal[
        "synthesize_and_route",
        "gather_then_handoff",
        "implement_then_handoff",
        "verify_and_close",
        "self_contained_delivery",
        "flexible",
    ] = "flexible"
    should_coordinate_parallel_work: bool = False
    should_produce_final_output: bool = False
    primary_focus: str | None = None
    upstream_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    downstream_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    preferred_collaborators: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    weak_handoff_targets: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    watchouts: list[str] = Field(default_factory=list)


class HarnessAgentRoleProfileSuggestion(BaseModel):
    profile_id: Literal["coordinator", "research", "implementation", "verification", "generalist"] = (
        "generalist"
    )
    suggested_skill_ids: list[str] = Field(default_factory=list)
    available_skill_ids: list[str] = Field(default_factory=list)
    missing_skill_ids: list[str] = Field(default_factory=list)
    suggested_tool_ids: list[str] = Field(default_factory=list)
    suggested_mcp_server_ids: list[str] = Field(default_factory=list)
    restrictive_tool_ids: list[str] = Field(default_factory=list)
    restrictive_mcp_server_ids: list[str] = Field(default_factory=list)


class HarnessAgentCapabilitySummary(BaseModel):
    agent_id: str
    loaded_skill_ids: list[str] = Field(default_factory=list)
    missing_skill_ids: list[str] = Field(default_factory=list)
    missing_skill_details: list[HarnessSkillCatalogItem] = Field(default_factory=list)
    suggested_skill_ids: list[str] = Field(default_factory=list)
    loaded_skill_hints: list[str] = Field(default_factory=list)
    required_skill_ids: list[str] = Field(default_factory=list)
    missing_required_skill_ids: list[str] = Field(default_factory=list)
    required_tool_ids: list[str] = Field(default_factory=list)
    missing_required_tool_ids: list[str] = Field(default_factory=list)
    configured_allowed_tool_ids: list[str] = Field(default_factory=list)
    configured_denied_tool_ids: list[str] = Field(default_factory=list)
    enabled_tool_ids: list[str] = Field(default_factory=list)
    disabled_tool_ids: list[str] = Field(default_factory=list)
    policy_added_tool_ids: list[str] = Field(default_factory=list)
    policy_blocked_tool_ids: list[str] = Field(default_factory=list)
    unknown_allowed_tool_ids: list[str] = Field(default_factory=list)
    requires_tool_calling: bool = False
    provider_limited_tool_ids: list[str] = Field(default_factory=list)
    tool_execution_support: Literal["supported", "unsupported", "unknown"] = "unknown"
    tool_execution_support_reason: str | None = None
    required_mcp_server_ids: list[str] = Field(default_factory=list)
    missing_required_mcp_server_ids: list[str] = Field(default_factory=list)
    configured_allowed_mcp_server_ids: list[str] = Field(default_factory=list)
    configured_denied_mcp_server_ids: list[str] = Field(default_factory=list)
    mcp_server_ids: list[str] = Field(default_factory=list)
    missing_mcp_server_ids: list[str] = Field(default_factory=list)
    missing_mcp_server_details: list[HarnessMcpServerCatalogItem] = Field(default_factory=list)
    policy_added_mcp_server_ids: list[str] = Field(default_factory=list)
    policy_blocked_mcp_server_ids: list[str] = Field(default_factory=list)
    unknown_allowed_mcp_server_ids: list[str] = Field(default_factory=list)
    delegation_lane_ids: list[str] = Field(default_factory=list)
    recommended_collaborators: list[HarnessDelegationTargetFit] = Field(default_factory=list)
    downstream_handoff_scores: list[HarnessDelegationTargetFit] = Field(default_factory=list)
    delegation_focus: str | None = None
    availability_status: Literal["available", "limited", "unavailable"] = "available"
    availability_blockers: list[str] = Field(default_factory=list)
    availability_warnings: list[str] = Field(default_factory=list)
    readiness_status: Literal["ready", "limited", "blocked"] = "ready"
    readiness_blockers: list[str] = Field(default_factory=list)
    readiness_warnings: list[str] = Field(default_factory=list)
    provider_route: str | None = None
    review_mode: str | None = None
    capability_brief: str | None = None
    execution_contract: HarnessAgentExecutionContract = Field(default_factory=HarnessAgentExecutionContract)
    delegation_contract: HarnessAgentDelegationContract = Field(default_factory=HarnessAgentDelegationContract)
    role_profile_suggestion: HarnessAgentRoleProfileSuggestion = Field(
        default_factory=HarnessAgentRoleProfileSuggestion
    )



class HarnessCoordinationAgentPreview(BaseModel):
    agent_id: str
    agent_name: str


class HarnessCapabilityOwnerEntry(BaseModel):
    capability_id: str
    owner_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)


class HarnessOrchestrationBriefCapabilityRisk(BaseModel):
    kind: Literal["skill", "tool", "mcp"]
    capability_id: str
    owner_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)


class HarnessOrchestrationPhaseSummary(BaseModel):
    phase_id: Literal["research", "synthesis", "implementation", "verification"]
    agent_count: int = Field(default=0, ge=0)
    agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)


class HarnessOrchestrationRepairPriority(BaseModel):
    priority_id: Literal[
        "availability",
        "capability_gaps",
        "policy_repair",
        "role_profile_alignment",
        "weak_handoffs",
        "best_next_handoffs",
        "connectivity",
        "single_owner_capabilities",
        "review_path",
    ]
    severity: Literal["high", "medium", "low"]
    count: int = Field(default=0, ge=0)


class HarnessOrchestrationAgentRoutingSummary(BaseModel):
    coordinator_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    research_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    implementation_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    verification_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    skill_capable_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    tool_capable_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    mcp_capable_anchors: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)


class HarnessOrchestrationSummary(BaseModel):
    total_agent_count: int = Field(default=0, ge=0)
    execution_step_count: int = Field(default=0, ge=0)
    review_enabled: bool = False
    readiness: Literal["blocked", "repair", "watch", "ready"] = "ready"
    start_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    terminal_agents: list[HarnessCoordinationAgentPreview] = Field(default_factory=list)
    shared_lane_count: int = Field(default=0, ge=0)
    single_owner_capability_count: int = Field(default=0, ge=0)
    single_owner_capability_risks: list[HarnessOrchestrationBriefCapabilityRisk] = Field(default_factory=list)
    unavailable_count: int = Field(default=0, ge=0)
    limited_availability_count: int = Field(default=0, ge=0)
    policy_repair_agent_count: int = Field(default=0, ge=0)
    role_profile_drift_agent_count: int = Field(default=0, ge=0)
    role_profile_overlap_risk_count: int = Field(default=0, ge=0)
    weak_edge_count: int = Field(default=0, ge=0)
    best_next_count: int = Field(default=0, ge=0)
    capability_gap_count: int = Field(default=0, ge=0)
    isolated_agent_count: int = Field(default=0, ge=0)
    underconnected_agent_count: int = Field(default=0, ge=0)
    phases: list[HarnessOrchestrationPhaseSummary] = Field(default_factory=list)
    repair_priorities: list[HarnessOrchestrationRepairPriority] = Field(default_factory=list)
    agent_routing: HarnessOrchestrationAgentRoutingSummary = Field(
        default_factory=HarnessOrchestrationAgentRoutingSummary
    )


class HarnessExecutionChecklistItem(BaseModel):
    item_id: str
    content: str = Field(min_length=1, max_length=240)
    status: Literal["pending", "in_progress", "completed"] = "pending"
    active_form: str | None = Field(default=None, max_length=240)


class HarnessStudioGraph(BaseModel):
    version: int = 1
    agents: list[HarnessCanvasAgent] = Field(default_factory=list)
    edges: list[HarnessCanvasEdge] = Field(default_factory=list)
    graph_diagnostics: HarnessStudioGraphDiagnostics = Field(default_factory=HarnessStudioGraphDiagnostics)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    execution_checklist: list[HarnessExecutionChecklistItem] = Field(default_factory=list)
    skill_pool: list[HarnessSkillPoolItem] = Field(default_factory=list)
    pending_skill_requests: list[HarnessSkillRequest] = Field(default_factory=list)
    skill_catalog: list[HarnessSkillCatalogItem] = Field(default_factory=list)
    tool_catalog: list[HarnessToolCatalogItem] = Field(default_factory=list)
    mcp_server_catalog: list[HarnessMcpServerCatalogItem] = Field(default_factory=list)
    agent_capability_summaries: list[HarnessAgentCapabilitySummary] = Field(default_factory=list)
    orchestration_summary: HarnessOrchestrationSummary = Field(default_factory=HarnessOrchestrationSummary)
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
