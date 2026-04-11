import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getSessionCacheScope } from '@/lib/auth/session';

export type HarnessApprovalDTO = {
  approval_id?: string;
  run_id?: string;
  action_type?: string;
  reason?: string | null;
  payload_json?: Record<string, unknown> | null;
  status?: string;
  requested_by?: string | null;
  resolved_by?: string | null;
  comment?: string | null;
  created_at?: number;
  resolved_at?: number | null;
};

export type HarnessVerificationDTO = {
  verification_id?: string;
  run_id?: string;
  status?: string;
  checks_json?: {
    checks_run?: string[];
  } | null;
  artifacts_json?: Record<string, unknown> | null;
  summary?: string | null;
  created_at?: number;
};

export type HarnessRuntimeReviewDTO = {
  stage?: string | null;
  status?: string | null;
  agent_id?: string | null;
  agent_name?: string | null;
  review_output?: string | null;
  check_count?: number | null;
  segment_index?: number | null;
  segment_count?: number | null;
  segment_start_char?: number | null;
  segment_end_char?: number | null;
  last_reviewed_char?: number | null;
};

export type HarnessRuntimeContinuationDTO = {
  enabled?: boolean;
  mode?: string | null;
  status?: string | null;
  agent_id?: string | null;
  agent_name?: string | null;
  step_index?: number | null;
  prefix_length?: number;
  resumed_at?: number | null;
  completed_at?: number | null;
};

export type HarnessRuntimeResearchDTO = {
  enabled?: boolean;
  mode?: string | null;
  paper_count?: number;
  browser_preview_count?: number;
  source_count?: number;
  cluster_ids?: string[];
};

export type HarnessRuntimeStateDTO = {
  review?: HarnessRuntimeReviewDTO | null;
  continuation?: HarnessRuntimeContinuationDTO | null;
  research?: HarnessRuntimeResearchDTO | null;
};

export type HarnessWorkflowStepDTO = {
  step_id: string;
  step_index: number;
  loop_number: number;
  label: string;
  execution_id?: string | null;
  node_id?: string | null;
  status?: 'pending' | 'in_progress' | 'completed' | 'blocked';
  kind?: 'agent' | 'cluster_member' | 'cluster_summary';
};

export type HarnessWorkflowProgressDTO = {
  enabled?: boolean;
  status?: 'idle' | 'pending' | 'running' | 'blocked' | 'completed' | 'failed';
  total_steps?: number;
  completed_steps?: number;
  blocked_steps?: number;
  review_enabled?: boolean;
  current_step_index?: number | null;
  current_step_label?: string | null;
  blocking_step_index?: number | null;
  blocking_step_label?: string | null;
  blocking_stage?: string | null;
  blocking_reason?: string | null;
  steps?: HarnessWorkflowStepDTO[];
};

export type HarnessEventDTO = {
  event_id?: string;
  event_type?: string;
  event_source?: string;
  user_id?: string;
  session_id?: string | null;
  run_id?: string | null;
  actor?: string | null;
  details_json?: Record<string, unknown> | null;
  created_at?: number;
};

export type HarnessRunChecklistSnapshotDTO = {
  enabled?: boolean;
  total_items?: number;
  open_items?: number;
  completed_items?: number;
  items?: HarnessExecutionChecklistItemDTO[];
};

export type HarnessRunSummaryDTO = {
  run_id: string;
  user_id: string;
  project_id?: string | null;
  project_name?: string | null;
  session_id?: string | null;
  task_type?: string;
  status?: string;
  policy_id?: string;
  current_step?: string | null;
  resume_count?: number;
  approval_required?: boolean;
  verification_status?: string | null;
  created_at?: number;
  updated_at?: number;
  finished_at?: number | null;
  can_retry?: boolean;
  policy?: {
    policy_id?: string;
    task_type?: string;
    approval_required?: boolean;
    allowed_tools?: string[];
    verification_profile?: string;
    retry_budget?: number;
  } | null;
  latest_approval?: HarnessApprovalDTO | null;
  latest_verification?: HarnessVerificationDTO | null;
  runtime_state?: HarnessRuntimeStateDTO | null;
  workflow_progress?: HarnessWorkflowProgressDTO | null;
  checklist_snapshot?: HarnessRunChecklistSnapshotDTO | null;
};

export type HarnessRunDetailDTO = HarnessRunSummaryDTO & {
  input_json?: Record<string, unknown>;
  metadata_json?: Record<string, unknown> | null;
  retry_count?: number;
  events?: HarnessEventDTO[];
};

export type HarnessPolicyDTO = NonNullable<HarnessRunSummaryDTO['policy']>;

export type HarnessCanvasPositionDTO = {
  x: number;
  y: number;
};

export type HarnessReviewAgentDTO = {
  enabled: boolean;
  hidden: boolean;
  name: string;
  model: string;
  preferred_provider_id?: string | null;
  fallback_provider_id?: string | null;
  system_prompt: string;
};

export type HarnessClusterMemberDTO = {
  member_id: string;
  name: string;
  role?: string;
  system_prompt?: string;
  model?: string;
  preferred_provider_id?: string | null;
  fallback_provider_id?: string | null;
  temperature?: number;
  timeout_seconds?: number | null;
};

export type HarnessCanvasAgentDTO = {
  agent_id: string;
  name: string;
  node_kind?: 'agent' | 'cluster';
  cluster_strategy?: 'brainstorm' | 'custom' | null;
  role?: string;
  description?: string | null;
  system_prompt?: string;
  model?: string;
  preferred_provider_id?: string | null;
  fallback_provider_id?: string | null;
  temperature?: number;
  max_iterations?: number;
  position?: HarnessCanvasPositionDTO;
  skill_ids?: string[];
  skill_intents?: string[];
  required_skill_ids?: string[];
  required_tool_ids?: string[];
  allowed_tool_ids?: string[];
  denied_tool_ids?: string[];
  requires_tool_calling?: boolean;
  required_mcp_server_ids?: string[];
  allowed_mcp_server_ids?: string[];
  denied_mcp_server_ids?: string[];
  cluster_members?: HarnessClusterMemberDTO[];
  brainstorm_rounds?: number;
  cluster_auto_research?: boolean;
  cluster_auto_review?: boolean;
};

export type HarnessCanvasEdgeDTO = {
  edge_id: string;
  source_agent_id: string;
  target_agent_id: string;
  interaction?: string;
  condition?: string | null;
};

export type HarnessSkillCatalogItemDTO = {
  skill_id: string;
  title: string;
  description?: string | null;
  source: string;
  status?: string;
  prompt_hint?: string | null;
  suggested_tool_ids?: string[];
  suggested_mcp_server_ids?: string[];
};

export type HarnessSkillPoolItemDTO = {
  skill_id: string;
  title: string;
  description?: string | null;
  source: string;
  status?: string;
  approved_at?: number | null;
};

export type HarnessSkillRequestDTO = {
  request_id: string;
  agent_id: string;
  skill_id: string;
  title: string;
  source: string;
  status?: string;
  reason?: string | null;
  discovered_at?: number;
  resolved_at?: number | null;
};

export type HarnessToolCatalogItemDTO = {
  tool_id: string;
  title: string;
  description?: string | null;
  status?: 'enabled' | 'disabled';
  requires_flag?: string | null;
};

export type HarnessMcpServerCatalogItemDTO = {
  server_id: string;
  title: string;
  description?: string | null;
  status?: 'enabled' | 'disabled';
  command_preview?: string | null;
};

export type HarnessDelegationTargetFitDTO = {
  agent_id: string;
  agent_name: string;
  score?: number;
  fit?: 'strong' | 'good' | 'weak';
  rationale?: string | null;
  new_skill_ids?: string[];
  overlap_lane_ids?: string[];
  complementary_lane_ids?: string[];
  new_tool_ids?: string[];
  new_mcp_server_ids?: string[];
  gap_cover_mcp_server_ids?: string[];
  source_profile_id?: 'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist' | null;
  target_profile_id?: 'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist' | null;
  same_role_profile?: boolean;
  same_role_profile_overlap_risk?: boolean;
  edge_present?: boolean;
  interaction?: string | null;
};

export type HarnessDelegationOpportunityDTO = {
  source_agent_id: string;
  source_agent_name: string;
  source_lane_ids?: string[];
  delegation_focus?: string | null;
  target: HarnessDelegationTargetFitDTO;
  suggested_replacements?: HarnessDelegationTargetFitDTO[];
};

export type HarnessStudioGraphDiagnosticsDTO = {
  weak_downstream_edges?: HarnessDelegationOpportunityDTO[];
  best_next_handoffs?: HarnessDelegationOpportunityDTO[];
  weak_edge_count?: number;
  best_next_count?: number;
};

export type HarnessCoordinationAgentPreviewDTO = {
  agent_id: string;
  agent_name: string;
};

export type HarnessCapabilityOwnerEntryDTO = {
  capability_id: string;
  owner_agents?: HarnessCoordinationAgentPreviewDTO[];
};

export type HarnessOrchestrationBriefCapabilityRiskDTO = {
  kind?: 'skill' | 'tool' | 'mcp';
  capability_id: string;
  owner_agents?: HarnessCoordinationAgentPreviewDTO[];
};

export type HarnessOrchestrationPhaseSummaryDTO = {
  phase_id?: 'research' | 'synthesis' | 'implementation' | 'verification';
  agent_count?: number;
  agents?: HarnessCoordinationAgentPreviewDTO[];
};

export type HarnessOrchestrationRepairPriorityDTO = {
  priority_id?:
    | 'availability'
    | 'capability_gaps'
    | 'policy_repair'
    | 'role_profile_alignment'
    | 'weak_handoffs'
    | 'best_next_handoffs'
    | 'connectivity'
    | 'single_owner_capabilities'
    | 'review_path';
  severity?: 'high' | 'medium' | 'low';
  count?: number;
};

export type HarnessOrchestrationAgentRoutingSummaryDTO = {
  coordinator_anchors?: HarnessCoordinationAgentPreviewDTO[];
  research_anchors?: HarnessCoordinationAgentPreviewDTO[];
  implementation_anchors?: HarnessCoordinationAgentPreviewDTO[];
  verification_anchors?: HarnessCoordinationAgentPreviewDTO[];
  skill_capable_anchors?: HarnessCoordinationAgentPreviewDTO[];
  tool_capable_anchors?: HarnessCoordinationAgentPreviewDTO[];
  mcp_capable_anchors?: HarnessCoordinationAgentPreviewDTO[];
};

export type HarnessOrchestrationSummaryDTO = {
  total_agent_count?: number;
  execution_step_count?: number;
  review_enabled?: boolean;
  readiness?: 'blocked' | 'repair' | 'watch' | 'ready';
  start_agents?: HarnessCoordinationAgentPreviewDTO[];
  terminal_agents?: HarnessCoordinationAgentPreviewDTO[];
  shared_lane_count?: number;
  single_owner_capability_count?: number;
  single_owner_capability_risks?: HarnessOrchestrationBriefCapabilityRiskDTO[];
  unavailable_count?: number;
  limited_availability_count?: number;
  policy_repair_agent_count?: number;
  role_profile_drift_agent_count?: number;
  role_profile_overlap_risk_count?: number;
  weak_edge_count?: number;
  best_next_count?: number;
  capability_gap_count?: number;
  isolated_agent_count?: number;
  underconnected_agent_count?: number;
  phases?: HarnessOrchestrationPhaseSummaryDTO[];
  repair_priorities?: HarnessOrchestrationRepairPriorityDTO[];
  agent_routing?: HarnessOrchestrationAgentRoutingSummaryDTO;
};

export type HarnessAgentExecutionContractDTO = {
  skill_execution_mode?: 'guidance_only';
  approved_skill_ids?: string[];
  suggested_skill_ids?: string[];
  tool_access_mode?: 'direct_execution' | 'planning_only' | 'mixed' | 'none';
  executable_tool_ids?: string[];
  planning_only_tool_ids?: string[];
  disabled_tool_ids?: string[];
  mcp_access_mode?: 'planning_only' | 'none';
  planning_only_mcp_server_ids?: string[];
  missing_mcp_server_ids?: string[];
};

export type HarnessAgentDelegationContractDTO = {
  primary_role_mode?: 'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist';
  supporting_role_modes?: Array<
    'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist'
  >;
  work_strategy?:
    | 'synthesize_and_route'
    | 'gather_then_handoff'
    | 'implement_then_handoff'
    | 'verify_and_close'
    | 'self_contained_delivery'
    | 'flexible';
  should_coordinate_parallel_work?: boolean;
  should_produce_final_output?: boolean;
  primary_focus?: string | null;
  upstream_agents?: HarnessCoordinationAgentPreviewDTO[];
  downstream_agents?: HarnessCoordinationAgentPreviewDTO[];
  preferred_collaborators?: HarnessCoordinationAgentPreviewDTO[];
  weak_handoff_targets?: HarnessCoordinationAgentPreviewDTO[];
  watchouts?: string[];
};

export type HarnessAgentRoleProfileSuggestionDTO = {
  profile_id?: 'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist';
  suggested_skill_ids?: string[];
  available_skill_ids?: string[];
  missing_skill_ids?: string[];
  suggested_tool_ids?: string[];
  suggested_mcp_server_ids?: string[];
  restrictive_tool_ids?: string[];
  restrictive_mcp_server_ids?: string[];
};

export type HarnessAgentCapabilitySummaryDTO = {
  agent_id: string;
  loaded_skill_ids?: string[];
  missing_skill_ids?: string[];
  missing_skill_details?: HarnessSkillCatalogItemDTO[];
  suggested_skill_ids?: string[];
  loaded_skill_hints?: string[];
  required_skill_ids?: string[];
  missing_required_skill_ids?: string[];
  required_tool_ids?: string[];
  missing_required_tool_ids?: string[];
  configured_allowed_tool_ids?: string[];
  configured_denied_tool_ids?: string[];
  enabled_tool_ids?: string[];
  disabled_tool_ids?: string[];
  policy_added_tool_ids?: string[];
  policy_blocked_tool_ids?: string[];
  unknown_allowed_tool_ids?: string[];
  requires_tool_calling?: boolean;
  provider_limited_tool_ids?: string[];
  tool_execution_support?: 'supported' | 'unsupported' | 'unknown';
  tool_execution_support_reason?: string | null;
  required_mcp_server_ids?: string[];
  missing_required_mcp_server_ids?: string[];
  configured_allowed_mcp_server_ids?: string[];
  configured_denied_mcp_server_ids?: string[];
  mcp_server_ids?: string[];
  missing_mcp_server_ids?: string[];
  missing_mcp_server_details?: HarnessMcpServerCatalogItemDTO[];
  policy_added_mcp_server_ids?: string[];
  policy_blocked_mcp_server_ids?: string[];
  unknown_allowed_mcp_server_ids?: string[];
  delegation_lane_ids?: string[];
  recommended_collaborators?: HarnessDelegationTargetFitDTO[];
  downstream_handoff_scores?: HarnessDelegationTargetFitDTO[];
  delegation_focus?: string | null;
  availability_status?: 'available' | 'limited' | 'unavailable';
  availability_blockers?: string[];
  availability_warnings?: string[];
  readiness_status?: 'ready' | 'limited' | 'blocked';
  readiness_blockers?: string[];
  readiness_warnings?: string[];
  provider_route?: string | null;
  review_mode?: string | null;
  capability_brief?: string | null;
  execution_contract?: HarnessAgentExecutionContractDTO | null;
  delegation_contract?: HarnessAgentDelegationContractDTO | null;
  role_profile_suggestion?: HarnessAgentRoleProfileSuggestionDTO | null;
};

export type HarnessStudioProviderConfigDTO = {
  preferred_provider_id?: string | null;
  fallback_provider_id?: string | null;
};

export type HarnessExecutionChecklistItemDTO = {
  item_id: string;
  content: string;
  status?: 'pending' | 'in_progress' | 'completed';
  active_form?: string | null;
};

export type HarnessModelProviderDTO = {
  provider_id: string;
  user_id: string;
  name: string;
  base_url: string;
  models: string[];
  is_default: boolean;
  enabled: boolean;
  created_at?: number;
  updated_at?: number;
};

export type HarnessStudioGraphDTO = {
  version?: number;
  agents?: HarnessCanvasAgentDTO[];
  edges?: HarnessCanvasEdgeDTO[];
  graph_diagnostics?: HarnessStudioGraphDiagnosticsDTO;
  orchestration_summary?: HarnessOrchestrationSummaryDTO;
  knowledge_base_ids?: string[];
  execution_checklist?: HarnessExecutionChecklistItemDTO[];
  skill_pool?: HarnessSkillPoolItemDTO[];
  pending_skill_requests?: HarnessSkillRequestDTO[];
  skill_catalog?: HarnessSkillCatalogItemDTO[];
  tool_catalog?: HarnessToolCatalogItemDTO[];
  mcp_server_catalog?: HarnessMcpServerCatalogItemDTO[];
  agent_capability_summaries?: HarnessAgentCapabilitySummaryDTO[];
  review_agent?: HarnessReviewAgentDTO;
  canvas?: {
    x?: number;
    y?: number;
    zoom?: number;
  };
  provider_config?: HarnessStudioProviderConfigDTO;
};

export type HarnessProjectSummaryDTO = {
  project_id: string;
  user_id?: string;
  name: string;
  description?: string | null;
  created_at?: number;
  updated_at?: number;
  agent_count?: number;
  edge_count?: number;
  checklist_count?: number;
  open_checklist_count?: number;
  loaded_skill_count?: number;
  pending_skill_request_count?: number;
  graph_json?: HarnessStudioGraphDTO;
};

export type HarnessProjectDetailDTO = HarnessProjectSummaryDTO & {
  graph_json: HarnessStudioGraphDTO;
  skill_request_result?: {
    available_skill_ids?: string[];
    created_requests?: HarnessSkillRequestDTO[];
  };
  resolved_skill_request?: HarnessSkillRequestDTO;
};

export const HARNESS_KEYS = {
  runs: (scope: string) => ['harness', scope, 'runs'] as const,
  run: (scope: string, runId: string | null) => ['harness', scope, 'run', runId] as const,
  policies: (scope: string) => ['harness', scope, 'policies'] as const,
  studioProjects: (scope: string) => ['harness', scope, 'studio', 'projects'] as const,
  currentProject: (scope: string) => ['harness', scope, 'studio', 'current-project'] as const,
  project: (scope: string, projectId: string | null) => ['harness', scope, 'studio', 'project', projectId] as const,
  modelProviders: (scope: string) => ['harness', scope, 'model-providers'] as const,
};

export function useHarnessRunsQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.runs(scope),
    queryFn: async () => apiClient<{ runs: HarnessRunSummaryDTO[] }>('/harness/runs'),
    refetchInterval: 5000,
  });
}

export function useHarnessRunDetailQuery(runId: string | null) {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.run(scope, runId),
    queryFn: async () => apiClient<HarnessRunDetailDTO>(`/harness/runs/${runId}`),
    enabled: !!runId,
    refetchInterval: 5000,
  });
}

export function useHarnessPoliciesQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.policies(scope),
    queryFn: async () => apiClient<{ policies: HarnessPolicyDTO[] }>('/harness/policies'),
    staleTime: 5 * 60 * 1000,
  });
}

export function useHarnessStudioProjectsQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.studioProjects(scope),
    queryFn: async () => apiClient<{ projects: HarnessProjectSummaryDTO[] }>('/harness/studio/projects'),
  });
}

export function useHarnessCurrentStudioProjectQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.currentProject(scope),
    queryFn: async () => apiClient<HarnessProjectDetailDTO>('/harness/studio/projects/current'),
  });
}

export function useHarnessStudioProjectQuery(projectId: string | null) {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.project(scope, projectId),
    queryFn: async () => apiClient<HarnessProjectDetailDTO>(`/harness/studio/projects/${projectId}`),
    enabled: !!projectId,
  });
}

export function useHarnessCreateStudioProjectMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({ name, description }: { name: string; description?: string }) =>
      apiClient<HarnessProjectDetailDTO>('/harness/studio/projects', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.studioProjects(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.currentProject(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.project(scope, payload.project_id) }),
      ]);
    },
  });
}

export function useHarnessUpdateStudioProjectMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      projectId,
      name,
      description,
      graphJson,
    }: {
      projectId: string;
      name?: string;
      description?: string | null;
      graphJson?: HarnessStudioGraphDTO;
    }) =>
      apiClient<HarnessProjectDetailDTO>(`/harness/studio/projects/${projectId}`, {
        method: 'PUT',
        body: JSON.stringify({
          name,
          description,
          graph_json: graphJson,
        }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.studioProjects(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.currentProject(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.project(scope, payload.project_id) }),
      ]);
    },
  });
}

export function useHarnessSkillRequestMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      projectId,
      agentId,
      requestedSkills,
    }: {
      projectId: string;
      agentId: string;
      requestedSkills: string[];
    }) =>
      apiClient<HarnessProjectDetailDTO>(`/harness/studio/projects/${projectId}/skill-requests`, {
        method: 'POST',
        body: JSON.stringify({
          agent_id: agentId,
          requested_skills: requestedSkills,
        }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.studioProjects(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.currentProject(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.project(scope, payload.project_id) }),
      ]);
    },
  });
}

export function useHarnessSkillDecisionMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      projectId,
      requestId,
      approved,
    }: {
      projectId: string;
      requestId: string;
      approved: boolean;
    }) =>
      apiClient<HarnessProjectDetailDTO>(`/harness/studio/projects/${projectId}/skill-requests/${requestId}`, {
        method: 'POST',
        body: JSON.stringify({ approved }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.studioProjects(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.currentProject(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.project(scope, payload.project_id) }),
      ]);
    },
  });
}

export function useHarnessStudioRunMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      projectId,
      runScope,
      agentIds,
      loopCount,
      task,
      timeoutSeconds,
    }: {
      projectId: string;
      runScope: 'all' | 'selected';
      agentIds: string[];
      loopCount: number;
      task?: string;
      timeoutSeconds?: number;
    }) =>
      apiClient<HarnessRunDetailDTO>(`/harness/studio/projects/${projectId}/run`, {
        method: 'POST',
        body: JSON.stringify({
          run_scope: runScope,
          agent_ids: agentIds,
          loop_count: loopCount,
          task,
          timeout_seconds: timeoutSeconds,
        }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.runs(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.run(scope, payload.run_id) }),
      ]);
    },
  });
}

export function useHarnessCreateRunMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      taskType,
      input,
      sessionId,
      metadata,
    }: {
      taskType: string;
      input: Record<string, unknown>;
      sessionId?: string;
      metadata?: Record<string, unknown>;
    }) =>
      apiClient<HarnessRunDetailDTO>('/harness/runs', {
        method: 'POST',
        body: JSON.stringify({
          task_type: taskType,
          input,
          session_id: sessionId || undefined,
          metadata,
        }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.runs(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.policies(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.run(scope, payload.run_id) }),
      ]);
    },
  });
}

export function useHarnessApprovalMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({
      runId,
      approved,
      comment,
    }: {
      runId: string;
      approved: boolean;
      comment?: string;
    }) =>
      apiClient(`/harness/runs/${runId}/approval`, {
        method: 'POST',
        body: JSON.stringify({ approved, comment: comment?.trim() || undefined }),
      }),
    onSuccess: async (_payload, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.runs(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.run(scope, variables.runId) }),
      ]);
    },
  });
}

export function useHarnessRetryRunMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({ runId }: { runId: string }) =>
      apiClient<HarnessRunDetailDTO>(`/harness/runs/${runId}/retry`, {
        method: 'POST',
      }),
    onSuccess: async (payload, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.runs(scope) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.run(scope, variables.runId) }),
        queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.run(scope, payload.run_id) }),
      ]);
    },
  });
}

export function useHarnessModelProvidersQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: HARNESS_KEYS.modelProviders(scope),
    queryFn: async () => apiClient<{ providers: HarnessModelProviderDTO[] }>('/harness/model-providers'),
  });
}

export function useHarnessCreateModelProviderMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (payload: {
      name: string;
      base_url: string;
      api_key: string;
      models: string[];
      is_default?: boolean;
      enabled?: boolean;
    }) =>
      apiClient<HarnessModelProviderDTO>('/harness/model-providers', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.modelProviders(scope) });
    },
  });
}

export function useHarnessUpdateModelProviderMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({ providerId, ...payload }: { providerId: string } & Partial<{
      name: string;
      base_url: string;
      api_key: string;
      models: string[];
      is_default: boolean;
      enabled: boolean;
    }>) =>
      apiClient<HarnessModelProviderDTO>(`/harness/model-providers/${providerId}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.modelProviders(scope) });
    },
  });
}

export function useHarnessDeleteModelProviderMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (providerId: string) =>
      apiClient<{ success: boolean }>(`/harness/model-providers/${providerId}`, {
        method: 'DELETE',
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: HARNESS_KEYS.modelProviders(scope) });
    },
  });
}
