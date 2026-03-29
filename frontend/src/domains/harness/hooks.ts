import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

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

export type HarnessRunSummaryDTO = {
  run_id: string;
  user_id: string;
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

export type HarnessStudioProviderConfigDTO = {
  preferred_provider_id?: string | null;
  fallback_provider_id?: string | null;
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
  skill_pool?: HarnessSkillPoolItemDTO[];
  pending_skill_requests?: HarnessSkillRequestDTO[];
  skill_catalog?: HarnessSkillCatalogItemDTO[];
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

export function useHarnessRunsQuery() {
  return useQuery({
    queryKey: ['harness', 'runs'],
    queryFn: async () => apiClient<{ runs: HarnessRunSummaryDTO[] }>('/harness/runs'),
    refetchInterval: 5000,
  });
}

export function useHarnessRunDetailQuery(runId: string | null) {
  return useQuery({
    queryKey: ['harness', 'run', runId],
    queryFn: async () => apiClient<HarnessRunDetailDTO>(`/harness/runs/${runId}`),
    enabled: !!runId,
    refetchInterval: 5000,
  });
}

export function useHarnessPoliciesQuery() {
  return useQuery({
    queryKey: ['harness', 'policies'],
    queryFn: async () => apiClient<{ policies: HarnessPolicyDTO[] }>('/harness/policies'),
    staleTime: 5 * 60 * 1000,
  });
}

export function useHarnessStudioProjectsQuery() {
  return useQuery({
    queryKey: ['harness', 'studio', 'projects'],
    queryFn: async () => apiClient<{ projects: HarnessProjectSummaryDTO[] }>('/harness/studio/projects'),
  });
}

export function useHarnessCurrentStudioProjectQuery() {
  return useQuery({
    queryKey: ['harness', 'studio', 'current-project'],
    queryFn: async () => apiClient<HarnessProjectDetailDTO>('/harness/studio/projects/current'),
  });
}

export function useHarnessStudioProjectQuery(projectId: string | null) {
  return useQuery({
    queryKey: ['harness', 'studio', 'project', projectId],
    queryFn: async () => apiClient<HarnessProjectDetailDTO>(`/harness/studio/projects/${projectId}`),
    enabled: !!projectId,
  });
}

export function useHarnessCreateStudioProjectMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ name, description }: { name: string; description?: string }) =>
      apiClient<HarnessProjectDetailDTO>('/harness/studio/projects', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      }),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'current-project'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'project', payload.project_id] }),
      ]);
    },
  });
}

export function useHarnessUpdateStudioProjectMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'current-project'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'project', payload.project_id] }),
      ]);
    },
  });
}

export function useHarnessSkillRequestMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'current-project'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'project', payload.project_id] }),
      ]);
    },
  });
}

export function useHarnessSkillDecisionMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'current-project'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'project', payload.project_id] }),
      ]);
    },
  });
}

export function useHarnessStudioRunMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'runs'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'run', payload.run_id] }),
      ]);
    },
  });
}

export function useHarnessCreateRunMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'runs'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'policies'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'run', payload.run_id] }),
      ]);
    },
  });
}

export function useHarnessApprovalMutation() {
  const queryClient = useQueryClient();

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
        queryClient.invalidateQueries({ queryKey: ['harness', 'runs'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'run', variables.runId] }),
      ]);
    },
  });
}

export function useHarnessRetryRunMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ runId }: { runId: string }) =>
      apiClient<HarnessRunDetailDTO>(`/harness/runs/${runId}/retry`, {
        method: 'POST',
      }),
    onSuccess: async (payload, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['harness', 'runs'] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'run', variables.runId] }),
        queryClient.invalidateQueries({ queryKey: ['harness', 'run', payload.run_id] }),
      ]);
    },
  });
}

export function useHarnessModelProvidersQuery() {
  return useQuery({
    queryKey: ['harness', 'model-providers'],
    queryFn: async () => apiClient<{ providers: HarnessModelProviderDTO[] }>('/harness/model-providers'),
  });
}

export function useHarnessCreateModelProviderMutation() {
  const queryClient = useQueryClient();

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
      await queryClient.invalidateQueries({ queryKey: ['harness', 'model-providers'] });
    },
  });
}

export function useHarnessUpdateModelProviderMutation() {
  const queryClient = useQueryClient();

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
      await queryClient.invalidateQueries({ queryKey: ['harness', 'model-providers'] });
    },
  });
}

export function useHarnessDeleteModelProviderMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (providerId: string) =>
      apiClient<{ success: boolean }>(`/harness/model-providers/${providerId}`, {
        method: 'DELETE',
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['harness', 'model-providers'] });
    },
  });
}
