import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type HarnessApprovalDTO = {
  approval_id?: string;
  run_id?: string;
  action_type?: string;
  reason?: string | null;
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
};

export type HarnessRunDetailDTO = HarnessRunSummaryDTO & {
  input_json?: Record<string, unknown>;
  metadata_json?: Record<string, unknown> | null;
  retry_count?: number;
  events?: HarnessEventDTO[];
};

export type HarnessPolicyDTO = NonNullable<HarnessRunSummaryDTO['policy']>;

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
