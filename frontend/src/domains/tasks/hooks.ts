import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type TaskStatus = 'queued' | 'running' | 'failed' | 'unknown';

export type TaskSummaryDTO = {
  total: number;
  status_counts: Record<string, number>;
  top_errors: Array<{
    error_code: string;
    count: number;
    title: string;
  }>;
  suspected_timeouts: Array<{
    task_id: string;
    status: string;
    stage?: string;
    age_seconds?: number;
    user_id?: string;
  }>;
  recent_incidents: TaskIncidentDTO[];
};

export type TaskIncidentDTO = {
  incident_id?: string;
  task_id: string;
  user_id?: string;
  error_code?: string;
  title?: string;
  user_message?: string;
  suggested_action?: string;
  handled?: boolean;
  archived?: boolean;
  created_at?: number;
  updated_at?: number;
};

export type TaskDetailDTO = {
  task_id: string;
  status: string;
  progress?: number | string;
  step?: string;
  message?: string;
  filename?: string;
  created_at?: number | string;
  error?: string;
  diagnostics?: {
    status: string;
    stage?: string;
    error_code?: string;
    error_message?: string;
    retryable?: boolean;
    age_seconds?: number;
    timeout_exceeded?: boolean;
    title?: string;
    user_message?: string;
    suggested_action?: string;
  };
};

export const TASK_KEYS = {
  summary: ['tasks', 'summary'] as const,
  incidents: (filters: { handled?: boolean; archived?: boolean }) => ['tasks', 'incidents', filters] as const,
  detail: (id: string) => ['tasks', id] as const,
};

export function useTaskSummaryQuery() {
  return useQuery({
    queryKey: TASK_KEYS.summary,
    queryFn: async () => apiClient<TaskSummaryDTO>('/tasks/summary'),
    refetchInterval: 10000,
  });
}

export function useTaskIncidentsQuery(filters: { handled?: boolean; archived?: boolean } = {}) {
  return useQuery({
    queryKey: TASK_KEYS.incidents(filters),
    queryFn: async () => {
      const params: Record<string, boolean | undefined> = {
        handled: filters.handled,
        archived: filters.archived,
      };

      const response = await apiClient<{ incidents: TaskIncidentDTO[] }>('/tasks/incidents', {
        params,
      });
      return response.incidents;
    },
    refetchInterval: 10000,
  });
}

export function useTaskDetailQuery(taskId: string) {
  return useQuery({
    queryKey: TASK_KEYS.detail(taskId),
    queryFn: async () => {
      return apiClient<TaskDetailDTO>(`/tasks/${taskId}`);
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const task = query.state.data;
      if (
        task &&
        (task.status === 'queued' ||
          task.status === 'running' ||
          task.status === 'processing' ||
          task.status === 'indexing')
      ) {
        return 3000; // Fast poll for active tasks
      }
      return false; // Stop polling when finished
    },
  });
}
