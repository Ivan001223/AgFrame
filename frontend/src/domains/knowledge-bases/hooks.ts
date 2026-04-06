import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type KnowledgeBaseDTO = {
  knowledge_base_id: string;
  user_id?: string;
  name: string;
  description?: string | null;
  document_count?: number;
  created_at?: number;
  updated_at?: number;
};

export const KNOWLEDGE_BASE_KEYS = {
  all: ['knowledge-bases'] as const,
};

export function useKnowledgeBasesQuery() {
  return useQuery({
    queryKey: KNOWLEDGE_BASE_KEYS.all,
    queryFn: async () => {
      const response = await apiClient<{ knowledge_bases: KnowledgeBaseDTO[] }>('/knowledge-bases');
      return response.knowledge_bases;
    },
  });
}

export function useCreateKnowledgeBaseMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ name, description }: { name: string; description?: string }) =>
      apiClient<KnowledgeBaseDTO>('/knowledge-bases', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: KNOWLEDGE_BASE_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}

export function useUpdateKnowledgeBaseMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ knowledgeBaseId, name, description }: { knowledgeBaseId: string; name?: string; description?: string | null }) =>
      apiClient<KnowledgeBaseDTO>(`/knowledge-bases/${knowledgeBaseId}`, {
        method: 'PUT',
        body: JSON.stringify({ name, description }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: KNOWLEDGE_BASE_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] });
    },
  });
}

export function useDeleteKnowledgeBaseMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (knowledgeBaseId: string) =>
      apiClient<{ deleted: boolean; knowledge_base_id: string }>(`/knowledge-bases/${knowledgeBaseId}`, {
        method: 'DELETE',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: KNOWLEDGE_BASE_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['harness', 'studio', 'projects'] });
    },
  });
}
