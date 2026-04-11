import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getSessionCacheScope, getStoredUsername } from '@/lib/auth/session';

export type ConversationDTO = {
  id: string;
  title?: string;
  created_at: number;
  updated_at: number;
  messages: Array<{
    role: string;
    content: string;
    created_at: number;
    token_count?: number | null;
  }>;
};

export const CONVERSATION_KEYS = {
  all: (scope: string) => ['conversations', scope] as const,
  detail: (scope: string, id: string) => ['conversations', scope, id] as const,
};

export function useConversationsQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: CONVERSATION_KEYS.all(scope),
    queryFn: async () => {
      const username = getStoredUsername();
      if (!username) {
        return [];
      }

      const response = await apiClient<{ history: ConversationDTO[] }>(`/history/${username}`);
      return response.history;
    },
  });
}

export function useConversationDetailQuery(id: string) {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: CONVERSATION_KEYS.detail(scope, id),
    queryFn: async () => {
      const username = getStoredUsername();
      if (!username) {
        throw new Error('Missing current user');
      }

      return apiClient<ConversationDTO>(`/history/${username}/${id}`);
    },
    enabled: !!id,
  });
}
