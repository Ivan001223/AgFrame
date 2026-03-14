import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getStoredUsername } from '@/lib/auth/session';

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
  all: ['conversations'] as const,
  detail: (id: string) => ['conversations', id] as const,
};

export function useConversationsQuery() {
  return useQuery({
    queryKey: CONVERSATION_KEYS.all,
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
  return useQuery({
    queryKey: CONVERSATION_KEYS.detail(id),
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
