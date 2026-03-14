import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getStoredUsername } from '@/lib/auth/session';

export type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

export type InterruptStatusDTO = {
  session_id: string;
  interrupted: boolean;
  action_required?: {
    action_type?: string;
    description?: string;
    approved?: boolean;
    approved_by?: string | null;
    approved_at?: string | null;
  } | null;
  checkpoint_saved_at?: string | null;
};

type ChatInvokeParams = {
  sessionId: string;
  messages: ChatMessage[];
};

function normalizeMessageContent(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item === 'object' && 'text' in item) {
          return String(item.text ?? '');
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  if (value && typeof value === 'object' && 'text' in value) {
    return String((value as { text?: unknown }).text ?? '');
  }
  return '';
}

function extractAssistantReply(payload: unknown): string {
  const root = (payload ?? {}) as Record<string, unknown>;
  const output = (root.output ?? root) as Record<string, unknown>;
  const messages = Array.isArray(output.messages) ? output.messages : [];

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as Record<string, unknown>;
    const type = String(message.type ?? message.role ?? '');
    if (type === 'ai' || type === 'assistant') {
      return normalizeMessageContent(message.content);
    }
  }

  if (typeof output.output === 'string') {
    return output.output;
  }
  if (typeof output.answer === 'string') {
    return output.answer;
  }
  return '';
}

async function persistConversation(sessionId: string, messages: ChatMessage[]) {
  const username = getStoredUsername();
  if (!username) {
    return;
  }

  const title =
    messages.find((message) => message.role === 'user')?.content.slice(0, 80) ||
    'New chat';

  await apiClient(`/history/${username}/save`, {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      title,
      messages,
    }),
  });
}

export function useChatInvokeMutation() {
  return useMutation({
    mutationFn: async ({ sessionId, messages }: ChatInvokeParams) => {
      const payload = await apiClient<Record<string, unknown>>('/chat/invoke', {
        method: 'POST',
        body: JSON.stringify({
          input: {
            messages,
            context: {
              session_id: sessionId,
            },
          },
          config: {
            configurable: {
              thread_id: sessionId,
            },
          },
        }),
        timeout: 60000,
      });

      const reply = extractAssistantReply(payload).trim();
      const nextMessages = reply
        ? [...messages, { role: 'assistant' as const, content: reply }]
        : messages;

      await persistConversation(sessionId, nextMessages);

      return {
        payload,
        reply,
        messages: nextMessages,
      };
    },
  });
}

export function useInterruptStatusQuery(sessionId: string) {
  return useQuery({
    queryKey: ['interrupt', sessionId],
    queryFn: async () => apiClient<InterruptStatusDTO>(`/interrupt/${sessionId}`),
    enabled: !!sessionId,
    retry: false,
    refetchInterval: 5000,
  });
}

export function useApproveInterruptMutation() {
  return useMutation({
    mutationFn: async ({
      sessionId,
      approved,
    }: {
      sessionId: string;
      approved: boolean;
    }) =>
      apiClient(`/interrupt/${sessionId}/approve`, {
        method: 'POST',
        body: JSON.stringify({ approved }),
      }),
  });
}
