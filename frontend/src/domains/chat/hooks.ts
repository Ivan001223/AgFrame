import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { extractContextPruning } from '@/domains/chat/pruning';

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

export type ResumeInterruptDTO = {
  session_id: string;
  resumed: boolean;
  interrupted?: boolean | null;
  reply?: string | null;
  messages?: Array<{
    role?: string;
    content?: unknown;
  }>;
  context?: Record<string, unknown> | null;
};

export type ChatInvokeDTO = {
  session_id: string;
  interrupted?: boolean | null;
  reply?: string | null;
  messages?: Array<{
    role?: string;
    content?: unknown;
  }>;
  context?: Record<string, unknown> | null;
};

type ChatInvokeParams = {
  sessionId: string;
  messages: ChatMessage[];
  contextFocusHint?: string;
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

function normalizeChatMessages(value: ResumeInterruptDTO['messages']): ChatMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((message) => {
    const role = message?.role === 'user' || message?.role === 'system' ? message.role : 'assistant';
    return {
      role,
      content: normalizeMessageContent(message?.content),
    };
  });
}

export function useChatInvokeMutation() {
  return useMutation({
    mutationFn: async ({ sessionId, messages, contextFocusHint }: ChatInvokeParams) => {
      const payload = await apiClient<ChatInvokeDTO>('/chat/workbench-invoke', {
        method: 'POST',
        body: JSON.stringify({
          input: {
            messages,
            context: {
              session_id: sessionId,
              context_focus_hint: contextFocusHint?.trim() || undefined,
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

      const persistedMessages = normalizeChatMessages(payload.messages);
      const reply = (payload.reply || extractAssistantReply({ messages: persistedMessages })).trim();
      const nextMessages = persistedMessages.length
        ? persistedMessages
        : reply
          ? [...messages, { role: 'assistant' as const, content: reply }]
          : messages;

      return {
        payload,
        reply,
        messages: nextMessages,
        contextPruning: extractContextPruning(payload),
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

export function useResumeInterruptMutation() {
  return useMutation({
    mutationFn: async ({ sessionId }: { sessionId: string }) => {
      const payload = await apiClient<ResumeInterruptDTO>(`/interrupt/${sessionId}/resume`, {
        method: 'POST',
        timeout: 60000,
      });
      const messages = normalizeChatMessages(payload.messages);
      return {
        payload,
        reply: (payload.reply || extractAssistantReply({ messages })).trim(),
        messages,
        contextPruning: extractContextPruning({ context: payload.context }),
      };
    },
  });
}
