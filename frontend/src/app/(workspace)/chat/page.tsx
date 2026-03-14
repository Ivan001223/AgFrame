'use client';

import { FormEvent, useMemo, useState } from 'react';
import { Bot, LoaderCircle, RefreshCw, Send, ShieldAlert } from 'lucide-react';
import {
  ChatMessage,
  useApproveInterruptMutation,
  useChatInvokeMutation,
  useInterruptStatusQuery,
} from '@/domains/chat/hooks';

function createSessionId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

const STARTER_MESSAGE: ChatMessage = {
  role: 'assistant',
  content: 'Ask a question to start a new workbench session.',
};

export default function ChatPage() {
  const [sessionId, setSessionId] = useState(createSessionId);
  const [draft, setDraft] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([STARTER_MESSAGE]);

  const chatMutation = useChatInvokeMutation();
  const interruptQuery = useInterruptStatusQuery(sessionId);
  const approveMutation = useApproveInterruptMutation();

  const timeline = useMemo(
    () => (messages.length === 1 && messages[0] === STARTER_MESSAGE ? [] : messages),
    [messages]
  );

  const startNewSession = () => {
    setSessionId(createSessionId());
    setDraft('');
    setMessages([STARTER_MESSAGE]);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = draft.trim();
    if (!content || chatMutation.isPending) {
      return;
    }

    const baseMessages = messages.length === 1 && messages[0] === STARTER_MESSAGE ? [] : messages;
    const nextMessages = [...baseMessages, { role: 'user' as const, content }];

    setMessages(nextMessages);
    setDraft('');

    chatMutation.mutate(
      {
        sessionId,
        messages: nextMessages,
      },
      {
        onSuccess: ({ messages: persistedMessages, reply }) => {
          setMessages(
            reply
              ? persistedMessages
              : [
                  ...nextMessages,
                  {
                    role: 'assistant',
                    content: 'No response body returned from the backend.',
                  },
                ]
          );
          interruptQuery.refetch();
        },
        onError: (error) => {
          setMessages([
            ...nextMessages,
            {
              role: 'assistant',
              content:
                error instanceof Error
                  ? `Request failed: ${error.message}`
                  : 'Request failed.',
            },
          ]);
        },
      }
    );
  };

  const actionRequired = interruptQuery.data?.action_required;

  return (
    <div className="mx-auto grid max-w-7xl gap-6 p-8 lg:grid-cols-[minmax(0,1fr)_320px]">
      <section className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 dark:border-gray-800">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Chat Workbench
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Session ID: <code className="font-mono">{sessionId}</code>
            </p>
          </div>
          <button
            type="button"
            onClick={startNewSession}
            className="rounded-md border border-gray-200 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
          >
            New session
          </button>
        </div>

        <div className="space-y-4 px-6 py-6">
          {timeline.length === 0 ? (
            <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-8 text-center text-sm text-gray-500 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-400">
              Ask a question to create the first turn.
            </div>
          ) : (
            timeline.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`rounded-2xl px-4 py-3 text-sm shadow-sm ${
                  message.role === 'user'
                    ? 'ml-auto max-w-3xl bg-indigo-600 text-white'
                    : 'max-w-3xl border border-gray-200 bg-gray-50 text-gray-900 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100'
                }`}
              >
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide opacity-70">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div className="whitespace-pre-wrap leading-6">{message.content}</div>
              </div>
            ))
          )}

          {chatMutation.isPending && (
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Waiting for backend response...
            </div>
          )}
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-200 px-6 py-4 dark:border-gray-800"
        >
          <div className="flex gap-3">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={3}
              placeholder="Ask the workbench to research, summarize, or act on uploaded knowledge."
              className="min-h-24 flex-1 rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm text-gray-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-gray-700 dark:bg-gray-950 dark:text-white dark:focus:ring-indigo-900"
            />
            <button
              type="submit"
              disabled={chatMutation.isPending || !draft.trim()}
              className="inline-flex items-center gap-2 self-end rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
              Send
            </button>
          </div>
        </form>
      </section>

      <aside className="space-y-4">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
            <Bot className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
            Session diagnostics
          </div>
          <dl className="mt-4 space-y-3 text-sm">
            <div>
              <dt className="text-gray-500 dark:text-gray-400">Turns</dt>
              <dd className="mt-1 text-gray-900 dark:text-white">{timeline.length}</dd>
            </div>
            <div>
              <dt className="text-gray-500 dark:text-gray-400">Interrupt state</dt>
              <dd className="mt-1 text-gray-900 dark:text-white">
                {interruptQuery.isError
                  ? 'No pending interrupt'
                  : interruptQuery.data?.interrupted
                    ? 'Approval required'
                    : 'Clear'}
              </dd>
            </div>
          </dl>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
              <ShieldAlert className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              Human approval
            </div>
            <button
              type="button"
              onClick={() => interruptQuery.refetch()}
              className="rounded-md p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>

          {!actionRequired ? (
            <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
              No approval checkpoint is waiting for this session.
            </p>
          ) : (
            <div className="mt-4 space-y-4">
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100">
                <div className="font-medium">
                  {actionRequired.action_type || 'Pending approval'}
                </div>
                <div className="mt-2">
                  {actionRequired.description ||
                    'The backend paused this session for confirmation.'}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={approveMutation.isPending}
                  onClick={() =>
                    approveMutation.mutate(
                      { sessionId, approved: true },
                      { onSuccess: () => interruptQuery.refetch() }
                    )
                  }
                  className="flex-1 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  Approve
                </button>
                <button
                  type="button"
                  disabled={approveMutation.isPending}
                  onClick={() =>
                    approveMutation.mutate(
                      { sessionId, approved: false },
                      { onSuccess: () => interruptQuery.refetch() }
                    )
                  }
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
                >
                  Reject
                </button>
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
