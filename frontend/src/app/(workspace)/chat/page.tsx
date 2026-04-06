'use client';

import Link from 'next/link';
import { useQueryClient } from '@tanstack/react-query';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { FormEvent, useDeferredValue, useMemo, useRef, useState } from 'react';
import { Bot, Clock3, LoaderCircle, MessageSquare, PlusCircle, RefreshCw, Search, Send, ShieldAlert, Workflow } from 'lucide-react';
import { ContextPruningSummary, formatPruningBlock } from '@/domains/chat/pruning';
import {
  ChatMessage,
  useApproveInterruptMutation,
  useChatInvokeMutation,
  useInterruptStatusQuery,
  useResumeInterruptMutation,
} from '@/domains/chat/hooks';
import {
  CONVERSATION_KEYS,
  type ConversationDTO,
  useConversationsQuery,
} from '@/domains/conversations/hooks';
import { formatMessage, useMessages } from '@/lib/i18n';
import { CHAT_MESSAGES } from './messages';

function createSessionId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

function humanizeToken(value: string) {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ');
}

function normalizeStoredMessages(messages?: ConversationDTO['messages']): ChatMessage[] {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages.map((message) => ({
    role: message.role === 'user' || message.role === 'system' ? message.role : 'assistant',
    content: String(message.content || ''),
  }));
}

function formatConversationTimestamp(value?: number | null) {
  if (!value) {
    return '';
  }
  const normalized = value > 1_000_000_000_000 ? value : value * 1000;
  return new Date(normalized).toLocaleString();
}

function getMessagePreview(content: string | undefined, fallback: string) {
  const compact = String(content || '').replace(/\s+/g, ' ').trim();
  if (!compact) {
    return fallback;
  }
  return compact.length > 88 ? `${compact.slice(0, 88)}...` : compact;
}

function getConversationDayBucket(value?: number | null) {
  if (!value) {
    return 'earlier';
  }

  const normalized = value > 1_000_000_000_000 ? value : value * 1000;
  const target = new Date(normalized);
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterdayStart = todayStart - 24 * 60 * 60 * 1000;
  const targetStart = new Date(target.getFullYear(), target.getMonth(), target.getDate()).getTime();

  if (targetStart >= todayStart) {
    return 'today';
  }
  if (targetStart >= yesterdayStart) {
    return 'yesterday';
  }
  return 'earlier';
}

export default function ChatPage() {
  const text = useMessages(CHAT_MESSAGES);
  const queryClient = useQueryClient();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const requestedSessionId = searchParams.get('session')?.trim() || null;
  const [sessionId, setSessionId] = useState(() => requestedSessionId || createSessionId());
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(() => requestedSessionId);
  const [draft, setDraft] = useState('');
  const [historySearch, setHistorySearch] = useState('');
  const deferredHistorySearch = useDeferredValue(historySearch);
  const [contextFocusHint, setContextFocusHint] = useState('');
  const [approvalComment, setApprovalComment] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [contextPruning, setContextPruning] = useState<ContextPruningSummary | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const requestTokenRef = useRef(0);

  const chatMutation = useChatInvokeMutation();
  const conversationsQuery = useConversationsQuery();
  const interruptQuery = useInterruptStatusQuery(sessionId);
  const approveMutation = useApproveInterruptMutation();
  const resumeMutation = useResumeInterruptMutation();

  const isChatPending = pendingSessionId === sessionId;
  const isApprovalPending = approveMutation.isPending || resumeMutation.isPending;

  const filteredConversations = useMemo(() => {
    const conversations = conversationsQuery.data ?? [];
    const keyword = deferredHistorySearch.trim().toLowerCase();
    if (!keyword) {
      return conversations;
    }

    return conversations.filter((conversation) => {
      const title = conversation.title?.toLowerCase() || '';
      const preview = conversation.messages[conversation.messages.length - 1]?.content?.toLowerCase() || '';
      return title.includes(keyword) || preview.includes(keyword) || conversation.id.toLowerCase().includes(keyword);
    });
  }, [conversationsQuery.data, deferredHistorySearch]);

  const activeConversation = (conversationsQuery.data ?? []).find((item) => item.id === selectedConversationId) || null;
  const displayedMessages =
    messages.length > 0 || !selectedConversationId || !activeConversation
      ? messages
      : normalizeStoredMessages(activeConversation.messages);
  const activeTitle =
    activeConversation?.title?.trim() ||
    (selectedConversationId
      ? formatMessage(text.sessionFallback, { sessionId: selectedConversationId.slice(0, 8) })
      : text.currentSession);

  const conversationSections = useMemo(() => {
    const sections = [
      { key: 'today', label: text.today, items: [] as ConversationDTO[] },
      { key: 'yesterday', label: text.yesterday, items: [] as ConversationDTO[] },
      { key: 'earlier', label: text.earlier, items: [] as ConversationDTO[] },
    ];

    for (const conversation of filteredConversations) {
      const bucket = getConversationDayBucket(conversation.updated_at);
      const section = sections.find((item) => item.key === bucket) || sections[2];
      section.items.push(conversation);
    }

    return sections.filter((section) => section.items.length > 0);
  }, [filteredConversations, text.earlier, text.today, text.yesterday]);

  const syncSessionQuery = (nextSessionId: string | null) => {
    const nextParams = new URLSearchParams(searchParams.toString());
    if (nextSessionId) {
      nextParams.set('session', nextSessionId);
    } else {
      nextParams.delete('session');
    }
    const nextQuery = nextParams.toString();
    router.replace(nextQuery ? `${pathname}?${nextQuery}` : pathname, { scroll: false });
  };

  const localizePruningValue = (value?: string | null) =>
    ({
      heuristic: text.heuristic,
      auto: text.autoMethod,
      reranker: text.reranker,
    }[value || ''] || value || text.heuristic);

  const localizeApprovalAction = (value?: string | null) =>
    ({
      resume: text.resumeApprovalAction,
      orchestration_review: text.orchestrationReviewAction,
      approval: text.approvalAction,
    }[value || ''] || (value ? humanizeToken(value) : text.pendingApproval));

  const startNewSession = () => {
    requestTokenRef.current += 1;
    setPendingSessionId(null);
    setSelectedConversationId(null);
    setSessionId(createSessionId());
    setDraft('');
    setHistorySearch('');
    setContextFocusHint('');
    setApprovalComment('');
    setContextPruning(null);
    setMessages([]);
    syncSessionQuery(null);
  };

  const handleSelectConversation = (conversationId: string) => {
    if (conversationId === selectedConversationId && sessionId === conversationId) {
      return;
    }
    const selectedConversation = (conversationsQuery.data ?? []).find((conversation) => conversation.id === conversationId);
    requestTokenRef.current += 1;
    setPendingSessionId(null);
    setSelectedConversationId(conversationId);
    setSessionId(conversationId);
    setMessages(normalizeStoredMessages(selectedConversation?.messages));
    setDraft('');
    setContextFocusHint('');
    setApprovalComment('');
    setContextPruning(null);
    syncSessionQuery(conversationId);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = draft.trim();
    if (!content || isChatPending) {
      return;
    }

    const requestSessionId = sessionId;
    const requestToken = requestTokenRef.current + 1;
    const nextMessages = [...displayedMessages, { role: 'user' as const, content }];

    requestTokenRef.current = requestToken;
    setPendingSessionId(requestSessionId);
    setContextPruning(null);
    setMessages(nextMessages);
    setDraft('');

    chatMutation.mutate(
      {
        sessionId: requestSessionId,
        messages: nextMessages,
        contextFocusHint,
      },
      {
        onSuccess: ({ messages: persistedMessages, reply, contextPruning: pruningSummary }) => {
          if (requestTokenRef.current !== requestToken) {
            return;
          }
          setPendingSessionId(null);
          setSelectedConversationId(requestSessionId);
          syncSessionQuery(requestSessionId);
          setMessages(
            reply
              ? persistedMessages
              : [
                  ...nextMessages,
                  {
                    role: 'assistant',
                    content: text.noReply,
                  },
                ]
          );
          setContextPruning(pruningSummary ?? null);
          void queryClient.invalidateQueries({ queryKey: CONVERSATION_KEYS.all });
          interruptQuery.refetch();
        },
        onError: (error) => {
          if (requestTokenRef.current !== requestToken) {
            return;
          }
          setPendingSessionId(null);
          setContextPruning(null);
          setMessages([
            ...nextMessages,
            {
              role: 'assistant',
              content:
                error instanceof Error
                  ? formatMessage(text.requestFailedWithReason, { message: error.message })
                  : text.requestFailed,
            },
          ]);
        },
      }
    );
  };

  const actionRequired = interruptQuery.data?.action_required;
  const candidateStats = formatPruningBlock(contextPruning?.candidatePruning);
  const promptDocStats = formatPruningBlock(contextPruning?.promptPruning?.docs);
  const promptMemoryStats = formatPruningBlock(contextPruning?.promptPruning?.memories);

  const appendAssistantMessage = (content: string) => {
    setMessages((current) =>
      current.length > 0 ? [...current, { role: 'assistant', content }] : [...displayedMessages, { role: 'assistant', content }]
    );
  };

  const finishInterruptedSessionResume = () => {
    resumeMutation.mutate(
      { sessionId },
      {
        onSuccess: ({ messages: resumedMessages, contextPruning: pruningSummary, reply }) => {
          if (resumedMessages.length > 0) {
            setMessages(resumedMessages);
          } else if (reply) {
            appendAssistantMessage(reply);
          }
          if (pruningSummary) {
            setContextPruning(pruningSummary);
          }
          setSelectedConversationId(sessionId);
          syncSessionQuery(sessionId);
          void queryClient.invalidateQueries({ queryKey: CONVERSATION_KEYS.all });
          interruptQuery.refetch();
        },
        onError: (error) => {
          appendAssistantMessage(
            error instanceof Error
              ? formatMessage(text.resumeFailedWithReason, { message: error.message })
              : text.resumeFailed
          );
          interruptQuery.refetch();
        },
      }
    );
  };

  const handleApproveAndResume = () => {
    if (!actionRequired) {
      return;
    }
    if (actionRequired.approved) {
      finishInterruptedSessionResume();
      return;
    }
    approveMutation.mutate(
      { sessionId, approved: true, comment: approvalComment },
      {
        onSuccess: () => {
          setApprovalComment('');
          finishInterruptedSessionResume();
        },
        onError: (error) => {
          appendAssistantMessage(
            error instanceof Error
              ? formatMessage(text.requestFailedWithReason, { message: error.message })
              : text.requestFailed
          );
          interruptQuery.refetch();
        },
      }
    );
  };

  const handleRejectInterrupt = () => {
    if (!interruptQuery.data?.interrupted || !actionRequired || isApprovalPending) {
      return;
    }
    approveMutation.mutate(
      { sessionId, approved: false, comment: approvalComment },
      {
        onSuccess: () => {
          setApprovalComment('');
          interruptQuery.refetch();
        },
        onError: (error) => {
          appendAssistantMessage(
            error instanceof Error
              ? formatMessage(text.requestFailedWithReason, { message: error.message })
              : text.requestFailed
          );
        },
      }
    );
  };

  const activeConversationShownInList = !!selectedConversationId && filteredConversations.some((item) => item.id === selectedConversationId);
  const currentPreview =
    displayedMessages.length > 0
      ? getMessagePreview(displayedMessages[displayedMessages.length - 1]?.content, text.noPreview)
      : text.newSessionDescription;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.14),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(250,204,21,0.12),_transparent_22%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_30%,#ffffff_100%)] p-3 lg:p-4">
      <div className="mx-auto grid max-w-[1900px] gap-4 xl:h-[calc(100dvh-32px)] xl:grid-cols-[300px_minmax(0,1fr)]">
        <aside className="flex min-h-[320px] flex-col overflow-hidden rounded-[30px] border border-slate-200 bg-white/88 shadow-[0_22px_70px_-48px_rgba(15,23,42,0.38)] backdrop-blur xl:min-h-0">
          <div className="border-b border-slate-200 px-5 py-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">{text.sessionHistory}</div>
                <h2 className="mt-2 text-lg font-semibold text-slate-950">{text.historyTitle}</h2>
                <p className="mt-1 text-sm text-slate-500">{text.historyDescription}</p>
              </div>
              <button
                type="button"
                onClick={startNewSession}
                className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-950 text-white transition hover:bg-slate-800"
                aria-label={text.newSession}
                title={text.newSession}
              >
                <PlusCircle className="h-5 w-5" />
              </button>
            </div>
            <label className="mt-4 flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              <Search className="h-4 w-4 text-slate-400" />
              <input
                value={historySearch}
                onChange={(event) => setHistorySearch(event.target.value)}
                placeholder={text.historySearchPlaceholder}
                className="w-full bg-transparent text-slate-900 outline-none placeholder:text-slate-400"
              />
            </label>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(248,250,252,0.92),rgba(255,255,255,0.92))] px-3 py-3">
            {!selectedConversationId ? (
              <button
                type="button"
                onClick={startNewSession}
                className="mb-3 w-full rounded-[24px] border border-slate-900 bg-slate-900 px-4 py-4 text-left text-white shadow-[0_18px_60px_-42px_rgba(15,23,42,0.75)]"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold">{text.currentSession}</div>
                  <span className="rounded-full bg-white/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100">
                    {text.liveSession}
                  </span>
                </div>
                <div className="mt-2 text-sm leading-6 text-slate-200">{currentPreview || text.newSessionDescription}</div>
                <div className="mt-3 text-xs text-slate-300">{formatMessage(text.sessionFallback, { sessionId: sessionId.slice(0, 8) })}</div>
              </button>
            ) : null}

            {selectedConversationId && !activeConversationShownInList ? (
              <button
                type="button"
                onClick={() => handleSelectConversation(selectedConversationId)}
                className="mb-3 w-full rounded-[24px] border border-slate-900 bg-slate-900 px-4 py-4 text-left text-white shadow-[0_18px_60px_-42px_rgba(15,23,42,0.75)]"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold">{activeTitle}</div>
                  <span className="rounded-full bg-white/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100">
                    {text.currentSession}
                  </span>
                </div>
                <div className="mt-2 text-sm leading-6 text-slate-200">{currentPreview}</div>
                <div className="mt-3 text-xs text-slate-300">{formatConversationTimestamp(activeConversation?.updated_at) || text.latestActivityPending}</div>
              </button>
            ) : null}

            {conversationsQuery.isLoading ? (
              <div className="flex items-center justify-center gap-2 px-4 py-10 text-sm text-slate-500">
                <LoaderCircle className="h-4 w-4 animate-spin" />
                {text.loadingHistory}
              </div>
            ) : conversationsQuery.isError ? (
              <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-5 text-sm text-rose-700">
                {text.historyLoadFailed}
              </div>
            ) : filteredConversations.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-sm text-slate-500">
                {deferredHistorySearch.trim() ? text.noSearchResult : text.noHistory}
              </div>
            ) : (
              <div className="space-y-5">
                {conversationSections.map((section) => (
                  <div key={section.key}>
                    <div className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                      {section.label}
                    </div>
                    <div className="space-y-3">
                      {section.items.map((conversation) => {
                        const isActive = conversation.id === selectedConversationId;
                        const preview = getMessagePreview(
                          conversation.messages[conversation.messages.length - 1]?.content,
                          text.noPreview
                        );
                        return (
                          <button
                            key={conversation.id}
                            type="button"
                            onClick={() => handleSelectConversation(conversation.id)}
                            className={`w-full rounded-[24px] border px-4 py-4 text-left transition ${
                              isActive
                                ? 'border-slate-900 bg-slate-900 text-white shadow-[0_18px_60px_-42px_rgba(15,23,42,0.75)]'
                                : 'border-slate-200 bg-white text-slate-900 hover:border-slate-300 hover:bg-slate-50'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div className="line-clamp-1 text-sm font-semibold">
                                {conversation.title?.trim() || formatMessage(text.sessionFallback, { sessionId: conversation.id.slice(0, 8) })}
                              </div>
                              <Clock3 className={`h-4 w-4 ${isActive ? 'text-slate-300' : 'text-slate-400'}`} />
                            </div>
                            <div className={`mt-2 line-clamp-2 text-sm leading-6 ${isActive ? 'text-slate-200' : 'text-slate-500'}`}>{preview}</div>
                            <div className={`mt-3 flex items-center justify-between gap-3 text-xs ${isActive ? 'text-slate-300' : 'text-slate-400'}`}>
                              <span>{formatConversationTimestamp(conversation.updated_at)}</span>
                              <span>{formatMessage(text.messagesCount, { count: conversation.messages.length })}</span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        <section className="flex min-w-0 flex-col overflow-hidden rounded-[30px] border border-slate-200 bg-white/88 shadow-[0_22px_70px_-48px_rgba(15,23,42,0.38)] backdrop-blur xl:min-h-0">
          <div className="border-b border-slate-200 px-5 py-5 lg:px-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-100">
                    <MessageSquare className="h-3.5 w-3.5" />
                    {text.title}
                  </div>
                  <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                    {selectedConversationId ? text.currentSession : text.liveSession}
                  </div>
                </div>
                <h1 className="mt-3 line-clamp-2 text-xl font-semibold text-slate-950 lg:text-2xl">{activeTitle}</h1>
                <div className="mt-3 flex flex-wrap gap-2">
                  <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700">
                    {text.turns}: {displayedMessages.length}
                  </div>
                  <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700">
                    {text.interruptState}:{' '}
                    {interruptQuery.isError
                      ? text.approvalStateUnavailable
                      : interruptQuery.data?.interrupted
                        ? text.approvalRequired
                        : text.clear}
                  </div>
                  <div className="max-w-full rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700">
                    <span className="text-slate-500">{text.activeFocus}: </span>
                    {contextPruning?.focus_hint || contextFocusHint || text.defaultFocus}
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Link
                  href="/harness"
                  className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  <Workflow className="h-4 w-4" />
                  {text.harness}
                </Link>
                <button
                  type="button"
                  onClick={startNewSession}
                  className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
                >
                  <PlusCircle className="h-4 w-4" />
                  {text.newSession}
                </button>
              </div>
            </div>

            <div className="mt-4 rounded-[24px] border border-slate-200 bg-slate-50/90 px-4 py-3 text-sm text-slate-500">
              {text.sessionId}: <code className="rounded bg-white px-2 py-1 font-mono text-xs text-slate-700 ring-1 ring-slate-200">{sessionId}</code>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(248,250,252,0.72),rgba(255,255,255,0.96)_18%,rgba(255,255,255,0.98)_100%)] px-5 py-5 lg:px-6">
            {displayedMessages.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <div className="max-w-2xl rounded-[32px] border border-dashed border-slate-300 bg-white/90 px-8 py-10 text-center shadow-[0_18px_60px_-46px_rgba(15,23,42,0.3)]">
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-slate-900 shadow-sm">
                    <Bot className="h-6 w-6" />
                  </div>
                  <h2 className="mt-5 text-2xl font-semibold text-slate-950">{text.firstTurnTitle}</h2>
                  <p className="mt-3 text-sm leading-7 text-slate-500">{text.firstTurn}</p>
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {displayedMessages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-4xl rounded-[28px] px-5 py-4 shadow-sm ${
                        message.role === 'user'
                          ? 'bg-slate-950 text-white'
                          : message.role === 'system'
                            ? 'border border-amber-200 bg-amber-50 text-amber-950'
                            : 'border border-slate-200 bg-white text-slate-900'
                      }`}
                    >
                      <div className={`mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] ${
                        message.role === 'user'
                          ? 'text-slate-300'
                          : message.role === 'system'
                            ? 'text-amber-700'
                            : 'text-slate-400'
                      }`}>
                        {message.role === 'user' ? <MessageSquare className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
                        {message.role === 'user' ? text.you : message.role === 'system' ? text.system : text.assistant}
                      </div>
                      <div className="whitespace-pre-wrap text-sm leading-7">{message.content}</div>
                    </div>
                  </div>
                ))}

                {isChatPending ? (
                  <div className="flex items-center gap-2 text-sm text-slate-500">
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    {text.waiting}
                  </div>
                ) : null}
              </div>
            )}
          </div>

          <div className="border-t border-slate-200 bg-white/92 px-5 py-5 backdrop-blur lg:px-6">
            {actionRequired ? (
              <div className="mb-4 rounded-[28px] border border-amber-200 bg-amber-50 p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-2xl bg-amber-100 p-2 text-amber-700">
                      <ShieldAlert className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-amber-950">
                        {localizeApprovalAction(actionRequired.action_type)}
                      </div>
                      <div className="mt-1 text-sm leading-6 text-amber-900">
                        {actionRequired.description || text.backendPaused}
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => interruptQuery.refetch()}
                    className="rounded-2xl border border-amber-200 bg-white p-2 text-amber-700 transition hover:bg-amber-100"
                    aria-label={text.refreshApproval}
                    title={text.refreshApproval}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>
                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    type="button"
                    disabled={isApprovalPending}
                    onClick={handleApproveAndResume}
                    className="inline-flex min-w-32 items-center justify-center rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
                  >
                    {isApprovalPending ? text.resuming : actionRequired.approved ? text.resumeApprovalAction : text.retryAutoResume}
                  </button>
                  {!actionRequired.approved ? (
                    <button
                      type="button"
                      disabled={isApprovalPending}
                      onClick={handleRejectInterrupt}
                      className="inline-flex min-w-32 items-center justify-center rounded-2xl border border-rose-200 bg-white px-4 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-50 disabled:opacity-50"
                    >
                      {text.reject}
                    </button>
                  ) : null}
                </div>
                {!actionRequired.approved ? (
                  <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.18em] text-amber-900">
                    {text.approvalComment}
                    <textarea
                      value={approvalComment}
                      onChange={(event) => setApprovalComment(event.target.value)}
                      rows={3}
                      placeholder={text.approvalCommentPlaceholder}
                      className="mt-2 w-full rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm font-normal text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-amber-300"
                    />
                  </label>
                ) : null}
                <div className="mt-3 text-sm leading-6 text-amber-900">
                  {actionRequired.approved ? text.approvalReady : text.autoReviewManaged}
                </div>
              </div>
            ) : null}

            {contextPruning ? (
              <div className="mb-4 rounded-[28px] border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Bot className="h-4 w-4 text-emerald-600" />
                  {text.contextPruning}
                </div>
                <div className="mt-4 grid gap-3 xl:grid-cols-4">
                  <div className="rounded-2xl bg-white px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.candidatePruning}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">{candidateStats.keptText}</div>
                    <div className="mt-1 text-xs text-slate-500">{candidateStats.itemsText ?? text.zeroItems}</div>
                  </div>
                  <div className="rounded-2xl bg-white px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.promptDocsKept}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">{promptDocStats.keptText}</div>
                    <div className="mt-1 text-xs text-slate-500">{text.candidateSaved}: {promptDocStats.savedText}</div>
                  </div>
                  <div className="rounded-2xl bg-white px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.promptMemoryKept}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">{promptMemoryStats.keptText}</div>
                    <div className="mt-1 text-xs text-slate-500">{text.promptMemorySaved}: {promptMemoryStats.savedText}</div>
                  </div>
                  <div className="rounded-2xl bg-white px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.pruningMethod}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">
                      {localizePruningValue(contextPruning.candidatePruning?.method || contextPruning.promptPruning?.method)}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {text.scoringSource}: {localizePruningValue(contextPruning.candidatePruning?.scoring_source || contextPruning.promptPruning?.scoring_source)}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            <form onSubmit={handleSubmit}>
              <label className="block text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                {text.goalHint}
                <input
                  value={contextFocusHint}
                  onChange={(event) => setContextFocusHint(event.target.value)}
                  placeholder={text.goalHintPlaceholder}
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-normal text-slate-900 outline-none transition focus:border-sky-300 focus:bg-white"
                />
              </label>

              <div className="mt-3 flex flex-col gap-3 lg:flex-row">
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  rows={4}
                  placeholder={text.promptPlaceholder}
                  className="min-h-32 flex-1 rounded-[28px] border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-sky-300 focus:bg-white"
                />
                <button
                  type="submit"
                  disabled={isChatPending || !draft.trim()}
                  className="inline-flex min-h-14 items-center justify-center gap-2 self-stretch rounded-[28px] bg-slate-950 px-6 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 lg:min-w-36"
                >
                  <Send className="h-4 w-4" />
                  {text.send}
                </button>
              </div>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}
