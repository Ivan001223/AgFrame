'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { ArrowLeft, MessageSquare } from 'lucide-react';
import { useConversationDetailQuery } from '@/domains/conversations/hooks';

export default function ConversationDetailPage() {
  const params = useParams();
  const conversationId = params.conversationId as string;
  const { data, isLoading, isError } = useConversationDetailQuery(conversationId);

  if (isLoading) {
    return <div className="p-8 text-gray-500">Loading conversation...</div>;
  }

  if (isError || !data) {
    return <div className="p-8 text-red-500">Failed to load conversation.</div>;
  }

  return (
    <div className="mx-auto max-w-5xl p-8">
      <Link
        href="/conversations"
        className="mb-6 inline-flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to conversations
      </Link>

      <div className="rounded-xl bg-white p-6 shadow dark:bg-gray-800">
        <div className="border-b border-gray-200 pb-4 dark:border-gray-700">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {data.title || 'Untitled Conversation'}
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Updated {new Date(data.updated_at * 1000).toLocaleString()}
          </p>
        </div>

        <div className="mt-6 space-y-4">
          {data.messages.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400">
              No messages in this conversation.
            </div>
          ) : (
            data.messages.map((message, index) => (
              <div
                key={`${message.created_at}-${index}`}
                className={`rounded-xl p-4 ${
                  message.role === 'user'
                    ? 'bg-indigo-50 dark:bg-indigo-950/30'
                    : 'bg-gray-50 dark:bg-gray-900'
                }`}
              >
                <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  <MessageSquare className="h-4 w-4" />
                  {message.role}
                </div>
                <div className="whitespace-pre-wrap text-sm leading-6 text-gray-900 dark:text-gray-100">
                  {message.content}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
