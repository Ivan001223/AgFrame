'use client';

import { useConversationsQuery } from '@/domains/conversations/hooks';
import { MessageSquare, Calendar, ChevronRight } from 'lucide-react';
import Link from 'next/link';

export default function ConversationsPage() {
  const { data: conversations, isLoading, isError } = useConversationsQuery();

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Conversations</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Review your past AI chat sessions and memory snapshots.
        </p>
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-gray-500">Loading conversations...</div>
      ) : isError ? (
        <div className="text-center py-12 text-red-500 bg-red-50 rounded-lg dark:bg-red-900/20">
          Failed to load conversations. Please try again later.
        </div>
      ) : !conversations || conversations.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-lg shadow-sm border border-gray-100 dark:bg-gray-800 dark:border-gray-700">
          <MessageSquare className="mx-auto h-12 w-12 text-gray-300 dark:text-gray-600 mb-4" />
          <h3 className="text-sm font-medium text-gray-900 dark:text-white">No conversations</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Get started by creating a new chat session in the workspace.
          </p>
          <div className="mt-6">
            <Link
              href="/knowledge"
              className="inline-flex items-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              Upload a document
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {conversations.map((conv) => (
            <Link 
              key={conv.id} 
              href={`/conversations/${conv.id}`}
              className="group block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-indigo-300 transition-all dark:bg-gray-800 dark:border-gray-700 dark:hover:border-indigo-700"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="bg-indigo-50 p-2 rounded-lg group-hover:bg-indigo-100 transition-colors dark:bg-indigo-900/40 dark:group-hover:bg-indigo-800/60">
                    <MessageSquare className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white line-clamp-1">
                    {conv.title || 'Untitled Conversation'}
                  </h3>
                </div>
                <ChevronRight className="h-5 w-5 text-gray-400 group-hover:text-indigo-500 transition-colors" />
              </div>
              
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                {conv.messages[0]?.content || 'No preview available.'}
              </p>
              
              <div className="mt-6 flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  <span>{new Date(conv.updated_at * 1000).toLocaleDateString()}</span>
                </div>
                {typeof conv.messages.length === 'number' && (
                  <div className="px-2 py-1 bg-gray-100 rounded-full dark:bg-gray-700">
                    {conv.messages.length} messages
                  </div>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
