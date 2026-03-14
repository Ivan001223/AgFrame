'use client';

import { useMemoryProfileQuery } from '@/domains/memory/hooks';
import { BrainCircuit, Tag, Calendar, User, FileText } from 'lucide-react';

export default function MemoryPage() {
  const { data, isLoading, isError } = useMemoryProfileQuery();
  const profile = data?.profile as
    | {
        summary?: string;
        tags?: string[];
        updated_at?: string | number;
        facts?: Array<{ text?: string }>;
      }
    | null
    | undefined;
  const tags = Array.isArray(profile?.tags) ? profile.tags : [];
  const fallbackSummary = Array.isArray(profile?.facts)
    ? profile.facts.map((fact) => fact.text).filter(Boolean).join('\n')
    : '';

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">User Memory Profile</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Review the synthesized AI memory profile built from your past interactions.
        </p>
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-gray-500">Loading memory profile...</div>
      ) : isError ? (
        <div className="text-center py-12 text-red-500 bg-red-50 rounded-lg dark:bg-red-900/20 px-4">
          Failed to load memory profile. You may need to have more conversations first.
        </div>
      ) : !profile ? (
        <div className="text-center py-16 bg-white rounded-lg shadow-sm border border-gray-100 dark:bg-gray-800 dark:border-gray-700">
          <BrainCircuit className="mx-auto h-12 w-12 text-gray-300 dark:text-gray-600 mb-4" />
          <h3 className="text-sm font-medium text-gray-900 dark:text-white">No memory profile yet</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Interact with the AI to start building your unique memory profile.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden bg-white shadow sm:rounded-lg dark:bg-gray-800">
          <div className="px-4 py-5 sm:px-6 flex items-center gap-4 border-b border-gray-200 dark:border-gray-700">
            <div className="bg-fuchsia-100 p-3 rounded-lg dark:bg-fuchsia-900/40">
              <BrainCircuit className="h-8 w-8 text-fuchsia-600 dark:text-fuchsia-400" />
            </div>
            <div>
              <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white">
                Core Memory
              </h3>
              <p className="mt-1 max-w-2xl text-sm text-gray-500 dark:text-gray-400 flex items-center gap-2">
                <User className="h-4 w-4" />
                User ID: {data?.user_id || 'Current User'}
              </p>
            </div>
          </div>
          
          <div className="px-4 py-5 sm:p-0">
            <dl className="divide-y divide-gray-200 dark:divide-gray-700">
              <div className="py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Synthesized Summary
                </dt>
                <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white bg-gray-50 p-4 rounded-md dark:bg-gray-900/50 leading-relaxed border border-gray-100 dark:border-gray-700">
                  {profile?.summary || fallbackSummary || 'No summary generated yet.'}
                </dd>
              </div>

              <div className="py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center gap-2">
                  <Tag className="h-4 w-4" />
                  Behavioral Tags
                </dt>
                <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                  {tags.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {tags.map((tag, idx) => (
                        <span 
                          key={idx} 
                          className="inline-flex items-center rounded-md bg-fuchsia-50 px-2.5 py-1 text-sm font-medium text-fuchsia-800 ring-1 ring-inset ring-fuchsia-600/20 dark:bg-fuchsia-900/30 dark:text-fuchsia-300 dark:ring-fuchsia-500/30"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-gray-500 italic">No tags associated</span>
                  )}
                </dd>
              </div>

              <div className="py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Last Updated
                </dt>
                <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                  {profile?.updated_at ? new Date(profile.updated_at).toLocaleString() : 'Never'}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
