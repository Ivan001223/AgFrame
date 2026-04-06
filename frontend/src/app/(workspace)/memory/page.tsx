'use client';

import { BrainCircuit, Calendar, FileText, Tag, User } from 'lucide-react';
import { useMemoryProfileQuery } from '@/domains/memory/hooks';
import { useMessages } from '@/lib/i18n';
import { MEMORY_MESSAGES } from './messages';

export default function MemoryPage() {
  const text = useMessages(MEMORY_MESSAGES);
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
    <div className="mx-auto max-w-5xl p-6 lg:p-8">
      <div className="grid gap-4">
        <section className="rounded-[12px] border border-slate-200 bg-white p-6 shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-[12px] bg-blue-50 text-blue-700">
              <BrainCircuit className="h-6 w-6" />
            </div>
            <div className="min-w-0">
              <h1 className="text-2xl font-semibold text-slate-950">{text.title}</h1>
              <p className="mt-2 text-sm leading-6 text-slate-500">{text.description}</p>
            </div>
          </div>
        </section>

        {isLoading ? (
          <section className="rounded-[12px] border border-slate-200 bg-white px-6 py-16 text-center text-sm text-slate-500 shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
            {text.loading}
          </section>
        ) : isError ? (
          <section className="rounded-[12px] border border-rose-200 bg-rose-50 px-6 py-16 text-center text-sm text-rose-600 shadow-[0_12px_36px_-28px_rgba(15,23,42,0.2)]">
            {text.failed}
          </section>
        ) : !profile ? (
          <section className="rounded-[12px] border border-slate-200 bg-white px-6 py-16 text-center shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
            <BrainCircuit className="mx-auto h-12 w-12 text-slate-300" />
            <h2 className="mt-4 text-base font-semibold text-slate-900">{text.emptyTitle}</h2>
            <p className="mt-2 text-sm text-slate-500">{text.emptyDescription}</p>
          </section>
        ) : (
          <section className="overflow-hidden rounded-[12px] border border-slate-200 bg-white shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
            <div className="border-b border-slate-200 px-6 py-5">
              <div className="flex flex-wrap items-center gap-3">
                <div className="rounded-[12px] bg-slate-50 p-3 text-blue-700">
                  <BrainCircuit className="h-6 w-6" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-slate-950">{text.coreMemory}</div>
                  <div className="mt-1 flex items-center gap-2 text-sm text-slate-500">
                    <User className="h-4 w-4" />
                    {text.userId}: {data?.user_id || text.currentUser}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-4 p-6">
              <div className="rounded-[12px] border border-slate-200 bg-slate-50 p-5">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <FileText className="h-4 w-4 text-slate-500" />
                  {text.summary}
                </div>
                <div className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">
                  {profile?.summary || fallbackSummary || text.noSummary}
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
                <div className="rounded-[12px] border border-slate-200 bg-white p-5">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                    <Tag className="h-4 w-4 text-slate-500" />
                    {text.tags}
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {tags.length > 0 ? (
                      tags.map((tag, index) => (
                        <span
                          key={`${tag}-${index}`}
                          className="inline-flex items-center rounded-[999px] bg-blue-50 px-3 py-1 text-sm font-medium text-blue-700 ring-1 ring-inset ring-blue-200"
                        >
                          {tag}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm italic text-slate-500">{text.noTags}</span>
                    )}
                  </div>
                </div>

                <div className="rounded-[12px] border border-slate-200 bg-white p-5">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                    <Calendar className="h-4 w-4 text-slate-500" />
                    {text.lastUpdated}
                  </div>
                  <div className="mt-4 text-sm text-slate-700">
                    {profile?.updated_at ? new Date(profile.updated_at).toLocaleString() : text.never}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
