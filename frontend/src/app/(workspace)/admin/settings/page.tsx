'use client';

import { InlineNotice } from '@/components/feedback/InlineNotice';
import { FormEvent, useMemo, useState } from 'react';
import { Shield, ShieldAlert } from 'lucide-react';
import { useCurrentUserQuery } from '@/domains/auth/hooks';
import {
  ContextPruningAdminConfig,
  getContextPruningConfig,
  getRerankerAvailability,
  useAdminSettingsQuery,
  useUpdateAdminSettingsMutation,
} from '@/domains/settings/hooks';

const PRUNING_NUMBER_FIELDS: Array<{
  key: Exclude<keyof ContextPruningAdminConfig, 'method'>;
  label: string;
  step?: string;
  className?: string;
}> = [
  { key: 'min_keywords', label: 'Min keywords' },
  { key: 'auto_reranker_min_lines', label: 'Auto reranker min lines' },
  { key: 'auto_reranker_min_chars', label: 'Auto reranker min chars' },
  { key: 'min_keep_lines', label: 'Min keep lines' },
  { key: 'max_keep_ratio', label: 'Max keep ratio', step: '0.01' },
  { key: 'neighbor_window', label: 'Neighbor window' },
  { key: 'reranker_window_radius', label: 'Reranker window radius' },
  { key: 'max_lines_per_item', label: 'Max lines per item' },
  { key: 'score_threshold', label: 'Score threshold', step: '0.01', className: 'md:col-span-2' },
];

const INPUT_CLASSNAME =
  'w-full rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-gray-700 dark:bg-gray-950 dark:text-white dark:focus:ring-indigo-900';

export default function AdminSettingsPage() {
  const { data: currentUser, isLoading: isCurrentUserLoading } = useCurrentUserQuery();
  const isAdmin = currentUser?.role === 'admin';
  const { data, isLoading, isError } = useAdminSettingsQuery(isAdmin);
  const updateMutation = useUpdateAdminSettingsMutation();
  const basePruningDraft = useMemo(() => getContextPruningConfig(data), [data]);
  const [pruningDraftOverride, setPruningDraftOverride] =
    useState<ContextPruningAdminConfig | null>(null);
  const [draftOverride, setDraftOverride] = useState<string | null>(null);
  const [lastSavedSection, setLastSavedSection] = useState<'json' | 'pruning' | null>(null);
  const [jsonValidationError, setJsonValidationError] = useState<string | null>(null);
  const pruningDraft = pruningDraftOverride ?? basePruningDraft;
  const draft = useMemo(() => {
    if (draftOverride !== null) {
      return draftOverride;
    }
    if (data) {
      return JSON.stringify(data, null, 2);
    }
    return '{}';
  }, [data, draftOverride]);
  const rerankerAvailability = useMemo(() => getRerankerAvailability(data), [data]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setJsonValidationError(null);

    try {
      const parsed = JSON.parse(draft) as Record<string, unknown>;
      updateMutation.mutate(parsed, {
        onSuccess: () => {
          setDraftOverride(null);
          setLastSavedSection('json');
        },
        onError: () => {
          setLastSavedSection(null);
        },
      });
    } catch {
      setJsonValidationError('Settings must be valid JSON.');
    }
  };

  const handlePruningSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateMutation.mutate({
      prompt: {
        context_pruning: {
          ...pruningDraft,
        },
      },
    }, {
      onSuccess: () => {
        setPruningDraftOverride(null);
        setLastSavedSection('pruning');
      },
      onError: () => {
        setLastSavedSection(null);
      },
    });
  };

  if (isCurrentUserLoading) {
    return (
      <div className="mx-auto max-w-3xl p-8">
        <div className="rounded-2xl border border-gray-200 bg-white p-6 text-sm text-gray-500 shadow-sm dark:border-gray-800 dark:bg-gray-900 dark:text-gray-400">
          Loading admin access...
        </div>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="mx-auto max-w-3xl p-8">
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-6 text-amber-900 shadow-sm dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100">
          <div className="flex items-center gap-3 text-lg font-semibold">
            <ShieldAlert className="h-5 w-5" />
            Admin access required
          </div>
          <p className="mt-3 text-sm">
            The backend protects <code className="font-mono">/settings</code>{' '}
            with an admin guard.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Admin Settings
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Edit the global server configuration exposed by{' '}
          <code className="font-mono">/settings</code>.
        </p>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
        <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
            <Shield className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
            Context pruning controls
          </div>
        </div>

        {isLoading ? (
          <div className="px-6 py-10 text-sm text-gray-500 dark:text-gray-400">
            Loading admin settings...
          </div>
        ) : isError ? (
          <div className="px-6 py-10 text-sm text-red-600 dark:text-red-400">
            Failed to load admin settings.
          </div>
        ) : (
          <form onSubmit={handlePruningSubmit} className="space-y-4 px-6 py-6">
            <div className={`rounded-xl border px-4 py-3 text-sm ${
              rerankerAvailability.configured
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200'
                : 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100'
            }`}>
              {rerankerAvailability.configured ? (
                <>
                  Reranker model configured: <code className="font-mono">{rerankerAvailability.modelName}</code>.
                  Active pruning source: <code className="font-mono">{rerankerAvailability.scoringSource ?? 'reranker_model'}</code>.
                </>
              ) : (
                <>
                  No reranker model configured. <code className="font-mono">reranker</code> and long-context <code className="font-mono">auto</code> paths will use <code className="font-mono">{rerankerAvailability.scoringSource ?? 'local_phrase_fallback'}</code>.
                </>
              )}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <label className="text-sm text-gray-700 dark:text-gray-300">
                <div className="mb-1 font-medium">Method</div>
                <select
                  value={pruningDraft.method}
                  onChange={(event) =>
                    setPruningDraftOverride((current) => ({
                      ...(current ?? basePruningDraft),
                      method: event.target.value,
                    }))
                  }
                  className={INPUT_CLASSNAME}
                >
                  <option value="heuristic">heuristic</option>
                  <option value="auto">auto</option>
                  <option value="reranker">reranker</option>
                </select>
              </label>
              {PRUNING_NUMBER_FIELDS.map((field) => (
                <label
                  key={field.key}
                  className={`text-sm text-gray-700 dark:text-gray-300 ${field.className ?? ''}`}
                >
                  <div className="mb-1 font-medium">{field.label}</div>
                  <input
                    type="number"
                    step={field.step}
                    value={pruningDraft[field.key]}
                    onChange={(event) =>
                      setPruningDraftOverride((current) => ({
                        ...(current ?? basePruningDraft),
                        [field.key]: Number(event.target.value),
                      }))
                    }
                    className={INPUT_CLASSNAME}
                  />
                </label>
              ))}
            </div>
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Saves only <code className="font-mono">prompt.context_pruning</code>.
              </p>
              <button
                type="submit"
                disabled={updateMutation.isPending}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                {updateMutation.isPending ? 'Saving...' : 'Save pruning config'}
              </button>
            </div>
            {updateMutation.isSuccess && lastSavedSection === 'pruning' && (
              <InlineNotice variant="success" message="Context pruning settings saved." />
            )}
          </form>
        )}
      </div>

      <div className="mt-6 rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
        <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
            <Shield className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
            Global configuration JSON
          </div>
        </div>

        {isLoading ? (
          <div className="px-6 py-10 text-sm text-gray-500 dark:text-gray-400">
            Loading admin settings...
          </div>
        ) : isError ? (
          <div className="px-6 py-10 text-sm text-red-600 dark:text-red-400">
            Failed to load admin settings.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4 px-6 py-6">
            {jsonValidationError && (
              <InlineNotice
                variant="error"
                message={jsonValidationError}
                onDismiss={() => setJsonValidationError(null)}
              />
            )}
            <textarea
              value={draft}
              onChange={(event) => setDraftOverride(event.target.value)}
              rows={24}
              className="w-full rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 font-mono text-sm text-gray-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-gray-700 dark:bg-gray-950 dark:text-white dark:focus:ring-indigo-900"
            />
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                This writes nested config values back through the FastAPI admin endpoint.
              </p>
              <button
                type="submit"
                disabled={updateMutation.isPending}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                {updateMutation.isPending ? 'Saving...' : 'Save admin settings'}
              </button>
            </div>
            {updateMutation.isSuccess && lastSavedSection === 'json' && (
              <InlineNotice variant="success" message="Admin settings saved." />
            )}
          </form>
        )}
      </div>
    </div>
  );
}
