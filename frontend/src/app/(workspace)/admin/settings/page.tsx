'use client';

import { FormEvent, useMemo, useState } from 'react';
import { Shield, ShieldAlert } from 'lucide-react';
import { useCurrentUserQuery } from '@/domains/auth/hooks';
import {
  useAdminSettingsQuery,
  useUpdateAdminSettingsMutation,
} from '@/domains/settings/hooks';

export default function AdminSettingsPage() {
  const { data: currentUser } = useCurrentUserQuery();
  const isAdmin = currentUser?.role === 'admin';
  const { data, isLoading, isError } = useAdminSettingsQuery(isAdmin);
  const updateMutation = useUpdateAdminSettingsMutation();
  const [draftOverride, setDraftOverride] = useState<string | null>(null);
  const draft = useMemo(() => {
    if (draftOverride !== null) {
      return draftOverride;
    }
    if (data) {
      return JSON.stringify(data, null, 2);
    }
    return '{}';
  }, [data, draftOverride]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    try {
      const parsed = JSON.parse(draft) as Record<string, unknown>;
      updateMutation.mutate(parsed);
    } catch {
      alert('Settings must be valid JSON.');
    }
  };

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
            Global configuration
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
            {updateMutation.isSuccess && (
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200">
                Admin settings saved.
              </div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
