'use client';

import { InlineNotice } from '@/components/feedback/InlineNotice';
import { FormEvent, useMemo, useState } from 'react';
import { Settings2 } from 'lucide-react';
import {
  useUpdateUserSettingsMutation,
  useUserSettingsQuery,
} from '@/domains/settings/hooks';

export default function SettingsPage() {
  const { data, isLoading, isError } = useUserSettingsQuery();
  const updateMutation = useUpdateUserSettingsMutation();
  const [draftOverride, setDraftOverride] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const draft = useMemo(() => {
    if (draftOverride !== null) {
      return draftOverride;
    }
    if (data) {
      return JSON.stringify(data, null, 2);
    }
    return '{\n  "theme": "system"\n}';
  }, [data, draftOverride]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setValidationError(null);

    try {
      const parsed = JSON.parse(draft) as Record<string, unknown>;
      updateMutation.mutate(parsed);
    } catch {
      setValidationError('Settings must be valid JSON.');
    }
  };

  return (
    <div className="mx-auto max-w-4xl p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          User Settings
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          These preferences are stored per user through{' '}
          <code className="font-mono">/settings/user</code>.
        </p>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
        <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
            <Settings2 className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
            Preference payload
          </div>
        </div>

        {isLoading ? (
          <div className="px-6 py-10 text-sm text-gray-500 dark:text-gray-400">
            Loading settings...
          </div>
        ) : isError ? (
          <div className="px-6 py-10 text-sm text-red-600 dark:text-red-400">
            Failed to load user settings.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4 px-6 py-6">
            {validationError && (
              <InlineNotice
                variant="error"
                message={validationError}
                onDismiss={() => setValidationError(null)}
              />
            )}
            <textarea
              value={draft}
              onChange={(event) => setDraftOverride(event.target.value)}
              rows={18}
              className="w-full rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 font-mono text-sm text-gray-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-gray-700 dark:bg-gray-950 dark:text-white dark:focus:ring-indigo-900"
            />
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Store any JSON object your backend profile settings should remember.
              </p>
              <button
                type="submit"
                disabled={updateMutation.isPending}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                {updateMutation.isPending ? 'Saving...' : 'Save settings'}
              </button>
            </div>
            {updateMutation.isSuccess && (
              <InlineNotice variant="success" message="Settings saved." />
            )}
          </form>
        )}
      </div>
    </div>
  );
}
