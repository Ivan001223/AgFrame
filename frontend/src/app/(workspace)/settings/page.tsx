'use client';

import { InlineNotice } from '@/components/feedback/InlineNotice';
import { useCurrentUserQuery } from '@/domains/auth/hooks';
import {
  getAdminBasicsConfig,
  getContextPruningConfig,
  getRerankerAvailability,
  getUserPreferenceSettings,
  type AdminBasicsConfig,
  type ContextPruningAdminConfig,
  type UserPreferencesConfig,
  useAdminSettingsQuery,
  useUpdateAdminSettingsMutation,
  useUpdateUserSettingsMutation,
  useUserSettingsQuery,
} from '@/domains/settings/hooks';
import { formatMessage, type AppLocale, useI18n, useMessages } from '@/lib/i18n';
import { saveStoredUserPreferences } from '@/lib/preferences';
import { TaskOperationsPanel } from '@/domains/tasks/TaskOperationsPanel';
import { Settings2, Shield, Sparkles } from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import { SETTINGS_MESSAGES } from './messages';

const INPUT_CLASSNAME =
  'mt-2 w-full rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-gray-700 dark:bg-gray-950 dark:text-white dark:focus:ring-indigo-900';

const PRUNING_NUMBER_FIELDS: Array<{
  key: Exclude<keyof ContextPruningAdminConfig, 'method'>;
  label: { en: string; zh: string };
  step?: string;
}> = [
  { key: 'min_keywords', label: { en: 'Min keywords', zh: '最少关键词' } },
  { key: 'auto_reranker_min_lines', label: { en: 'Auto reranker min lines', zh: '自动重排最少行数' } },
  { key: 'auto_reranker_min_chars', label: { en: 'Auto reranker min chars', zh: '自动重排最少字符数' } },
  { key: 'min_keep_lines', label: { en: 'Min keep lines', zh: '最少保留行数' } },
  { key: 'max_keep_ratio', label: { en: 'Max keep ratio', zh: '最大保留比例' }, step: '0.01' },
  { key: 'neighbor_window', label: { en: 'Neighbor window', zh: '邻接窗口' } },
  { key: 'reranker_window_radius', label: { en: 'Reranker window radius', zh: '重排窗口半径' } },
  { key: 'max_lines_per_item', label: { en: 'Max lines per item', zh: '每项最大行数' } },
  { key: 'score_threshold', label: { en: 'Score threshold', zh: '分数阈值' }, step: '0.01' },
] as const;

export default function SettingsPage() {
  const text = useMessages(SETTINGS_MESSAGES);
  const { locale, setLocale } = useI18n();
  const { data: currentUser, isLoading: isCurrentUserLoading } = useCurrentUserQuery();
  const isAdmin = currentUser?.role === 'admin';

  const userSettingsQuery = useUserSettingsQuery();
  const adminSettingsQuery = useAdminSettingsQuery(isAdmin);
  const updateUserMutation = useUpdateUserSettingsMutation();
  const updateAdminMutation = useUpdateAdminSettingsMutation();

  const userDefaults = useMemo(
    () => getUserPreferenceSettings(userSettingsQuery.data),
    [userSettingsQuery.data]
  );
  const adminDefaults = useMemo(
    () => getAdminBasicsConfig(adminSettingsQuery.data),
    [adminSettingsQuery.data]
  );
  const pruningDefaults = useMemo(
    () => getContextPruningConfig(adminSettingsQuery.data),
    [adminSettingsQuery.data]
  );
  const rerankerAvailability = useMemo(
    () => getRerankerAvailability(adminSettingsQuery.data),
    [adminSettingsQuery.data]
  );

  const [userDraft, setUserDraft] = useState<UserPreferencesConfig>(userDefaults);
  const [adminDraft, setAdminDraft] = useState<AdminBasicsConfig>(adminDefaults);
  const [pruningDraft, setPruningDraft] = useState<ContextPruningAdminConfig>(pruningDefaults);
  const [lastSavedSection, setLastSavedSection] = useState<'user' | 'admin' | 'pruning' | null>(null);

  useEffect(() => {
    setUserDraft(userDefaults);
  }, [userDefaults]);

  useEffect(() => {
    setAdminDraft(adminDefaults);
  }, [adminDefaults]);

  useEffect(() => {
    setPruningDraft(pruningDefaults);
  }, [pruningDefaults]);

  const handleUserSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateUserMutation.mutate(
      {
        ...(userSettingsQuery.data ?? {}),
        ...userDraft,
      },
      {
        onSuccess: () => {
          setLocale(userDraft.language as AppLocale);
          saveStoredUserPreferences(userDraft, currentUser?.username);
          setLastSavedSection('user');
        },
      }
    );
  };

  const handleAdminSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateAdminMutation.mutate(
      {
        general: {
          app_name: adminDraft.app_name,
        },
        llm: {
          model: adminDraft.llm_model,
        },
        local_models: {
          embedding_model: adminDraft.embedding_model,
          rerank_model: adminDraft.reranker_model,
        },
        search: {
          provider: adminDraft.search_provider,
        },
        auth: {
          access_token_expire_minutes: adminDraft.access_token_expire_minutes,
        },
        storage_local: {
          documents_dir: adminDraft.documents_dir,
          uploads_dir: adminDraft.uploads_dir,
        },
        feature_flags: {
          enable_docs_rag: adminDraft.enable_docs_rag,
          enable_chat_memory: adminDraft.enable_chat_memory,
        },
      },
      {
        onSuccess: () => setLastSavedSection('admin'),
      }
    );
  };

  const handlePruningSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateAdminMutation.mutate(
      {
        prompt: {
          context_pruning: {
            ...pruningDraft,
          },
        },
      },
      {
        onSuccess: () => setLastSavedSection('pruning'),
      }
    );
  };

  if (isCurrentUserLoading && !currentUser) {
    return <div className="p-8 text-sm text-gray-500">{text.loadingSettings}</div>;
  }

  return (
    <div className="mx-auto max-w-6xl p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{text.title}</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {text.description}
        </p>
      </div>

      <div className="grid gap-6">
        <section className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
          <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
              <Settings2 className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
              {text.userPreferences}
            </div>
          </div>

          {userSettingsQuery.isLoading ? (
            <div className="px-6 py-10 text-sm text-gray-500 dark:text-gray-400">
              {text.loadingPreferences}
            </div>
          ) : userSettingsQuery.isError ? (
            <div className="px-6 py-10 text-sm text-red-600 dark:text-red-400">
              {text.failedUserSettings}
            </div>
          ) : (
            <form onSubmit={handleUserSubmit} className="space-y-5 px-6 py-6">
              <div className="grid gap-4 md:grid-cols-2">
                <label className="text-sm text-gray-700 dark:text-gray-300">
                  <div className="font-medium">{text.interfaceLanguage}</div>
                  <select
                    value={userDraft.language}
                    onChange={(event) => {
                      const nextLocale = event.target.value as AppLocale;
                      setUserDraft((current) => ({ ...current, language: nextLocale }));
                      setLocale(nextLocale);
                    }}
                    className={INPUT_CLASSNAME}
                  >
                    <option value="en">{text.english}</option>
                    <option value="zh-CN">{text.simplifiedChinese}</option>
                  </select>
                </label>
                <label className="text-sm text-gray-700 dark:text-gray-300">
                  <div className="font-medium">{text.themePreference}</div>
                  <select
                    value={userDraft.theme}
                    onChange={(event) =>
                      setUserDraft((current) => ({
                        ...current,
                        theme: event.target.value as UserPreferencesConfig['theme'],
                      }))
                    }
                    className={INPUT_CLASSNAME}
                  >
                    <option value="system">{text.followSystem}</option>
                    <option value="light">{text.light}</option>
                    <option value="dark">{text.dark}</option>
                  </select>
                </label>
                <label className="text-sm text-gray-700 dark:text-gray-300">
                  <div className="font-medium">{text.preferredResponseLanguage}</div>
                  <select
                    value={userDraft.response_language}
                    onChange={(event) =>
                      setUserDraft((current) => ({
                        ...current,
                        response_language: event.target.value as UserPreferencesConfig['response_language'],
                      }))
                    }
                    className={INPUT_CLASSNAME}
                  >
                    <option value="auto">{text.auto}</option>
                    <option value="en">{text.english}</option>
                    <option value="zh-CN">{text.simplifiedChinese}</option>
                  </select>
                </label>
                <label className="text-sm text-gray-700 dark:text-gray-300">
                  <div className="font-medium">{text.defaultLandingPage}</div>
                  <select
                    value={userDraft.start_page}
                    onChange={(event) =>
                      setUserDraft((current) => ({
                        ...current,
                        start_page: event.target.value as UserPreferencesConfig['start_page'],
                      }))
                    }
                    className={INPUT_CLASSNAME}
                  >
                    <option value="/chat">{text.chat}</option>
                    <option value="/harness">{text.harness}</option>
                    <option value="/knowledge">{text.knowledge}</option>
                  </select>
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="flex items-center gap-3 rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-300">
                  <input
                    type="checkbox"
                    checked={userDraft.compact_mode}
                    onChange={(event) =>
                      setUserDraft((current) => ({ ...current, compact_mode: event.target.checked }))
                    }
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <span>{text.compactMode}</span>
                </label>
                <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-300">
                  {text.taskOpsMoved}
                </div>
              </div>

              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {text.languageSavedHint}
                </p>
                <button
                  type="submit"
                  disabled={updateUserMutation.isPending}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
                >
                  {updateUserMutation.isPending ? text.saving : text.savePreferences}
                </button>
              </div>

              {lastSavedSection === 'user' && updateUserMutation.isSuccess ? (
                <InlineNotice variant="success" message={text.preferencesSaved} />
              ) : null}
            </form>
          )}
        </section>

        <section id="task-operations" className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
          <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
              <Settings2 className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
              {text.taskOperations}
            </div>
          </div>
          <div className="px-6 py-6">
            <TaskOperationsPanel compact />
          </div>
        </section>

        {isAdmin ? (
          <>
            <section className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
              <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
                  <Shield className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                  {text.adminSettings}
                </div>
              </div>

              {adminSettingsQuery.isLoading ? (
                <div className="px-6 py-10 text-sm text-gray-500 dark:text-gray-400">
                  {text.loadingAdminSettings}
                </div>
              ) : adminSettingsQuery.isError ? (
                <div className="px-6 py-10 text-sm text-red-600 dark:text-red-400">
                  {text.failedAdminSettings}
                </div>
              ) : (
                <form onSubmit={handleAdminSubmit} className="space-y-5 px-6 py-6">
                  <div className={`rounded-xl border px-4 py-3 text-sm ${
                    rerankerAvailability.configured
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200'
                      : 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100'
                  }`}>
                    {rerankerAvailability.configured
                      ? formatMessage(text.rerankerConfigured, {
                          name: rerankerAvailability.modelName ?? text.unknown,
                        })
                      : text.rerankerNotConfigured}
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.appName}</div>
                      <input
                        value={adminDraft.app_name}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, app_name: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.defaultLlmModel}</div>
                      <input
                        value={adminDraft.llm_model}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, llm_model: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.embeddingModel}</div>
                      <input
                        value={adminDraft.embedding_model}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, embedding_model: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.rerankerModel}</div>
                      <input
                        value={adminDraft.reranker_model}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, reranker_model: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.searchProvider}</div>
                      <input
                        value={adminDraft.search_provider}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, search_provider: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.tokenExpiry}</div>
                      <input
                        type="number"
                        min={1}
                        value={adminDraft.access_token_expire_minutes}
                        onChange={(event) =>
                          setAdminDraft((current) => ({
                            ...current,
                            access_token_expire_minutes: Number(event.target.value) || 1,
                          }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.documentsDirectory}</div>
                      <input
                        value={adminDraft.documents_dir}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, documents_dir: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                    <label className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{text.uploadsDirectory}</div>
                      <input
                        value={adminDraft.uploads_dir}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, uploads_dir: event.target.value }))
                        }
                        className={INPUT_CLASSNAME}
                      />
                    </label>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <label className="flex items-center gap-3 rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-300">
                      <input
                        type="checkbox"
                        checked={adminDraft.enable_docs_rag}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, enable_docs_rag: event.target.checked }))
                        }
                        className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span>{text.enableDocsRag}</span>
                    </label>
                    <label className="flex items-center gap-3 rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-300">
                      <input
                        type="checkbox"
                        checked={adminDraft.enable_chat_memory}
                        onChange={(event) =>
                          setAdminDraft((current) => ({ ...current, enable_chat_memory: event.target.checked }))
                        }
                        className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span>{text.enableChatMemory}</span>
                    </label>
                    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100">
                      <div className="font-medium">{text.automatedReviewLock}</div>
                      <div className="mt-1 leading-6">{text.automatedReviewHint}</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {text.adminSettingsHint}
                    </p>
                    <button
                      type="submit"
                      disabled={updateAdminMutation.isPending}
                      className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
                    >
                      {updateAdminMutation.isPending ? text.saving : text.saveAdminSettings}
                    </button>
                  </div>

                  {lastSavedSection === 'admin' && updateAdminMutation.isSuccess ? (
                    <InlineNotice variant="success" message={text.adminSettingsSaved} />
                  ) : null}
                </form>
              )}
            </section>

            <section className="rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
              <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-white">
                  <Sparkles className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                  {text.contextPruning}
                </div>
              </div>

              <form onSubmit={handlePruningSubmit} className="space-y-5 px-6 py-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="text-sm text-gray-700 dark:text-gray-300">
                    <div className="font-medium">{text.method}</div>
                    <select
                      value={pruningDraft.method}
                      onChange={(event) =>
                        setPruningDraft((current) => ({ ...current, method: event.target.value }))
                      }
                      className={INPUT_CLASSNAME}
                    >
                      <option value="heuristic">{text.heuristic}</option>
                      <option value="auto">{text.autoMethod}</option>
                      <option value="reranker">{text.rerankerMethod}</option>
                    </select>
                  </label>

                  {PRUNING_NUMBER_FIELDS.map((field) => (
                    <label key={field.key} className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium">{locale === 'zh-CN' ? field.label.zh : field.label.en}</div>
                      <input
                        type="number"
                        step={field.step}
                        value={pruningDraft[field.key]}
                        onChange={(event) =>
                          setPruningDraft((current) => ({
                            ...current,
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
                      {text.pruningHint}
                    </p>
                  <button
                    type="submit"
                    disabled={updateAdminMutation.isPending}
                    className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
                  >
                    {updateAdminMutation.isPending ? text.saving : text.savePruningSettings}
                  </button>
                </div>

                {lastSavedSection === 'pruning' && updateAdminMutation.isSuccess ? (
                  <InlineNotice variant="success" message={text.pruningSettingsSaved} />
                ) : null}
              </form>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );
}
