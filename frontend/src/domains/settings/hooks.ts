import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getSessionCacheScope } from '@/lib/auth/session';

export type UserSettingsDTO = Record<string, unknown>;
export type AdminSettingsDTO = Record<string, unknown>;
export type UserStartPage = '/chat' | '/harness' | '/knowledge';
export type UserPreferencesConfig = {
  language: 'en' | 'zh-CN';
  theme: 'system' | 'light' | 'dark';
  response_language: 'auto' | 'en' | 'zh-CN';
  start_page: UserStartPage;
  compact_mode: boolean;
};

export type AdminBasicsConfig = {
  app_name: string;
  llm_model: string;
  embedding_model: string;
  reranker_model: string;
  search_provider: string;
  access_token_expire_minutes: number;
  documents_dir: string;
  uploads_dir: string;
  enable_docs_rag: boolean;
  enable_chat_memory: boolean;
};

type RuntimeRerankerStatusDTO = {
  configured?: boolean;
  model_name?: string | null;
  pruning_scoring_source?: string;
};

type RuntimeStatusDTO = {
  reranker?: RuntimeRerankerStatusDTO;
};

export type ContextPruningAdminConfig = {
  method: string;
  auto_reranker_min_lines: number;
  auto_reranker_min_chars: number;
  min_keywords: number;
  min_keep_lines: number;
  max_keep_ratio: number;
  neighbor_window: number;
  reranker_window_radius: number;
  max_lines_per_item: number;
  score_threshold: number;
};

export type RerankerAvailability = {
  configured: boolean;
  modelName: string | null;
  scoringSource: string | null;
};

export const SETTINGS_KEYS = {
  user: (scope: string) => ['settings', scope, 'user'] as const,
  admin: (scope: string) => ['settings', scope, 'admin'] as const,
};

export function useUserSettingsQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: SETTINGS_KEYS.user(scope),
    queryFn: async () => apiClient<UserSettingsDTO>('/settings/user'),
  });
}

export function useUpdateUserSettingsMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (settings: UserSettingsDTO) =>
      apiClient<{ message: string }>('/settings/user', {
        method: 'POST',
        body: JSON.stringify(settings),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: SETTINGS_KEYS.user(scope) });
    },
  });
}

export function useAdminSettingsQuery(enabled = true) {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: SETTINGS_KEYS.admin(scope),
    queryFn: async () => apiClient<AdminSettingsDTO>('/settings'),
    enabled,
    retry: false,
  });
}

export function useUpdateAdminSettingsMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (settings: AdminSettingsDTO) =>
      apiClient<AdminSettingsDTO>('/settings', {
        method: 'POST',
        body: JSON.stringify(settings),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: SETTINGS_KEYS.admin(scope) });
    },
  });
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function asPath(
  value: unknown,
  fallback: UserStartPage
): UserStartPage {
  if (
    value === '/chat' ||
    value === '/harness' ||
    value === '/knowledge'
  ) {
    return value;
  }
  return fallback;
}

function normalizeContextPruningConfig(value: unknown): Partial<ContextPruningAdminConfig> {
  const pruning = asRecord(value);
  if (!pruning) {
    return {};
  }
  return {
    method: asString(pruning.method),
    auto_reranker_min_lines: asNumber(pruning.auto_reranker_min_lines),
    auto_reranker_min_chars: asNumber(pruning.auto_reranker_min_chars),
    min_keywords: asNumber(pruning.min_keywords),
    min_keep_lines: asNumber(pruning.min_keep_lines),
    max_keep_ratio: asNumber(pruning.max_keep_ratio),
    neighbor_window: asNumber(pruning.neighbor_window),
    reranker_window_radius: asNumber(pruning.reranker_window_radius),
    max_lines_per_item: asNumber(pruning.max_lines_per_item),
    score_threshold: asNumber(pruning.score_threshold),
  };
}

function normalizeRuntimeStatus(value: unknown): RuntimeStatusDTO | undefined {
  const root = asRecord(value);
  if (!root) {
    return undefined;
  }
  const reranker = asRecord(root.reranker);
  return {
    reranker: reranker
      ? {
          configured: asBoolean(reranker.configured),
          model_name:
            reranker.model_name === null ? null : asString(reranker.model_name),
          pruning_scoring_source: asString(reranker.pruning_scoring_source),
        }
      : undefined,
  };
}

export function getContextPruningConfig(
  settings: AdminSettingsDTO | undefined
): ContextPruningAdminConfig {
  const root = asRecord(settings);
  const prompt = asRecord(root?.prompt);
  const pruning = normalizeContextPruningConfig(prompt?.context_pruning);
  return {
    method: pruning.method ?? 'heuristic',
    auto_reranker_min_lines: pruning.auto_reranker_min_lines ?? 40,
    auto_reranker_min_chars: pruning.auto_reranker_min_chars ?? 2500,
    min_keywords: pruning.min_keywords ?? 2,
    min_keep_lines: pruning.min_keep_lines ?? 4,
    max_keep_ratio: pruning.max_keep_ratio ?? 0.45,
    neighbor_window: pruning.neighbor_window ?? 1,
    reranker_window_radius: pruning.reranker_window_radius ?? 1,
    max_lines_per_item: pruning.max_lines_per_item ?? 24,
    score_threshold: pruning.score_threshold ?? 0.18,
  };
}

export function getUserPreferenceSettings(
  settings: UserSettingsDTO | undefined
): UserPreferencesConfig {
  const root = asRecord(settings);
  const language = root?.language;
  const theme = root?.theme;
  const responseLanguage = root?.response_language;

  return {
    language: language === 'zh-CN' ? 'zh-CN' : 'en',
    theme: theme === 'light' || theme === 'dark' ? theme : 'system',
    response_language:
      responseLanguage === 'en' || responseLanguage === 'zh-CN'
        ? responseLanguage
        : 'auto',
    start_page: asPath(root?.start_page, '/chat'),
    compact_mode: asBoolean(root?.compact_mode) ?? false,
  };
}

export function getAdminBasicsConfig(
  settings: AdminSettingsDTO | undefined
): AdminBasicsConfig {
  const root = asRecord(settings);
  const general = asRecord(root?.general);
  const llm = asRecord(root?.llm);
  const localModels = asRecord(root?.local_models);
  const search = asRecord(root?.search);
  const auth = asRecord(root?.auth);
  const storageLocal = asRecord(root?.storage_local);
  const featureFlags = asRecord(root?.feature_flags);

  return {
    app_name: asString(general?.app_name) ?? 'AgFrame',
    llm_model: asString(llm?.model) ?? 'gpt-5.2',
    embedding_model: asString(localModels?.embedding_model) ?? '',
    reranker_model: asString(localModels?.rerank_model) ?? '',
    search_provider: asString(search?.provider) ?? 'duckduckgo',
    access_token_expire_minutes: asNumber(auth?.access_token_expire_minutes) ?? 30,
    documents_dir: asString(storageLocal?.documents_dir) ?? 'data/documents',
    uploads_dir: asString(storageLocal?.uploads_dir) ?? 'data/uploads',
    enable_docs_rag: asBoolean(featureFlags?.enable_docs_rag) ?? true,
    enable_chat_memory: asBoolean(featureFlags?.enable_chat_memory) ?? true,
  };
}

export function getRerankerAvailability(
  settings: AdminSettingsDTO | undefined
): RerankerAvailability {
  const root = asRecord(settings);
  const runtimeStatus = normalizeRuntimeStatus(root?.runtime_status);
  const runtimeReranker = runtimeStatus?.reranker;
  const configured = runtimeReranker?.configured;
  const runtimeModelName = runtimeReranker?.model_name;
  if (typeof configured === 'boolean') {
    return {
      configured,
      modelName: runtimeModelName ?? null,
      scoringSource: runtimeReranker?.pruning_scoring_source ?? null,
    };
  }
  const reranker = asRecord(root?.reranker);
  const localModels = asRecord(root?.local_models);
  const modelName = (asString(reranker?.model_name) ?? asString(localModels?.rerank_model) ?? '').trim();
  return {
    configured: modelName.length > 0,
    modelName: modelName || null,
    scoringSource: modelName.length > 0 ? 'reranker_model' : 'local_phrase_fallback',
  };
}
