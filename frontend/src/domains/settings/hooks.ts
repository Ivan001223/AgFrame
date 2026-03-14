import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type UserSettingsDTO = Record<string, unknown>;
export type AdminSettingsDTO = Record<string, unknown>;

export const SETTINGS_KEYS = {
  user: ['settings', 'user'] as const,
  admin: ['settings', 'admin'] as const,
};

export function useUserSettingsQuery() {
  return useQuery({
    queryKey: SETTINGS_KEYS.user,
    queryFn: async () => apiClient<UserSettingsDTO>('/settings/user'),
  });
}

export function useUpdateUserSettingsMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: UserSettingsDTO) =>
      apiClient<{ message: string }>('/settings/user', {
        method: 'POST',
        body: JSON.stringify(settings),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: SETTINGS_KEYS.user });
    },
  });
}

export function useAdminSettingsQuery(enabled = true) {
  return useQuery({
    queryKey: SETTINGS_KEYS.admin,
    queryFn: async () => apiClient<AdminSettingsDTO>('/settings'),
    enabled,
    retry: false,
  });
}

export function useUpdateAdminSettingsMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: AdminSettingsDTO) =>
      apiClient<AdminSettingsDTO>('/settings', {
        method: 'POST',
        body: JSON.stringify(settings),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: SETTINGS_KEYS.admin });
    },
  });
}
