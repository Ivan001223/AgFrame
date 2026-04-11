import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { getSessionCacheScope } from '@/lib/auth/session';

export type MemoryProfileDTO = {
  user_id: string;
  profile: Record<string, unknown> | null;
};

export const MEMORY_KEYS = {
  profile: (scope: string) => ['memory_profile', scope] as const,
};

export function useMemoryProfileQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: MEMORY_KEYS.profile(scope),
    queryFn: async () => {
      return apiClient<MemoryProfileDTO>('/memory/profile');
    },
  });
}
