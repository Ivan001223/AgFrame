import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type MemoryProfileDTO = {
  user_id: string;
  profile: Record<string, unknown> | null;
};

export const MEMORY_KEYS = {
  profile: ['memory_profile'] as const,
};

export function useMemoryProfileQuery() {
  return useQuery({
    queryKey: MEMORY_KEYS.profile,
    queryFn: async () => {
      return apiClient<MemoryProfileDTO>('/memory/profile');
    },
  });
}
