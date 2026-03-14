import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';

export type LoginRequest = {
  username: string;
  password?: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type CurrentUser = {
  id: string;
  username: string;
  is_active: boolean;
  is_superuser: boolean;
  full_name?: string;
  email?: string;
};

export const AUTH_KEYS = {
  currentUser: ['currentUser'] as const,
};

export function useLoginMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: LoginRequest) => {
      return apiClient<TokenResponse>('/auth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: data.username,
          password: data.password || '123456', // Mock password as backend accepts any
          grant_type: 'password',
        }).toString(),
      });
    },
    onSuccess: (data: TokenResponse) => {
      if (typeof window !== 'undefined') {
        localStorage.setItem('agframe_token', data.access_token);
      }
      queryClient.invalidateQueries({ queryKey: AUTH_KEYS.currentUser });
    },
  });
}
