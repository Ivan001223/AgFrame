import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import {
  clearStoredSession,
  getStoredToken,
  setStoredSession,
} from '@/lib/auth/session';

export type LoginRequest = {
  username: string;
  password?: string;
};

export type RegisterRequest = {
  username: string;
  password: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type CurrentUser = {
  username: string;
  role: string;
  is_active: boolean;
};

export const AUTH_KEYS = {
  currentUser: ['currentUser'] as const,
};

export function useCurrentUserQuery() {
  const token = getStoredToken();

  return useQuery({
    queryKey: AUTH_KEYS.currentUser,
    queryFn: async () => apiClient<CurrentUser>('/auth/users/me'),
    enabled: !!token,
    retry: false,
    staleTime: 5 * 60 * 1000,
  });
}

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
          password: data.password || '',
          grant_type: 'password',
        }).toString(),
      });
    },
    onSuccess: (data: TokenResponse, variables: LoginRequest) => {
      setStoredSession(data.access_token, variables.username);
      queryClient.invalidateQueries({ queryKey: AUTH_KEYS.currentUser });
    },
  });
}

export function useRegisterMutation() {
  return useMutation({
    mutationFn: async (data: RegisterRequest) => {
      return apiClient<CurrentUser>('/auth/register', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();

  return () => {
    clearStoredSession();
    queryClient.removeQueries({ queryKey: AUTH_KEYS.currentUser });
  };
}
