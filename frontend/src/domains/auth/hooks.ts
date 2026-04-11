import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/http/client';
import { ApiError } from '@/lib/http/errors';
import {
  clearStoredSession,
  getSessionCacheScope,
  setStoredSession,
} from '@/lib/auth/session';

export type LoginRequest = {
  username: string;
  password?: string;
};

export type RegisterRequest = {
  username: string;
  password: string;
  bootstrapAdminToken?: string;
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
  const scope = getSessionCacheScope();

  return useQuery({
    queryKey: [...AUTH_KEYS.currentUser, scope],
    queryFn: async () => {
      const user = await apiClient<CurrentUser>('/auth/users/me');
      setStoredSession(user.username);
      return user;
    },
    retry: (failureCount, error) => {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        return false;
      }
      return failureCount < 2;
    },
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
      setStoredSession(variables.username);
      queryClient.invalidateQueries({ queryKey: AUTH_KEYS.currentUser });
    },
  });
}

export function useRegisterMutation() {
  return useMutation({
    mutationFn: async (data: RegisterRequest) => {
      return apiClient<CurrentUser>('/auth/register', {
        method: 'POST',
        headers: data.bootstrapAdminToken
          ? {
              'X-Bootstrap-Admin-Token': data.bootstrapAdminToken,
            }
          : undefined,
        body: JSON.stringify(data),
      });
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();

  return () => {
    void apiClient('/auth/logout', {
      method: 'POST',
    }).catch(() => undefined);
    clearStoredSession();
    queryClient.clear();
  };
}
