'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredToken } from '@/lib/auth/session';
import { useCurrentUserQuery, useLogout } from '@/domains/auth/hooks';
import { getErrorMessage, isAuthApiError } from '@/lib/http/errors';
import { useMessages } from '@/lib/i18n';
import { APP_SHELL_MESSAGES } from './messages';

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const logout = useLogout();
  const token = getStoredToken();
  const currentUserQuery = useCurrentUserQuery();
  const text = useMessages(APP_SHELL_MESSAGES);

  useEffect(() => {
    if (!token) {
      router.replace('/login');
      return;
    }

    const handleExpired = () => {
      logout();
      router.replace('/login');
    };

    window.addEventListener('agframe:auth-expired', handleExpired);
    return () => {
      window.removeEventListener('agframe:auth-expired', handleExpired);
    };
  }, [logout, router, token]);

  useEffect(() => {
    if (token && isAuthApiError(currentUserQuery.error)) {
      logout();
      router.replace('/login');
    }
  }, [currentUserQuery.error, logout, router, token]);

  if (!token || currentUserQuery.isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50 px-6 dark:bg-gray-950">
        <div className="rounded-xl border border-gray-200 bg-white px-6 py-5 text-sm text-gray-600 shadow-sm dark:border-gray-800 dark:bg-gray-900 dark:text-gray-300">
          {text.verifyingSession}
        </div>
      </div>
    );
  }

  if (!currentUserQuery.data) {
    if (currentUserQuery.isError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-gray-50 px-6 dark:bg-gray-950">
          <div className="rounded-xl border border-amber-200 bg-white px-6 py-5 text-sm text-gray-600 shadow-sm dark:border-amber-900/40 dark:bg-gray-900 dark:text-gray-300">
            {getErrorMessage(currentUserQuery.error, text.verifyingSession)}
          </div>
        </div>
      );
    }
    return null;
  }

  return <>{children}</>;
}
