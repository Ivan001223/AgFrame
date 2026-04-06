'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredToken } from '@/lib/auth/session';
import { useCurrentUserQuery, useLogout } from '@/domains/auth/hooks';
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
    if (token && currentUserQuery.isError) {
      logout();
      router.replace('/login');
    }
  }, [currentUserQuery.isError, logout, router, token]);

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
    return null;
  }

  return <>{children}</>;
}
