'use client';

import Link from 'next/link';
import { InlineNotice } from '@/components/feedback/InlineNotice';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useCurrentUserQuery, useLoginMutation } from '@/domains/auth/hooks';
import { getErrorMessage } from '@/lib/http/errors';
import { useMessages } from '@/lib/i18n';
import { getPreferredStartPage } from '@/lib/preferences';
import { PUBLIC_MESSAGES } from '../messages';

const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter();
  const loginMutation = useLoginMutation();
  const currentUserQuery = useCurrentUserQuery();
  const [loginError, setLoginError] = useState<string | null>(null);
  const text = useMessages(PUBLIC_MESSAGES);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
  });

  useEffect(() => {
    if (currentUserQuery.data) {
      router.replace(getPreferredStartPage(currentUserQuery.data.username));
    }
  }, [currentUserQuery.data, router]);

  const onSubmit = (data: LoginFormValues) => {
    setLoginError(null);
    loginMutation.mutate(data, {
      onSuccess: () => {
        router.replace(getPreferredStartPage(data.username));
      },
      onError: (error: unknown) => {
        setLoginError(getErrorMessage(error, text.loginFailed));
      },
    });
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8 dark:bg-gray-900">
      <div className="w-full max-w-md space-y-8 rounded-xl bg-white p-10 shadow-lg dark:bg-gray-800">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
            {text.signInTitle}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
            {text.signInSubtitle}
          </p>
        </div>
        {currentUserQuery.isLoading && (
          <div className="rounded-md border border-indigo-100 bg-indigo-50 px-4 py-3 text-sm text-indigo-700 dark:border-indigo-900/40 dark:bg-indigo-950/40 dark:text-indigo-200">
            {text.restoringSession}
          </div>
        )}
        {loginError ? (
          <InlineNotice
            variant="error"
            message={loginError}
            onDismiss={() => setLoginError(null)}
          />
        ) : null}
        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 rounded-md shadow-sm">
            <div>
              <label htmlFor="username" className="sr-only">
                Username
              </label>
              <input
                id="username"
                type="text"
                autoComplete="username"
                required
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.usernameExample}
                {...register('username')}
              />
              {errors.username && (
                <p className="mt-2 text-sm text-red-600">{text.usernameRequired}</p>
              )}
            </div>
            <div>
              <label htmlFor="password" className="sr-only">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.password}
                {...register('password')}
              />
              {errors.password && (
                <p className="mt-2 text-sm text-red-600">{text.passwordRequired}</p>
              )}
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loginMutation.isPending}
              className="group relative flex w-full justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
            >
              {loginMutation.isPending ? text.signingIn : text.signIn}
            </button>
          </div>
        </form>
        <p className="text-center text-sm text-gray-600 dark:text-gray-400">
          {text.noAccount}{' '}
          <Link href="/register" className="font-semibold text-indigo-600 hover:text-indigo-500">
            {text.createOne}
          </Link>
        </p>
      </div>
    </div>
  );
}
