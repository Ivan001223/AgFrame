'use client';

import Link from 'next/link';
import { InlineNotice } from '@/components/feedback/InlineNotice';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useCurrentUserQuery, useLoginMutation, useRegisterMutation } from '@/domains/auth/hooks';
import { getErrorMessage } from '@/lib/http/errors';
import { useMessages } from '@/lib/i18n';
import { getPreferredStartPage } from '@/lib/preferences';
import { PUBLIC_MESSAGES } from '../messages';

function buildRegisterSchema(passwordsMismatch: string) {
  return z
    .object({
      username: z.string().min(1),
      password: z.string().min(6),
      confirmPassword: z.string().min(6),
      bootstrapAdminToken: z.string().optional(),
    })
    .refine((value) => value.password === value.confirmPassword, {
      message: passwordsMismatch,
      path: ['confirmPassword'],
    });
}

type RegisterFormValues = z.infer<ReturnType<typeof buildRegisterSchema>>;

export default function RegisterPage() {
  const router = useRouter();
  const text = useMessages(PUBLIC_MESSAGES);
  const registerSchema = useMemo(() => buildRegisterSchema(text.passwordsMismatch), [text.passwordsMismatch]);
  const currentUserQuery = useCurrentUserQuery();
  const registerMutation = useRegisterMutation();
  const loginMutation = useLoginMutation();
  const [registerError, setRegisterError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
  });

  useEffect(() => {
    if (currentUserQuery.data) {
      router.replace(getPreferredStartPage(currentUserQuery.data.username));
    }
  }, [currentUserQuery.data, router]);

  const onSubmit = async (data: RegisterFormValues) => {
    setRegisterError(null);

    try {
      await registerMutation.mutateAsync({
        username: data.username,
        password: data.password,
        bootstrapAdminToken: data.bootstrapAdminToken?.trim() || undefined,
      });
      await loginMutation.mutateAsync({
        username: data.username,
        password: data.password,
      });
      router.replace(getPreferredStartPage(data.username));
    } catch (error) {
      setRegisterError(getErrorMessage(error, text.registrationFailed));
    }
  };

  const isSubmitting = registerMutation.isPending || loginMutation.isPending;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8 dark:bg-gray-900">
      <div className="w-full max-w-md space-y-8 rounded-xl bg-white p-10 shadow-lg dark:bg-gray-800">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
            {text.createAccountTitle}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
            {text.firstUserAdmin}
          </p>
        </div>

        {registerError ? (
          <InlineNotice
            variant="error"
            message={registerError}
            onDismiss={() => setRegisterError(null)}
          />
        ) : null}

        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 rounded-md shadow-sm">
            <div>
              <input
                type="text"
                autoComplete="username"
                required
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.username}
                {...register('username')}
              />
              {errors.username && (
                <p className="mt-2 text-sm text-red-600">{text.usernameRequired}</p>
              )}
            </div>
            <div>
              <input
                type="password"
                autoComplete="off"
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.bootstrapToken}
                {...register('bootstrapAdminToken')}
              />
            </div>
            <div>
              <input
                type="password"
                autoComplete="new-password"
                required
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.passwordMin}
                {...register('password')}
              />
              {errors.password && (
                <p className="mt-2 text-sm text-red-600">{text.passwordMinError}</p>
              )}
            </div>
            <div>
              <input
                type="password"
                autoComplete="new-password"
                required
                className="relative block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:z-10 focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder={text.confirmPassword}
                {...register('confirmPassword')}
              />
              {errors.confirmPassword && (
                <p className="mt-2 text-sm text-red-600">
                  {text.passwordsMismatch}
                </p>
              )}
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={isSubmitting}
              className="group relative flex w-full justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
            >
              {isSubmitting ? text.creatingAccount : text.createAccount}
            </button>
          </div>
        </form>

        <p className="text-center text-sm text-gray-600 dark:text-gray-400">
          {text.haveAccount}{' '}
          <Link href="/login" className="font-semibold text-indigo-600 hover:text-indigo-500">
            {text.signInLink}
          </Link>
        </p>
      </div>
    </div>
  );
}
