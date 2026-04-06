'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
  Bot,
  Book,
  BrainCircuit,
  Workflow,
  Settings,
  LogOut,
  Languages,
} from 'lucide-react';
import { useCurrentUserQuery, useLogout } from '@/domains/auth/hooks';
import { getUserPreferenceSettings, useUserSettingsQuery } from '@/domains/settings/hooks';
import { useI18n, useMessages } from '@/lib/i18n';
import { saveStoredUserPreferences } from '@/lib/preferences';
import { APP_SHELL_MESSAGES } from './messages';

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const logout = useLogout();
  const { data: currentUser } = useCurrentUserQuery();
  const userSettingsQuery = useUserSettingsQuery();
  const { locale, setLocale } = useI18n();
  const text = useMessages(APP_SHELL_MESSAGES);
  const userPreferences = getUserPreferenceSettings(userSettingsQuery.data);
  const [isLanguageMenuOpen, setIsLanguageMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    router.replace('/login');
  };

  useEffect(() => {
    if (userSettingsQuery.data) {
      saveStoredUserPreferences(userPreferences, currentUser?.username);
    }
  }, [currentUser?.username, userPreferences, userSettingsQuery.data]);

  const navigation = [
    { href: '/chat', icon: Bot, label: text.chat },
    { href: '/harness', icon: Workflow, label: text.harness },
    { href: '/knowledge', icon: Book, label: text.knowledge },
    { href: '/memory', icon: BrainCircuit, label: text.memory },
    { href: '/settings', icon: Settings, label: text.settings },
  ] as const;

  return (
    <div className="flex h-screen w-full bg-transparent">
      <aside className="flex w-[280px] flex-col border-r border-slate-200 bg-white/92 px-4 py-4 shadow-[0_18px_50px_-42px_rgba(15,23,42,0.4)] backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-800 dark:bg-slate-900">
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-400">
            {text.workspaceLabel}
          </div>
          <div className="mt-2 text-2xl font-semibold tracking-tight text-blue-600 dark:text-blue-300">
            AgFrame
          </div>
          <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {text.workspace}
          </div>
        </div>

        <nav className="flex-1 space-y-1.5 py-4">
          {navigation.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsLanguageMenuOpen(false)}
                className={`group flex items-center gap-3 rounded-[8px] px-3 py-2.5 text-sm font-medium transition ${
                  isActive
                    ? 'border border-blue-200 bg-blue-50 text-blue-700 shadow-[0_12px_30px_-24px_rgba(37,99,235,0.55)] dark:border-blue-900/60 dark:bg-blue-950/40 dark:text-blue-200'
                    : 'border border-transparent text-slate-700 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-900'
                }`}
              >
                <item.icon
                  className={`h-[18px] w-[18px] flex-shrink-0 ${
                    isActive ? 'text-blue-600 dark:text-blue-300' : 'text-slate-400'
                  }`}
                  aria-hidden="true"
                />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-slate-200 pt-4 dark:border-slate-800">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-slate-900 dark:text-white">
                {currentUser?.username || text.workspace}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setIsLanguageMenuOpen((current) => !current)}
                  className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 shadow-sm transition hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-900"
                  aria-label={text.language}
                >
                  <Languages className="h-4 w-4" />
                </button>
                {isLanguageMenuOpen ? (
                  <div className="absolute bottom-12 right-0 z-20 min-w-[156px] rounded-[12px] border border-slate-200 bg-white p-2 shadow-[0_18px_50px_-32px_rgba(15,23,42,0.4)] dark:border-slate-800 dark:bg-slate-950">
                    <button
                      type="button"
                      onClick={() => {
                        setLocale('zh-CN');
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`flex w-full items-center rounded-[8px] px-3 py-2 text-sm transition ${
                        locale === 'zh-CN'
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900'
                      }`}
                    >
                      {text.simplifiedChinese}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setLocale('en');
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`mt-1 flex w-full items-center rounded-[8px] px-3 py-2 text-sm transition ${
                        locale === 'en'
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900'
                      }`}
                    >
                      {text.english}
                    </button>
                  </div>
                ) : null}
              </div>
              <button
                onClick={handleLogout}
                className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 shadow-sm transition hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-900"
                aria-label={text.signOut}
              >
                <LogOut className="h-[18px] w-[18px]" aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto bg-transparent">
        {children}
      </main>
    </div>
  );
}
