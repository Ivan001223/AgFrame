'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';

export type AppLocale = 'en' | 'zh-CN';

const STORAGE_KEY = 'agframe_locale';

type I18nContextValue = {
  locale: AppLocale;
  setLocale: (locale: AppLocale) => void;
  isChinese: boolean;
};

const I18nContext = createContext<I18nContextValue | null>(null);

function getInitialLocale(): AppLocale {
  if (typeof window === 'undefined') {
    return 'en';
  }

  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === 'en' || stored === 'zh-CN') {
    return stored;
  }

  return window.navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en';
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<AppLocale>(() => getInitialLocale());

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, locale);
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale: (nextLocale) => setLocaleState(nextLocale),
      isChinese: locale === 'zh-CN',
    }),
    [locale]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within I18nProvider');
  }
  return context;
}

export function useTx() {
  const { isChinese } = useI18n();
  return (en: string, zh: string) => (isChinese ? zh : en);
}

export function formatMessage(template: string, values: Record<string, string | number | null | undefined>) {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, value === null || value === undefined ? '' : String(value)),
    template
  );
}

export type LocalizedText = {
  en: string;
  zh: string;
};

export type LocalizedTextMap<T extends string> = Record<T, LocalizedText>;

export function useMessages<T extends string>(messages: LocalizedTextMap<T>) {
  const { isChinese } = useI18n();

  return useMemo(() => {
    const entries = Object.entries(messages) as Array<[T, LocalizedText]>;
    return Object.fromEntries(entries.map(([key, value]) => [key, isChinese ? value.zh : value.en])) as Record<T, string>;
  }, [isChinese, messages]);
}
