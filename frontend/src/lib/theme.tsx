'use client';

import { useEffect, useState } from 'react';
import { getStoredUsername } from '@/lib/auth/session';
import { getStoredUserPreferences } from '@/lib/preferences';

type ThemePreference = 'system' | 'light' | 'dark';

function readThemePreference(): ThemePreference {
  const username = getStoredUsername();
  return getStoredUserPreferences(username).theme || 'system';
}

function resolveTheme(preference: ThemePreference) {
  if (
    preference === 'system' &&
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
  ) {
    return 'dark';
  }

  return preference === 'dark' ? 'dark' : 'light';
}

function applyTheme(preference: ThemePreference) {
  if (typeof document === 'undefined') {
    return;
  }

  const resolvedTheme = resolveTheme(preference);
  document.documentElement.dataset.theme = preference;
  document.documentElement.classList.toggle('dark', resolvedTheme === 'dark');
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<ThemePreference>(() => readThemePreference());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    const handlePreferencesChanged = () => {
      setTheme(readThemePreference());
    };

    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const handleMediaChanged = () => {
      if (readThemePreference() === 'system') {
        setTheme('system');
      }
    };

    window.addEventListener('agframe:preferences-changed', handlePreferencesChanged);
    media.addEventListener('change', handleMediaChanged);

    return () => {
      window.removeEventListener('agframe:preferences-changed', handlePreferencesChanged);
      media.removeEventListener('change', handleMediaChanged);
    };
  }, []);

  return <>{children}</>;
}
