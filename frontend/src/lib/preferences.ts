'use client';

export type StoredUserPreferences = {
  language?: 'en' | 'zh-CN';
  theme?: 'system' | 'light' | 'dark';
  response_language?: 'auto' | 'en' | 'zh-CN';
  start_page?: '/chat' | '/harness' | '/knowledge';
  compact_mode?: boolean;
};

const PREFERENCES_KEY = 'agframe_user_preferences';

function hasWindow() {
  return typeof window !== 'undefined';
}

function buildPreferencesKey(username?: string) {
  return username ? `${PREFERENCES_KEY}:${username}` : PREFERENCES_KEY;
}

export function getStoredUserPreferences(username?: string): StoredUserPreferences {
  if (!hasWindow()) {
    return {};
  }

  const raw = window.localStorage.getItem(buildPreferencesKey(username));
  if (!raw) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as StoredUserPreferences;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function saveStoredUserPreferences(preferences: StoredUserPreferences, username?: string) {
  if (!hasWindow()) {
    return;
  }

  window.localStorage.setItem(buildPreferencesKey(username), JSON.stringify(preferences));
  window.dispatchEvent(new Event('agframe:preferences-changed'));
}

export function getPreferredStartPage(username?: string) {
  return getStoredUserPreferences(username).start_page || '/chat';
}
