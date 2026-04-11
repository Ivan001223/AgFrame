'use client';

const USERNAME_KEY = 'agframe_username';

function hasWindow() {
  return typeof window !== 'undefined';
}

export function getStoredUsername(): string {
  if (!hasWindow()) {
    return '';
  }
  return localStorage.getItem(USERNAME_KEY) || '';
}

export function getSessionCacheScope(): string {
  const username = getStoredUsername().trim();
  return username || 'anonymous';
}

export function setStoredSession(username: string) {
  if (!hasWindow()) {
    return;
  }
  localStorage.setItem(USERNAME_KEY, username);
}

export function clearStoredSession() {
  if (!hasWindow()) {
    return;
  }
  localStorage.removeItem(USERNAME_KEY);
}
