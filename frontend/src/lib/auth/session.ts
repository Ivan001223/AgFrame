'use client';

const TOKEN_KEY = 'agframe_token';
const USERNAME_KEY = 'agframe_username';

function hasWindow() {
  return typeof window !== 'undefined';
}

export function getStoredToken(): string {
  if (!hasWindow()) {
    return '';
  }
  return localStorage.getItem(TOKEN_KEY) || '';
}

export function getStoredUsername(): string {
  if (!hasWindow()) {
    return '';
  }
  return localStorage.getItem(USERNAME_KEY) || '';
}

export function setStoredSession(token: string, username: string) {
  if (!hasWindow()) {
    return;
  }
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USERNAME_KEY, username);
}

export function clearStoredSession() {
  if (!hasWindow()) {
    return;
  }
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
}
