'use client';

import { getStoredUsername } from '@/lib/auth/session';

export type KnowledgeUploadStatus =
  | 'uploading'
  | 'queued'
  | 'running'
  | 'processing'
  | 'indexing'
  | 'completed'
  | 'failed';

export type KnowledgeUploadSession = {
  status: KnowledgeUploadStatus;
  filename: string | null;
  progress: number;
  taskId: string | null;
  message: string | null;
  updatedAt: number;
};

const STORAGE_KEY = 'agframe_knowledge_upload';
const STORAGE_EVENT = 'agframe:knowledge-upload-changed';

function hasWindow() {
  return typeof window !== 'undefined';
}

function buildStorageKey(username?: string) {
  return username ? `${STORAGE_KEY}:${username}` : STORAGE_KEY;
}

function normalizeUploadSession(value: unknown): KnowledgeUploadSession | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }

  const record = value as Record<string, unknown>;
  const status = typeof record.status === 'string' ? record.status : null;
  if (!status) {
    return null;
  }

  return {
    status: status as KnowledgeUploadStatus,
    filename: typeof record.filename === 'string' ? record.filename : null,
    progress:
      typeof record.progress === 'number' && Number.isFinite(record.progress)
        ? Math.max(0, Math.min(100, record.progress))
        : 0,
    taskId: typeof record.taskId === 'string' ? record.taskId : null,
    message: typeof record.message === 'string' ? record.message : null,
    updatedAt:
      typeof record.updatedAt === 'number' && Number.isFinite(record.updatedAt)
        ? record.updatedAt
        : Date.now(),
  };
}

function emitUploadSessionChanged() {
  if (!hasWindow()) {
    return;
  }
  window.dispatchEvent(new Event(STORAGE_EVENT));
}

export function getKnowledgeUploadSession(username = getStoredUsername()): KnowledgeUploadSession | null {
  if (!hasWindow()) {
    return null;
  }

  const raw = window.localStorage.getItem(buildStorageKey(username));
  if (!raw) {
    return null;
  }

  try {
    return normalizeUploadSession(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function saveKnowledgeUploadSession(
  session: KnowledgeUploadSession,
  username = getStoredUsername()
): KnowledgeUploadSession {
  if (!hasWindow()) {
    return session;
  }

  window.localStorage.setItem(buildStorageKey(username), JSON.stringify(session));
  emitUploadSessionChanged();
  return session;
}

export function patchKnowledgeUploadSession(
  patch: Partial<Omit<KnowledgeUploadSession, 'updatedAt'>>,
  username = getStoredUsername()
): KnowledgeUploadSession {
  const current = getKnowledgeUploadSession(username);
  const next = {
    status: patch.status ?? current?.status ?? 'uploading',
    filename:
      patch.filename !== undefined
        ? patch.filename
        : current?.filename ?? null,
    progress:
      patch.progress !== undefined
        ? Math.max(0, Math.min(100, patch.progress))
        : current?.progress ?? 0,
    taskId:
      patch.taskId !== undefined
        ? patch.taskId
        : current?.taskId ?? null,
    message:
      patch.message !== undefined
        ? patch.message
        : current?.message ?? null,
    updatedAt: Date.now(),
  } satisfies KnowledgeUploadSession;

  return saveKnowledgeUploadSession(next, username);
}

export function clearKnowledgeUploadSession(username = getStoredUsername()) {
  if (!hasWindow()) {
    return;
  }

  window.localStorage.removeItem(buildStorageKey(username));
  emitUploadSessionChanged();
}

export function subscribeKnowledgeUploadSession(callback: () => void) {
  if (!hasWindow()) {
    return () => undefined;
  }

  const handleStorage = (event: StorageEvent) => {
    if (!event.key || event.key.startsWith(STORAGE_KEY)) {
      callback();
    }
  };

  window.addEventListener(STORAGE_EVENT, callback);
  window.addEventListener('storage', handleStorage);

  return () => {
    window.removeEventListener(STORAGE_EVENT, callback);
    window.removeEventListener('storage', handleStorage);
  };
}

export function isKnowledgeUploadActive(session: KnowledgeUploadSession | null) {
  if (!session) {
    return false;
  }

  return ['uploading', 'queued', 'running', 'processing', 'indexing'].includes(session.status);
}
