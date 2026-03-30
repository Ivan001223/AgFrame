import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL, apiClient } from '@/lib/http/client';
import { getStoredToken } from '@/lib/auth/session';

export type DocumentDTO = {
  doc_id: number;
  user_id: string;
  filename: string;
  source_path: string;
  download_url?: string | null;
  checksum?: string | null;
  created_at: number;
  parent_chunk_count: number;
  embedding_count: number;
  preview?: Array<{
    content: string;
    chunk_index?: number;
  }>;
};

export const DOCUMENT_KEYS = {
  all: ['documents'] as const,
  detail: (id: string | number) => ['documents', id] as const,
};

// 1. Fetch all documents
export function useDocumentsQuery() {
  return useQuery({
    queryKey: DOCUMENT_KEYS.all,
    queryFn: async () => {
      const response = await apiClient<{ documents: DocumentDTO[] }>('/documents');
      return response.documents;
    },
  });
}

// 2. Fetch one document
export function useDocumentDetailQuery(docId: string) {
  return useQuery({
    queryKey: DOCUMENT_KEYS.detail(docId),
    queryFn: async () => {
      return apiClient<DocumentDTO>(`/documents/${docId}`);
    },
    enabled: !!docId,
  });
}

// 3. Delete a document
export function useDeleteDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (docId: number) => {
      return apiClient(`/documents/${docId}`, { method: 'DELETE' });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all });
    },
  });
}

// 4. Reindex a document
export function useReindexDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (docId: number) => {
      return apiClient<{ task_id: string }>(`/documents/${docId}/reindex`, {
        method: 'POST',
      });
    },
    onSuccess: (_, docId) => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.detail(docId) });
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all });
      // In a real app we would also invalidate the 'tasks' query
    },
  });
}

// 5. Upload document (multipart/form-data)
export function useUploadDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('files', file);
      const token = getStoredToken();

      // Note: We bypass apiClient here to prevent it from setting Content-Type: application/json
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${baseUrl}/upload`, {
        method: 'POST',
        headers: {
          ...(token && { Authorization: `Bearer ${token}` }),
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      return response.json() as Promise<{
        results: Array<{
          status: string;
          task_id?: string;
          existing_doc_id?: number;
          message?: string;
        }>;
      }>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all });
    },
  });
}

export async function downloadDocumentFile(doc: Pick<DocumentDTO, 'filename' | 'download_url'>) {
  if (!doc.download_url) {
    throw new Error('Download URL is unavailable');
  }

  const token = getStoredToken();
  const response = await fetch(`${API_BASE_URL}${doc.download_url}`, {
    method: 'GET',
    headers: {
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  });

  if (!response.ok) {
    throw new Error('Download failed');
  }

  const blob = await response.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download = doc.filename || 'document';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(objectUrl);
}
