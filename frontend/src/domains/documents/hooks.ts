import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL, apiClient } from '@/lib/http/client';
import { getSessionCacheScope, getStoredToken } from '@/lib/auth/session';

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
  knowledge_base_id?: string | null;
  knowledge_base_name?: string | null;
  preview?: Array<{
    content: string;
    chunk_index?: number;
  }>;
};

export const DOCUMENT_KEYS = {
  all: (scope: string) => ['documents', scope] as const,
  detail: (scope: string, id: string | number) => ['documents', scope, String(id)] as const,
};

export type UploadDocumentRequest = {
  file: File;
  knowledgeBaseId?: string | null;
  onProgress?: (progress: number) => void;
};

// 1. Fetch all documents
export function useDocumentsQuery() {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: DOCUMENT_KEYS.all(scope),
    queryFn: async () => {
      const response = await apiClient<{ documents: DocumentDTO[] }>('/documents');
      return response.documents;
    },
  });
}

// 2. Fetch one document
export function useDocumentDetailQuery(docId: string) {
  const scope = getSessionCacheScope();
  return useQuery({
    queryKey: DOCUMENT_KEYS.detail(scope, docId),
    queryFn: async () => {
      return apiClient<DocumentDTO>(`/documents/${docId}`);
    },
    enabled: !!docId,
  });
}

// 3. Delete a document
export function useDeleteDocumentMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (docId: number) => {
      return apiClient(`/documents/${docId}`, { method: 'DELETE' });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all(scope) });
    },
  });
}

export function useAssignDocumentKnowledgeBaseMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({ docId, knowledgeBaseId }: { docId: number; knowledgeBaseId?: string | null }) =>
      apiClient<DocumentDTO>(`/documents/${docId}/knowledge-base`, {
        method: 'PUT',
        body: JSON.stringify({ knowledge_base_id: knowledgeBaseId ?? null }),
      }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all(scope) });
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.detail(scope, variables.docId) });
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases', scope] });
    },
  });
}

// 4. Reindex a document
export function useReindexDocumentMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async (docId: number) => {
      return apiClient<{ task_id: string }>(`/documents/${docId}/reindex`, {
        method: 'POST',
      });
    },
    onSuccess: (_, docId) => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.detail(scope, docId) });
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all(scope) });
      // In a real app we would also invalidate the 'tasks' query
    },
  });
}

// 5. Upload document (multipart/form-data)
export function useUploadDocumentMutation() {
  const queryClient = useQueryClient();
  const scope = getSessionCacheScope();

  return useMutation({
    mutationFn: async ({ file, knowledgeBaseId, onProgress }: UploadDocumentRequest) => {
      const token = getStoredToken();
      const formData = new FormData();
      formData.append('files', file);
      if (knowledgeBaseId) {
        formData.append('knowledge_base_id', knowledgeBaseId);
      }

      return new Promise<{
        results: Array<{
          status: string;
          task_id?: string;
          existing_doc_id?: number;
          knowledge_base_id?: string | null;
          message?: string;
        }>;
      }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_BASE_URL}/upload`);

        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }

        xhr.upload.onprogress = (event) => {
          if (!event.lengthComputable || !onProgress) {
            return;
          }
          onProgress(Math.round((event.loaded / event.total) * 100));
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch {
              reject(new Error('Upload failed'));
            }
            return;
          }

          reject(new Error('Upload failed'));
        };

        xhr.onerror = () => reject(new Error('Upload failed'));
        xhr.onabort = () => reject(new Error('Upload aborted'));
        xhr.send(formData);
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENT_KEYS.all(scope) });
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases', scope] });
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
