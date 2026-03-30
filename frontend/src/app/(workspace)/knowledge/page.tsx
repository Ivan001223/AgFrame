'use client';

import { useState } from 'react';
import { InlineNotice } from '@/components/feedback/InlineNotice';
import {
  downloadDocumentFile,
  useDocumentsQuery,
  useUploadDocumentMutation,
  useDeleteDocumentMutation,
  useReindexDocumentMutation,
} from '@/domains/documents/hooks';
import { getErrorMessage } from '@/lib/http/errors';
import Link from 'next/link';
import { Trash2, RefreshCw, Eye, FileText, UploadCloud, AlertCircle, Download } from 'lucide-react';

type Notice = {
  variant: 'success' | 'error' | 'info';
  message: string;
};

export default function KnowledgePage() {
  const { data: documents, isLoading, isError } = useDocumentsQuery();
  const uploadMutation = useUploadDocumentMutation();
  const deleteMutation = useDeleteDocumentMutation();
  const reindexMutation = useReindexDocumentMutation();

  const [isUploading, setIsUploading] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsUploading(true);
      setNotice(null);
      try {
        const response = await uploadMutation.mutateAsync(file);
        const result = response.results[0];
        if (!result) {
          setNotice({ variant: 'info', message: 'Upload completed.' });
        } else if (result.status === 'queued') {
          setNotice({ variant: 'success', message: `${file.name} uploaded and queued for indexing.` });
        } else if (result.status === 'duplicate') {
          setNotice({ variant: 'info', message: `${file.name} is already in the knowledge base.` });
        } else if (result.status === 'already_queued') {
          setNotice({ variant: 'info', message: `${file.name} is already queued for processing.` });
        } else {
          setNotice({ variant: 'error', message: result.message || `Failed to upload ${file.name}.` });
        }
      } catch (error) {
        setNotice({ variant: 'error', message: getErrorMessage(error, 'Failed to upload document.') });
      } finally {
        setIsUploading(false);
        // Reset the input value
        e.target.value = '';
      }
    }
  };

  const handleDelete = async (id: number, name: string) => {
    if (window.confirm(`Are you sure you want to delete ${name}?`)) {
      setNotice(null);
      try {
        await deleteMutation.mutateAsync(id);
        setNotice({ variant: 'success', message: `${name} was deleted.` });
      } catch (error) {
        setNotice({ variant: 'error', message: getErrorMessage(error, `Failed to delete ${name}.`) });
      }
    }
  };

  const handleReindex = async (id: number) => {
    const target = documents?.find((doc) => doc.doc_id === id);
    setNotice(null);
    try {
      await reindexMutation.mutateAsync(id);
      setNotice({
        variant: 'success',
        message: `${target?.filename ?? 'Document'} was queued for re-indexing.`,
      });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, 'Failed to queue document re-indexing.'),
      });
    }
  };

  const handleDownload = async (filename: string, downloadUrl?: string | null) => {
    setNotice(null);
    try {
      await downloadDocumentFile({ filename, download_url: downloadUrl });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, `Failed to download ${filename}.`),
      });
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Knowledge Base</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Manage documents, trigger re-indexing, and monitor ingestion tasks.
          </p>
        </div>
        
        {/* Upload Button */}
        <div className="relative">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleFileChange}
            disabled={isUploading}
          />
          <label
            htmlFor="file-upload"
            className={`flex cursor-pointer items-center justify-center rounded-md px-4 py-2 text-sm font-medium text-white shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
              isUploading ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            <UploadCloud className="mr-2 h-4 w-4" />
            {isUploading ? 'Uploading...' : 'Upload Document'}
          </label>
        </div>
      </div>

      {notice ? (
        <InlineNotice
          variant={notice.variant}
          message={notice.message}
          onDismiss={() => setNotice(null)}
          className="mb-6"
        />
      ) : null}

      <div className="overflow-hidden rounded-lg bg-white shadow dark:bg-gray-800">
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {isLoading ? (
            <li className="p-6 text-center text-gray-500 dark:text-gray-400">Loading documents...</li>
          ) : isError ? (
            <li className="p-6 text-center text-red-500">Failed to load documents</li>
          ) : !documents || documents.length === 0 ? (
            <li className="p-6 text-center text-gray-500 flex flex-col items-center">
              <FileText className="h-12 w-12 text-gray-300 mb-2" />
              <p>No documents found.</p>
            </li>
          ) : (
            documents.map((doc) => (
              <li key={doc.doc_id} className="flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-700">
                <div className="flex items-center min-w-0 gap-4">
                  <div className="flex-shrink-0">
                    <FileText className="h-8 w-8 text-gray-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
                      {doc.filename}
                    </p>
                    <div className="mt-1 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                      <span className="inline-flex items-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                        {doc.embedding_count} embeddings
                      </span>
                      <span>•</span>
                      <span>{doc.parent_chunk_count} chunks</span>
                      <span>•</span>
                      <span>{new Date(doc.created_at * 1000).toLocaleString()}</span>
                      {doc.checksum && (
                        <>
                          <span>•</span>
                          <span className="flex items-center truncate max-w-xs" title={doc.checksum}>
                            <AlertCircle className="w-3 h-3 mr-1" />
                            {doc.checksum.slice(0, 8)}...
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                  <button
                    onClick={() => handleDownload(doc.filename, doc.download_url)}
                    className="p-2 text-gray-400 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors"
                    title="Download original file"
                  >
                    <Download className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => handleReindex(doc.doc_id)}
                    className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Re-index document"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                  <Link
                    href={`/knowledge/${doc.doc_id}`}
                    className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="View details"
                  >
                    <Eye className="h-4 w-4" />
                  </Link>
                  <button
                    onClick={() => handleDelete(doc.doc_id, doc.filename)}
                    className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                    title="Delete document"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}
