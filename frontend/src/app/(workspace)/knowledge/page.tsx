'use client';

import { useState } from 'react';
import {
  useDocumentsQuery,
  useUploadDocumentMutation,
  useDeleteDocumentMutation,
  useReindexDocumentMutation,
} from '@/domains/documents/hooks';
import Link from 'next/link';
import { Trash2, RefreshCw, Eye, FileText, UploadCloud, AlertCircle } from 'lucide-react';

export default function KnowledgePage() {
  const { data: documents, isLoading, isError } = useDocumentsQuery();
  const uploadMutation = useUploadDocumentMutation();
  const deleteMutation = useDeleteDocumentMutation();
  const reindexMutation = useReindexDocumentMutation();

  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsUploading(true);
      try {
        await uploadMutation.mutateAsync(file);
      } catch {
        alert('Failed to upload document.');
      } finally {
        setIsUploading(false);
        // Reset the input value
        e.target.value = '';
      }
    }
  };

  const handleDelete = async (id: number, name: string) => {
    if (window.confirm(`Are you sure you want to delete ${name}?`)) {
      await deleteMutation.mutateAsync(id);
    }
  };

  const handleReindex = async (id: number) => {
    await reindexMutation.mutateAsync(id);
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
