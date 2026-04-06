'use client';

import { useState } from 'react';
import { InlineNotice } from '@/components/feedback/InlineNotice';
import { useParams, useRouter } from 'next/navigation';
import { downloadDocumentFile, useDocumentDetailQuery } from '@/domains/documents/hooks';
import { getErrorMessage } from '@/lib/http/errors';
import { useMessages } from '@/lib/i18n';
import { ArrowLeft, Download, FileText } from 'lucide-react';
import Link from 'next/link';
import { DOCUMENT_DETAIL_MESSAGES } from './messages';

export default function DocumentDetailPage() {
  const text = useMessages(DOCUMENT_DETAIL_MESSAGES);
  const params = useParams();
  const router = useRouter();
  const docId = params.docId as string;
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const { data: doc, isLoading, isError } = useDocumentDetailQuery(docId);

  const handleDownload = async () => {
    if (!doc) {
      return;
    }
    setDownloadError(null);
    try {
      await downloadDocumentFile(doc);
    } catch (error) {
      setDownloadError(getErrorMessage(error, text.downloadFailed));
    }
  };

  if (isLoading) {
    return <div className="p-8 text-gray-500">{text.loading}</div>;
  }

  if (isError || !doc) {
    return (
      <div className="p-8 text-red-500">
        <h2 className="text-xl font-bold">{text.error}</h2>
        <p>{text.failedDetail}</p>
        <button
          onClick={() => router.push('/knowledge')}
          className="mt-4 text-indigo-600 hover:text-indigo-800"
        >
          {text.returnKnowledge}
        </button>
      </div>
    );
  }

  const documentStatus =
    doc.embedding_count > 0 ? text.indexed : doc.parent_chunk_count > 0 ? text.chunked : text.uploaded;
  const ownerLabel = doc.user_id || text.unknownOwner;

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <Link
        href="/knowledge"
        className="inline-flex items-center text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 mb-6"
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        {text.backToDocuments}
      </Link>

      {downloadError ? (
        <InlineNotice
          variant="error"
          message={downloadError}
          onDismiss={() => setDownloadError(null)}
          className="mb-6"
        />
      ) : null}

      <div className="overflow-hidden bg-white shadow sm:rounded-lg dark:bg-gray-800">
        <div className="px-4 py-5 sm:px-6 flex items-center gap-4 border-b border-gray-200 dark:border-gray-700">
          <div className="bg-indigo-100 p-3 rounded-lg dark:bg-indigo-900/50">
            <FileText className="h-8 w-8 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white">
              {doc.filename}
            </h3>
            <p className="mt-1 max-w-2xl text-sm text-gray-500 dark:text-gray-400">
              {text.documentId}: {doc.doc_id}
            </p>
          </div>
          <button
            type="button"
            onClick={handleDownload}
            className="inline-flex items-center gap-2 rounded-md bg-emerald-600 px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-emerald-700"
          >
            <Download className="h-4 w-4" />
            {text.download}
          </button>
        </div>
        
        <div className="px-4 py-5 sm:p-0">
          <dl className="divide-y divide-gray-200 dark:divide-gray-700">
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.status}</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                {documentStatus}
              </dd>
            </div>
            
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.owner}</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                {ownerLabel}
              </dd>
            </div>
            
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.createdAt}</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                {new Date(doc.created_at * 1000).toLocaleString()}
              </dd>
            </div>
            
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.chunks}</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                {doc.parent_chunk_count}
              </dd>
            </div>
            
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.embeddings}</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-white">
                {doc.embedding_count}
              </dd>
            </div>

            {doc.preview && doc.preview.length > 0 && (
              <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{text.preview}</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0 dark:text-gray-300">
                  <pre className="bg-gray-50 p-3 rounded-md overflow-x-auto dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-xs">
                    {doc.preview.map((item) => item.content).join('\n\n---\n\n')}
                  </pre>
                </dd>
              </div>
            )}
          </dl>
        </div>
      </div>
    </div>
  );
}
