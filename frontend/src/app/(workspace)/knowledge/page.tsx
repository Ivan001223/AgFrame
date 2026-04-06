'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Download, Eye, FileText, RefreshCw, Trash2, UploadCloud } from 'lucide-react';
import { InlineNotice } from '@/components/feedback/InlineNotice';
import {
  useAssignDocumentKnowledgeBaseMutation,
  downloadDocumentFile,
  useDeleteDocumentMutation,
  useDocumentsQuery,
  useReindexDocumentMutation,
  useUploadDocumentMutation,
} from '@/domains/documents/hooks';
import {
  useCreateKnowledgeBaseMutation,
  useKnowledgeBasesQuery,
  type KnowledgeBaseDTO,
} from '@/domains/knowledge-bases/hooks';
import {
  clearKnowledgeUploadSession,
  getKnowledgeUploadSession,
  isKnowledgeUploadActive,
  patchKnowledgeUploadSession,
  saveKnowledgeUploadSession,
  subscribeKnowledgeUploadSession,
  type KnowledgeUploadSession,
} from '@/domains/documents/upload-session';
import { useTaskDetailQuery } from '@/domains/tasks/hooks';
import { getErrorMessage } from '@/lib/http/errors';
import { formatMessage, useMessages } from '@/lib/i18n';
import { KNOWLEDGE_MESSAGES } from './messages';

type Notice = {
  variant: 'success' | 'error' | 'info';
  message: string;
};

type DocumentCategoryKey = 'reference' | 'notes' | 'data' | 'code' | 'other';

const SUCCESS_STATUSES = new Set(['succeeded', 'completed', 'success']);
const FAILURE_STATUSES = new Set(['failed', 'error']);
const UNASSIGNED_LIBRARY_KEY = '__unassigned__';

function clampProgress(value: number) {
  return Math.max(4, Math.min(100, value));
}

function getDocumentExtension(filename: string, sourcePath: string) {
  const candidate = (filename || sourcePath || '').split('/').pop() || '';
  const match = candidate.toLowerCase().match(/\.([a-z0-9]+)$/);
  return match?.[1] ?? '';
}

function getDocumentCategory(doc: { filename: string; source_path: string }): DocumentCategoryKey {
  const extension = getDocumentExtension(doc.filename, doc.source_path);

  if (['md', 'mdx', 'txt'].includes(extension)) {
    return 'notes';
  }
  if (['csv', 'tsv', 'xls', 'xlsx', 'json', 'jsonl'].includes(extension)) {
    return 'data';
  }
  if (['py', 'js', 'jsx', 'ts', 'tsx', 'java', 'go', 'rs', 'yaml', 'yml', 'toml', 'ini', 'sh', 'sql', 'xml'].includes(extension)) {
    return 'code';
  }
  if (['pdf', 'doc', 'docx', 'ppt', 'pptx', 'rtf', 'odt'].includes(extension)) {
    return 'reference';
  }
  return 'other';
}

export default function KnowledgePage() {
  const text = useMessages(KNOWLEDGE_MESSAGES);
  const documentsQuery = useDocumentsQuery();
  const { data: documents, isLoading, isError } = documentsQuery;
  const knowledgeBasesQuery = useKnowledgeBasesQuery();
  const knowledgeBases = useMemo(() => knowledgeBasesQuery.data ?? [], [knowledgeBasesQuery.data]);
  const uploadMutation = useUploadDocumentMutation();
  const deleteMutation = useDeleteDocumentMutation();
  const reindexMutation = useReindexDocumentMutation();
  const assignKnowledgeBaseMutation = useAssignDocumentKnowledgeBaseMutation();
  const createKnowledgeBaseMutation = useCreateKnowledgeBaseMutation();
  const [uploadSession, setUploadSession] = useState<KnowledgeUploadSession | null>(() => getKnowledgeUploadSession());
  const [notice, setNotice] = useState<Notice | null>(null);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState<string>('');
  const [newKnowledgeBaseName, setNewKnowledgeBaseName] = useState('');
  const [newKnowledgeBaseDescription, setNewKnowledgeBaseDescription] = useState('');

  const activeUploadTaskId = uploadSession?.taskId ?? null;
  const activeUploadName = uploadSession?.filename ?? null;
  const uploadProgress = uploadSession?.progress ?? 0;
  const isUploading = uploadSession?.status === 'uploading';
  const uploadTaskQuery = useTaskDetailQuery(activeUploadTaskId ?? '');
  const fallbackDocumentName = activeUploadName ?? text.documentFallback;
  const totalChunks = (documents ?? []).reduce((sum, document) => sum + (document.parent_chunk_count || 0), 0);
  const totalEmbeddings = (documents ?? []).reduce((sum, document) => sum + (document.embedding_count || 0), 0);
  const totalKnowledgeBases = knowledgeBases.length;
  const categoryLabels: Record<DocumentCategoryKey, string> = useMemo(
    () => ({
      reference: text.categoryReference,
      notes: text.categoryNotes,
      data: text.categoryData,
      code: text.categoryCode,
      other: text.categoryOther,
    }),
    [text.categoryCode, text.categoryData, text.categoryNotes, text.categoryOther, text.categoryReference]
  );
  const librarySections = useMemo(() => {
    const docsByLibrary = new Map<string, NonNullable<typeof documents>>();
    for (const doc of documents ?? []) {
      const key = doc.knowledge_base_id || UNASSIGNED_LIBRARY_KEY;
      const existing = docsByLibrary.get(key) ?? [];
      existing.push(doc);
      docsByLibrary.set(key, existing);
    }

    const sections: Array<{
      key: string;
      label: string;
      description: string | null;
      documents: NonNullable<typeof documents>;
      knowledgeBase: KnowledgeBaseDTO | null;
    }> = knowledgeBases.map((knowledgeBase) => ({
      key: knowledgeBase.knowledge_base_id,
      label: knowledgeBase.name,
      description: knowledgeBase.description || null,
      documents: docsByLibrary.get(knowledgeBase.knowledge_base_id) ?? [],
      knowledgeBase,
    }));

    const unassignedDocuments = docsByLibrary.get(UNASSIGNED_LIBRARY_KEY) ?? [];
    if (unassignedDocuments.length > 0 || sections.length === 0) {
      sections.push({
        key: UNASSIGNED_LIBRARY_KEY,
        label: text.unassignedLibrary,
        description: null,
        documents: unassignedDocuments,
        knowledgeBase: null,
      });
    }

    return sections;
  }, [documents, knowledgeBases, text.unassignedLibrary]);

  useEffect(() => subscribeKnowledgeUploadSession(() => setUploadSession(getKnowledgeUploadSession())), []);

  useEffect(() => {
    if (selectedKnowledgeBaseId) {
      return;
    }
    if (knowledgeBases.length > 0) {
      setSelectedKnowledgeBaseId(knowledgeBases[0].knowledge_base_id);
    }
  }, [knowledgeBases, selectedKnowledgeBaseId]);

  useEffect(() => {
    if (activeUploadTaskId && uploadTaskQuery.isError) {
      setNotice({
        variant: 'error',
        message: text.unableRefreshUpload,
      });
      clearKnowledgeUploadSession();
      setUploadSession(null);
      documentsQuery.refetch();
      knowledgeBasesQuery.refetch();
      return;
    }

    if (!activeUploadTaskId || !uploadTaskQuery.data) {
      return;
    }

    const progressValue = Number(uploadTaskQuery.data.progress ?? uploadSession?.progress ?? 0);
    const normalizedProgress = Number.isFinite(progressValue) ? Math.max(0, Math.min(100, progressValue)) : uploadSession?.progress ?? 0;
    const status = String(uploadTaskQuery.data.status || 'processing');
    const message = String(uploadTaskQuery.data.message || uploadTaskQuery.data.step || '');
    const normalizedStatus =
      SUCCESS_STATUSES.has(status)
        ? 'completed'
        : FAILURE_STATUSES.has(status)
          ? 'failed'
          : (status as KnowledgeUploadSession['status']);
    const nextSession = patchKnowledgeUploadSession({
      status: normalizedStatus,
      progress: normalizedProgress,
      message: message || null,
      taskId: activeUploadTaskId,
      filename: fallbackDocumentName,
    });
    setUploadSession(nextSession);

    if (SUCCESS_STATUSES.has(status)) {
      setNotice({
        variant: 'success',
        message: formatMessage(text.uploadedIndexed, { name: fallbackDocumentName }),
      });
      clearKnowledgeUploadSession();
      setUploadSession(null);
      documentsQuery.refetch();
      return;
    }

    if (FAILURE_STATUSES.has(status)) {
      setNotice({
        variant: 'error',
        message: formatMessage(text.uploadedIndexFailed, { name: fallbackDocumentName }),
      });
      clearKnowledgeUploadSession();
      setUploadSession(null);
      documentsQuery.refetch();
      knowledgeBasesQuery.refetch();
    }
  }, [
    activeUploadTaskId,
    documentsQuery,
    fallbackDocumentName,
    knowledgeBasesQuery,
    text.unableRefreshUpload,
    text.uploadedIndexFailed,
    text.uploadedIndexed,
    uploadSession?.progress,
    uploadTaskQuery.data,
    uploadTaskQuery.isError,
  ]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setNotice(null);
    const nextSession = saveKnowledgeUploadSession({
      status: 'uploading',
      filename: file.name,
      progress: 0,
      taskId: null,
      message: text.uploadingToServer,
      updatedAt: Date.now(),
    });
    setUploadSession(nextSession);

    try {
      const response = await uploadMutation.mutateAsync({
        file,
        knowledgeBaseId: selectedKnowledgeBaseId || null,
        onProgress: (progress) => {
          const updated = patchKnowledgeUploadSession({
            status: 'uploading',
            filename: file.name,
            progress,
            taskId: null,
            message: text.uploadingToServer,
          });
          setUploadSession(updated);
        },
      });
      const result = response.results[0];

      if (!result) {
        clearKnowledgeUploadSession();
        setUploadSession(null);
        setNotice({ variant: 'info', message: text.uploadCompleted });
      } else if (result.status === 'queued') {
        const updated = patchKnowledgeUploadSession({
          status: 'queued',
          filename: file.name,
          progress: 100,
          taskId: result.task_id ?? null,
          message: formatMessage(text.uploadedIndexingStarted, { name: file.name }),
        });
        setUploadSession(updated);
        setNotice({
          variant: 'info',
          message: formatMessage(text.uploadedIndexingStarted, { name: file.name }),
        });
      } else if (result.status === 'duplicate') {
        clearKnowledgeUploadSession();
        setUploadSession(null);
        setNotice({
          variant: 'info',
          message: formatMessage(text.alreadyInKnowledgeBase, { name: file.name }),
        });
        documentsQuery.refetch();
        knowledgeBasesQuery.refetch();
      } else if (result.status === 'already_queued') {
        const updated = patchKnowledgeUploadSession({
          status: 'queued',
          filename: file.name,
          progress: 100,
          taskId: result.task_id ?? null,
          message: formatMessage(text.alreadyQueued, { name: file.name }),
        });
        setUploadSession(updated);
        setNotice({
          variant: 'info',
          message: formatMessage(text.alreadyQueued, { name: file.name }),
        });
      } else {
        clearKnowledgeUploadSession();
        setUploadSession(null);
        setNotice({
          variant: 'error',
          message:
            result.message ||
            formatMessage(text.failedUploadNamed, { name: file.name }),
        });
      }
    } catch (error) {
      clearKnowledgeUploadSession();
      setUploadSession(null);
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, text.failedUploadDocument),
      });
      knowledgeBasesQuery.refetch();
    } finally {
      event.target.value = '';
    }
  };

  const handleCreateKnowledgeBase = async () => {
    const name = newKnowledgeBaseName.trim();
    if (!name) {
      return;
    }
    setNotice(null);
    try {
      const created = await createKnowledgeBaseMutation.mutateAsync({
        name,
        description: newKnowledgeBaseDescription.trim() || undefined,
      });
      setNewKnowledgeBaseName('');
      setNewKnowledgeBaseDescription('');
      setSelectedKnowledgeBaseId(created.knowledge_base_id);
      setNotice({
        variant: 'success',
        message: formatMessage(text.knowledgeBaseCreated, { name: created.name }),
      });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, text.failedCreateKnowledgeBase),
      });
    }
  };

  const handleAssignKnowledgeBase = async (docId: number, filename: string, knowledgeBaseId: string) => {
    setNotice(null);
    try {
      const updated = await assignKnowledgeBaseMutation.mutateAsync({
        docId,
        knowledgeBaseId: knowledgeBaseId || null,
      });
      setNotice({
        variant: 'success',
        message: formatMessage(text.libraryAssigned, {
          name: filename,
          library: updated.knowledge_base_name || text.unassignedLibrary,
        }),
      });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, text.failedAssignLibrary),
      });
    }
  };

  const handleDelete = async (id: number, name: string) => {
    if (!window.confirm(formatMessage(text.confirmDelete, { name }))) {
      return;
    }

    setNotice(null);
    try {
      await deleteMutation.mutateAsync(id);
      setNotice({ variant: 'success', message: formatMessage(text.deletedNamed, { name }) });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, formatMessage(text.failedDeleteNamed, { name })),
      });
    }
  };

  const handleReindex = async (id: number) => {
    const target = documents?.find((doc) => doc.doc_id === id);
    setNotice(null);
    try {
      await reindexMutation.mutateAsync(id);
      setNotice({
        variant: 'success',
        message: formatMessage(text.reindexQueuedNamed, { name: target?.filename ?? text.documentFallback }),
      });
    } catch (error) {
      setNotice({
        variant: 'error',
        message: getErrorMessage(error, text.failedQueueReindex),
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
        message: getErrorMessage(error, formatMessage(text.failedDownloadNamed, { name: filename })),
      });
    }
  };

  const renderDocumentRow = (doc: NonNullable<typeof documents>[number]) => (
    <div key={doc.doc_id} className="rounded-[12px] border border-slate-200 bg-white px-5 py-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="truncate text-sm font-semibold text-slate-950">{doc.filename}</div>
            <span className="rounded-[999px] bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">
              {categoryLabels[getDocumentCategory(doc)]}
            </span>
            <span className="rounded-[999px] bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
              {doc.knowledge_base_name || text.unassignedLibrary}
            </span>
            <span className="rounded-[999px] bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
              {formatMessage(text.chunksCount, { count: doc.parent_chunk_count || 0 })}
            </span>
            <span className="rounded-[999px] bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700">
              {formatMessage(text.embeddingsCount, { count: doc.embedding_count || 0 })}
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-500">{doc.source_path}</div>
          {doc.preview && doc.preview.length > 0 ? (
            <div className="mt-3 line-clamp-2 rounded-[10px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600">
              {doc.preview[0]?.content}
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap gap-2 xl:max-w-[420px] xl:justify-end">
          <label className="min-w-[180px] text-xs font-medium text-slate-500">
            {text.moveToKnowledgeBase}
            <select
              value={doc.knowledge_base_id || ''}
              onChange={(event) => handleAssignKnowledgeBase(doc.doc_id, doc.filename, event.target.value)}
              className="mt-1 w-full rounded-[10px] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
            >
              <option value="">{text.unassignedLibrary}</option>
              {knowledgeBases.map((knowledgeBase) => (
                <option key={knowledgeBase.knowledge_base_id} value={knowledgeBase.knowledge_base_id}>
                  {knowledgeBase.name}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => handleDownload(doc.filename, doc.download_url)}
            className="inline-flex items-center gap-2 rounded-[10px] border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <Download className="h-4 w-4" />
            {text.downloadOriginalFile}
          </button>
          <button
            type="button"
            onClick={() => handleReindex(doc.doc_id)}
            className="inline-flex items-center gap-2 rounded-[10px] border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <RefreshCw className="h-4 w-4" />
            {text.reindexDocument}
          </button>
          <Link
            href={`/knowledge/${doc.doc_id}`}
            className="inline-flex items-center gap-2 rounded-[10px] border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <Eye className="h-4 w-4" />
            {text.viewDetails}
          </Link>
          <button
            type="button"
            onClick={() => handleDelete(doc.doc_id, doc.filename)}
            className="inline-flex items-center gap-2 rounded-[10px] border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-medium text-rose-600 transition hover:bg-rose-100"
          >
            <Trash2 className="h-4 w-4" />
            {text.deleteDocument}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="mx-auto max-w-7xl p-6 lg:p-8">
      <div className="grid gap-4">
        <section className="rounded-[12px] border border-slate-200 bg-white p-6 shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="inline-flex items-center gap-2 rounded-[999px] bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                <FileText className="h-3.5 w-3.5" />
                {text.title}
              </div>
              <h1 className="mt-3 text-2xl font-semibold text-slate-950">{text.title}</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
                {text.description}
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{text.librariesCountLabel}</div>
                <div className="mt-1 font-semibold text-slate-900">{totalKnowledgeBases}</div>
              </div>
              <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{text.documentsCountLabel}</div>
                <div className="mt-1 font-semibold text-slate-900">{documents?.length ?? 0}</div>
              </div>
              <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{text.chunksLabel}</div>
                <div className="mt-1 font-semibold text-slate-900">{totalChunks}</div>
              </div>
              <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{text.embeddingsLabel}</div>
                <div className="mt-1 font-semibold text-slate-900">{totalEmbeddings}</div>
              </div>
              <div className="relative">
                <div className="mb-2 text-xs font-medium text-slate-500">{text.uploadTargetLabel}</div>
                <select
                  value={selectedKnowledgeBaseId}
                  onChange={(event) => setSelectedKnowledgeBaseId(event.target.value)}
                  className="mb-3 h-12 w-full min-w-[220px] rounded-[10px] border border-slate-200 bg-white px-4 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                >
                  <option value="">{text.unassignedLibrary}</option>
                  {knowledgeBases.map((knowledgeBase) => (
                    <option key={knowledgeBase.knowledge_base_id} value={knowledgeBase.knowledge_base_id}>
                      {knowledgeBase.name}
                    </option>
                  ))}
                </select>
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  onChange={handleFileChange}
                  disabled={isKnowledgeUploadActive(uploadSession)}
                />
                <label
                  htmlFor="file-upload"
                  className={`inline-flex h-12 cursor-pointer items-center justify-center gap-2 rounded-[10px] px-5 text-sm font-semibold text-white transition ${
                    isKnowledgeUploadActive(uploadSession)
                      ? 'bg-slate-300'
                      : 'bg-blue-600 hover:bg-blue-500'
                  }`}
                >
                  <UploadCloud className="h-4 w-4" />
                  {isUploading ? text.uploading : activeUploadTaskId ? text.indexing : text.uploadDocument}
                </label>
                <div className="mt-2 text-xs text-slate-500">{text.uploadTargetHint}</div>
              </div>
            </div>
          </div>

          {notice ? (
            <InlineNotice
              variant={notice.variant}
              message={notice.message}
              onDismiss={() => setNotice(null)}
              className="mt-5"
            />
          ) : null}

          {isKnowledgeUploadActive(uploadSession) ? (
            <div className="mt-5 rounded-[12px] border border-blue-200 bg-blue-50 px-5 py-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-950">
                    {activeUploadName || text.documentUpload}
                  </div>
                  <div className="mt-1 text-sm text-slate-600">
                    {activeUploadTaskId
                      ? formatMessage(text.backgroundIndexing, {
                          status: String(uploadTaskQuery.data?.message || uploadTaskQuery.data?.step || uploadSession?.message || text.processingFallback),
                        })
                      : uploadSession?.message || text.uploadingToServer}
                  </div>
                </div>
                <div className="text-sm font-semibold text-blue-700">{uploadProgress}%</div>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full rounded-full bg-blue-600 transition-all duration-300"
                  style={{ width: `${clampProgress(uploadProgress)}%` }}
                />
              </div>
            </div>
          ) : null}

          <div className="mt-5 grid gap-3 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1.8fr)_auto]">
            <input
              value={newKnowledgeBaseName}
              onChange={(event) => setNewKnowledgeBaseName(event.target.value)}
              placeholder={text.knowledgeBaseName}
              className="h-12 rounded-[10px] border border-slate-200 bg-white px-4 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
            />
            <input
              value={newKnowledgeBaseDescription}
              onChange={(event) => setNewKnowledgeBaseDescription(event.target.value)}
              placeholder={text.knowledgeBaseDescription}
              className="h-12 rounded-[10px] border border-slate-200 bg-white px-4 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
            />
            <button
              type="button"
              onClick={handleCreateKnowledgeBase}
              disabled={createKnowledgeBaseMutation.isPending || !newKnowledgeBaseName.trim()}
              className="inline-flex h-12 items-center justify-center gap-2 rounded-[10px] bg-slate-950 px-5 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
            >
              <FileText className="h-4 w-4" />
              {createKnowledgeBaseMutation.isPending ? text.creatingKnowledgeBase : text.createKnowledgeBase}
            </button>
          </div>
          {knowledgeBases.length === 0 ? (
            <div className="mt-3 rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              {text.noKnowledgeBasesHint}
            </div>
          ) : null}
        </section>

        <section className="rounded-[12px] border border-slate-200 bg-white p-6 shadow-[0_12px_36px_-28px_rgba(15,23,42,0.35)]">
          <div className="flex flex-col gap-4 border-b border-slate-200 pb-5">
            <div>
              <div className="text-sm font-semibold text-slate-900">{text.groupedLibrary}</div>
              <div className="mt-1 text-sm text-slate-500">{text.groupedLibraryDescription}</div>
            </div>
            {!isLoading && !isError && librarySections.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {librarySections.map((group) => (
                  <div
                    key={group.key}
                    className="rounded-[999px] border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold text-slate-700"
                  >
                    {group.label} · {group.documents.length}
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-6">
            {isLoading ? (
              <div className="py-10 text-center text-sm text-slate-500">{text.loadingDocuments}</div>
            ) : isError ? (
              <div className="py-10 text-center text-sm text-rose-600">{text.failedDocuments}</div>
            ) : !documents || documents.length === 0 ? (
              <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center">
                <AlertCircle className="mx-auto h-10 w-10 text-slate-300" />
                <div className="mt-4 text-base font-semibold text-slate-900">{text.noDocuments}</div>
                <p className="mt-2 text-sm text-slate-500">{text.description}</p>
              </div>
            ) : (
              <div className="space-y-6">
                {librarySections.map((group) => (
                  <section
                    key={group.key}
                    className="rounded-[16px] border border-slate-200 bg-slate-50/70 p-4"
                  >
                    <div className="flex flex-col gap-2 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="text-base font-semibold text-slate-950">{group.label}</div>
                        {group.description ? (
                          <div className="mt-1 text-sm text-slate-500">{group.description}</div>
                        ) : null}
                      </div>
                      <div className="rounded-[999px] bg-white px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">
                        {formatMessage(text.libraryDocumentCount, { count: group.documents.length })}
                      </div>
                    </div>
                    {group.documents.length === 0 ? (
                      <div className="mt-4 rounded-[12px] border border-dashed border-slate-300 bg-white px-4 py-6 text-sm text-slate-500">
                        {text.noDocumentsInLibrary}
                      </div>
                    ) : (
                      <div className="mt-4 space-y-3">
                        {group.documents.map((doc) => renderDocumentRow(doc))}
                      </div>
                    )}
                  </section>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
