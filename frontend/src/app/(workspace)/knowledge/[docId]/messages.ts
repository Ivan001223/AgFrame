import type { LocalizedTextMap } from '@/lib/i18n';

export const DOCUMENT_DETAIL_MESSAGES = {
  downloadFailed: { en: 'Failed to download document.', zh: '下载文档失败。' },
  loading: { en: 'Loading document details...', zh: '正在加载文档详情...' },
  error: { en: 'Error', zh: '错误' },
  failedDetail: { en: 'Failed to load document details. It may have been deleted.', zh: '加载文档详情失败，它可能已经被删除。' },
  returnKnowledge: { en: 'Return to Knowledge Base', zh: '返回知识库' },
  backToDocuments: { en: 'Back to documents', zh: '返回文档列表' },
  documentId: { en: 'Document Id', zh: '文档编号' },
  download: { en: 'Download', zh: '下载' },
  status: { en: 'Status', zh: '状态' },
  uploaded: { en: 'Uploaded', zh: '已上传' },
  chunked: { en: 'Chunked', zh: '已切片' },
  indexed: { en: 'Indexed', zh: '已索引' },
  owner: { en: 'Owner', zh: '所属用户' },
  unknownOwner: { en: 'Unknown', zh: '未知' },
  createdAt: { en: 'Created At', zh: '创建时间' },
  chunks: { en: 'Chunks', zh: '切片数' },
  embeddings: { en: 'Embeddings', zh: '向量数' },
  preview: { en: 'Preview', zh: '内容预览' },
} satisfies LocalizedTextMap<string>;
