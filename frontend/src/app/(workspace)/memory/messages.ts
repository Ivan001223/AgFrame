import type { LocalizedTextMap } from '@/lib/i18n';

export const MEMORY_MESSAGES = {
  title: { en: 'User Memory Profile', zh: '用户记忆画像' },
  description: { en: 'Review the synthesized AI memory profile built from your past interactions.', zh: '查看系统根据过往互动生成的记忆画像。' },
  loading: { en: 'Loading memory profile...', zh: '正在加载记忆画像...' },
  failed: { en: 'Failed to load memory profile. You may need to have more conversations first.', zh: '加载记忆画像失败。你可能需要先进行更多对话。' },
  emptyTitle: { en: 'No memory profile yet', zh: '还没有记忆画像' },
  emptyDescription: { en: 'Interact with the AI to start building your unique memory profile.', zh: '继续和 AI 互动后，这里会逐渐形成你的个性化记忆画像。' },
  coreMemory: { en: 'Core Memory', zh: '核心记忆' },
  userId: { en: 'User ID', zh: '用户 ID' },
  currentUser: { en: 'Current User', zh: '当前用户' },
  summary: { en: 'Synthesized Summary', zh: '归纳摘要' },
  noSummary: { en: 'No summary generated yet.', zh: '还没有生成摘要。' },
  tags: { en: 'Behavioral Tags', zh: '行为标签' },
  noTags: { en: 'No tags associated', zh: '暂无标签' },
  lastUpdated: { en: 'Last Updated', zh: '最近更新时间' },
  never: { en: 'Never', zh: '从未' },
} satisfies LocalizedTextMap<string>;
