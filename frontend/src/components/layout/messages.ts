import type { LocalizedTextMap } from '@/lib/i18n';

export const APP_SHELL_MESSAGES = {
  chat: { en: 'Chat', zh: '对话' },
  harness: { en: 'Harness', zh: '编排台' },
  knowledge: { en: 'Knowledge', zh: '知识库' },
  conversations: { en: 'Conversations', zh: '会话历史' },
  memory: { en: 'Memory', zh: '记忆' },
  settings: { en: 'Settings', zh: '设置' },
  language: { en: 'Language', zh: '语言' },
  english: { en: 'English', zh: '英语' },
  simplifiedChinese: { en: 'Simplified Chinese', zh: '简体中文' },
  workspaceLabel: { en: 'Agent Workspace', zh: '智能体工作区' },
  workspace: { en: 'Workspace', zh: '工作区' },
  adminRole: { en: 'Admin', zh: '管理员' },
  userRole: { en: 'User', zh: '成员' },
  signOut: { en: 'Sign out', zh: '退出登录' },
  verifyingSession: { en: 'Verifying your session...', zh: '正在验证登录状态...' },
} satisfies LocalizedTextMap<string>;
