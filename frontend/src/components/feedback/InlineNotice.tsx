'use client';

import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';

type InlineNoticeProps = {
  variant?: 'info' | 'success' | 'error';
  message: string;
  onDismiss?: () => void;
  className?: string;
};

const VARIANT_STYLES: Record<NonNullable<InlineNoticeProps['variant']>, string> = {
  info: 'border-indigo-100 bg-indigo-50 text-indigo-700 dark:border-indigo-900/40 dark:bg-indigo-950/40 dark:text-indigo-200',
  success:
    'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200',
  error: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-200',
};

const VARIANT_ICON = {
  info: Info,
  success: CheckCircle2,
  error: AlertCircle,
} satisfies Record<NonNullable<InlineNoticeProps['variant']>, typeof Info>;

export function InlineNotice({
  variant = 'info',
  message,
  onDismiss,
  className = '',
}: InlineNoticeProps) {
  const Icon = VARIANT_ICON[variant];

  return (
    <div
      className={`rounded-lg border px-4 py-3 text-sm ${VARIANT_STYLES[variant]} ${className}`.trim()}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-4 w-4 flex-shrink-0" />
        <div className="flex-1 leading-6">{message}</div>
        {onDismiss ? (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md p-1 opacity-70 transition-opacity hover:opacity-100"
            aria-label="Dismiss notice"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>
    </div>
  );
}
