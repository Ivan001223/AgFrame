'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React, { useState } from 'react';
import { I18nProvider } from '@/lib/i18n';
import { ThemeProvider } from '@/lib/theme';

export function ReactQueryProvider({ children }: { children: React.ReactNode }) {
  // Ensure the client is instantiated once per render cycle on the client side
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            retry: 1,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <I18nProvider>
          {children}
          <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
        </I18nProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
