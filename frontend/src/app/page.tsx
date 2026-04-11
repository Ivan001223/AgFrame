'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useCurrentUserQuery } from '@/domains/auth/hooks';
import { getPreferredStartPage } from '@/lib/preferences';

export default function HomePage() {
  const router = useRouter();
  const currentUserQuery = useCurrentUserQuery();

  useEffect(() => {
    if (currentUserQuery.isLoading) {
      return;
    }
    if (currentUserQuery.data) {
      router.replace(getPreferredStartPage(currentUserQuery.data.username));
      return;
    }
    router.replace('/login');
  }, [currentUserQuery.data, currentUserQuery.isLoading, router]);

  return null;
}
