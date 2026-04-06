'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredToken, getStoredUsername } from '@/lib/auth/session';
import { getPreferredStartPage } from '@/lib/preferences';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace(getStoredToken() ? getPreferredStartPage(getStoredUsername()) : '/login');
  }, [router]);

  return null;
}
