'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredToken } from '@/lib/auth/session';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace(getStoredToken() ? '/chat' : '/login');
  }, [router]);

  return null;
}
