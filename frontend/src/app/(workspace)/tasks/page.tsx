'use client';

import { redirect } from 'next/navigation';

export default function TasksPage() {
  redirect('/settings#task-operations');
}
