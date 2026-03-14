'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  MessageSquare,
  Book,
  CheckSquare,
  Archive,
  BrainCircuit,
  Settings,
  LogOut,
} from 'lucide-react';

const NAVIGATION = [
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Knowledge', href: '/knowledge', icon: Book },
  { name: 'Tasks', href: '/tasks', icon: CheckSquare },
  { name: 'Conversations', href: '/conversations', icon: Archive },
  { name: 'Memory', href: '/memory', icon: BrainCircuit },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('agframe_token');
    }
    router.push('/login');
  };

  return (
    <div className="flex h-screen w-full bg-white dark:bg-gray-900">
      {/* Sidebar */}
      <aside className="w-64 border-r border-gray-200 bg-gray-50 flex flex-col dark:border-gray-800 dark:bg-gray-900">
        <div className="flex h-16 items-center px-6">
          <span className="text-xl font-bold tracking-tight text-indigo-600 dark:text-indigo-400">
            AgFrame
          </span>
        </div>
        
        <nav className="flex-1 space-y-1 px-3 py-4">
          {NAVIGATION.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`group flex items-center rounded-md px-3 py-2 text-sm font-medium ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/50 dark:text-indigo-300'
                    : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                }`}
              >
                <item.icon
                  className={`mr-3 h-5 w-5 flex-shrink-0 ${
                    isActive ? 'text-indigo-600 dark:text-indigo-300' : 'text-gray-400'
                  }`}
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-gray-200 p-4 dark:border-gray-800">
          <button
            onClick={handleLogout}
            className="flex w-full items-center rounded-md px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          >
            <LogOut className="mr-3 h-5 w-5 text-gray-400" aria-hidden="true" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-white dark:bg-gray-950">
        {children}
      </main>
    </div>
  );
}
