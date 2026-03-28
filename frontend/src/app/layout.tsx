import type { Metadata } from 'next';
import './globals.css';
import { ReactQueryProvider } from './provider';

export const metadata: Metadata = {
  title: 'AgFrame Workbench',
  description: 'Operations workbench for the AgFrame FastAPI backend.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ReactQueryProvider>
          {children}
        </ReactQueryProvider>
      </body>
    </html>
  );
}
