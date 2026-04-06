import { redirect } from 'next/navigation';

export default async function ConversationDetailPage({
  params,
}: {
  params: Promise<{ conversationId: string }>;
}) {
  const { conversationId } = await params;
  redirect(`/chat?session=${encodeURIComponent(conversationId)}`);
}
