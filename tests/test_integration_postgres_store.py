from __future__ import annotations

import os
import time
import uuid

import pytest
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.orm import sessionmaker

from app.infrastructure.database.models import ChatSession, DocContent, DocEmbedding, Document, UserProfile
from app.infrastructure.database.schema import ensure_schema
from app.infrastructure.database.stores import MySQLConversationStore, MySQLDocStore, MySQLProfileStore, PgDocEmbeddingStore


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL is not configured for PostgreSQL integration test",
)


@pytest.fixture()
def pg_env(monkeypatch: pytest.MonkeyPatch):
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {exc}")

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)

    def _get_session():
        s = SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr("app.infrastructure.database.orm._engine", engine)
    monkeypatch.setattr("app.infrastructure.database.orm._SessionLocal", SessionLocal)
    monkeypatch.setattr("app.infrastructure.database.stores.get_session", lambda: _SessionContext(SessionLocal))
    ensure_schema()
    return engine, SessionLocal


class _SessionContext:
    def __init__(self, SessionLocal):
        self._SessionLocal = SessionLocal
        self._session = None

    def __enter__(self):
        self._session = self._SessionLocal()
        return self._session

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._session.commit()
        else:
            self._session.rollback()
        self._session.close()


def _cleanup(SessionLocal, user_id: str, source_path: str):
    with SessionLocal() as session:
        doc_ids = session.execute(
            select(Document.doc_id).where(Document.user_id == user_id)
        ).scalars().all()
        if doc_ids:
            session.execute(delete(DocEmbedding).where(DocEmbedding.doc_id.in_(doc_ids)))
            session.execute(delete(DocContent).where(DocContent.doc_id.in_(doc_ids)))
        session.execute(delete(Document).where(Document.user_id == user_id))
        session.execute(delete(ChatSession).where(ChatSession.user_id == user_id))
        session.execute(delete(UserProfile).where(UserProfile.user_id == user_id))
        session.commit()


def test_postgres_profile_and_conversation_store_roundtrip(pg_env):
    _, SessionLocal = pg_env
    user_id = f"itest_user_{uuid.uuid4().hex[:8]}"

    try:
        profile_store = MySQLProfileStore()
        profile_store.upsert_profile(user_id, {"facts": [{"text": "prefers tests"}]}, version=int(time.time()))
        profile = profile_store.get_profile(user_id)
        assert profile is not None
        assert profile["profile"]["facts"][0]["text"] == "prefers tests"

        conv_store = MySQLConversationStore()
        session_id = f"itest_session_{uuid.uuid4().hex[:8]}"
        saved = conv_store.save_session(
            user_id,
            session_id,
            [{"role": "user", "content": "hello"}],
            title="integration chat",
        )
        assert saved["id"] == session_id

        detail = conv_store.get_session_detail(user_id, session_id)
        assert detail is not None
        assert detail["messages"][0]["content"] == "hello"

        search = conv_store.search_sessions(user_id, "integration")
        assert len(search) == 1
    finally:
        _cleanup(SessionLocal, user_id, "")


def test_postgres_document_and_embedding_store_roundtrip(pg_env):
    _, SessionLocal = pg_env
    user_id = f"itest_user_{uuid.uuid4().hex[:8]}"
    source_path = f"/tmp/{uuid.uuid4().hex}.pdf"

    try:
        doc_store = MySQLDocStore()
        doc_id = doc_store.upsert_document(source_path=source_path, user_id=user_id, checksum="abc123")
        assert isinstance(doc_id, int)

        found = doc_store.find_by_checksum(user_id=user_id, checksum="abc123")
        assert found is not None
        assert found["doc_id"] == doc_id

        parent_ids = doc_store.insert_parent_chunks(
            doc_id,
            [{"content": "parent block", "page_num": 1}],
        )
        assert len(parent_ids) == 1

        PgDocEmbeddingStore().add_embeddings(
            [
                {
                    "doc_id": doc_id,
                    "parent_chunk_id": parent_ids[0],
                    "child_index": 0,
                    "source_path": source_path,
                    "content": "child block",
                    "embedding": [0.1] * 1024,
                    "metadata_json": {"user_id": user_id, "type": "doc_fragment"},
                }
            ]
        )

        docs = doc_store.search_documents(user_id, filename_query=".pdf")
        assert len(docs) == 1

        preview = doc_store.get_document_preview(doc_id, limit=3)
        assert len(preview) == 1
        assert preview[0]["content"] == "parent block"

        deleted = doc_store.delete_parent_chunks(doc_id)
        assert deleted >= 1
    finally:
        _cleanup(SessionLocal, user_id, source_path)
