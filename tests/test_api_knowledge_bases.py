from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import Base, Document, KnowledgeBase, KnowledgeBaseDocument
from app.server.api import knowledge_bases as knowledge_bases_api


@dataclass(frozen=True)
class _U:
    username: str
    role: str = "user"
    is_active: bool = True


def _build_client(monkeypatch: Any, user: _U) -> TestClient:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[
            Document.__table__,
            KnowledgeBase.__table__,
            KnowledgeBaseDocument.__table__,
        ],
    )
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )

    @contextmanager
    def _get_session() -> Any:
        session: Session = SessionLocal()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(knowledge_bases_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr("app.infrastructure.database.stores.get_session", _get_session)

    with _get_session() as session:
        session.add(
            KnowledgeBase(
                knowledge_base_id="kb-u1",
                user_id="u1",
                name="Ops KB",
                description="Existing description",
                created_at=100,
                updated_at=101,
            )
        )

    app = FastAPI()
    app.include_router(knowledge_bases_api.router)
    app.dependency_overrides[knowledge_bases_api.get_current_active_user] = lambda: user
    return TestClient(app)


def test_create_knowledge_base_rejects_blank_trimmed_name(monkeypatch: Any):
    client = _build_client(monkeypatch, _U(username="u1"))

    response = client.post(
        "/knowledge-bases",
        json={"name": "   ", "description": "ignored"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Knowledge base name cannot be blank"


def test_update_knowledge_base_trims_name_and_allows_clearing_description(monkeypatch: Any):
    client = _build_client(monkeypatch, _U(username="u1"))

    response = client.put(
        "/knowledge-bases/kb-u1",
        json={"name": "  Research KB  ", "description": None},
    )

    assert response.status_code == 200
    assert response.json()["knowledge_base_id"] == "kb-u1"
    assert response.json()["name"] == "Research KB"
    assert response.json()["description"] is None
