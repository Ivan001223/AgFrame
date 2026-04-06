import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import KnowledgeBaseStore
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)


class KnowledgeBaseUpdatePayload(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)


def _serialize_knowledge_base(row: dict[str, object]) -> dict[str, object]:
    return {
        "knowledge_base_id": row.get("knowledge_base_id"),
        "user_id": row.get("user_id"),
        "name": row.get("name"),
        "description": row.get("description"),
        "document_count": row.get("document_count", 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _ensure_owned_knowledge_base(store: KnowledgeBaseStore, *, knowledge_base_id: str, user: User) -> dict[str, object]:
    knowledge_base = store.get_knowledge_base(knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if user.role != "admin" and str(knowledge_base.get("user_id") or "") != user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this knowledge base")
    return knowledge_base


@router.get("")
async def list_knowledge_bases(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        return {"knowledge_bases": []}
    store = KnowledgeBaseStore()
    rows = store.list_knowledge_bases(
        current_user.username,
        include_all_users=current_user.role == "admin",
    )
    return {"knowledge_bases": [_serialize_knowledge_base(row) for row in rows]}


@router.post("")
async def create_knowledge_base(
    payload: KnowledgeBaseCreatePayload,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Knowledge base name cannot be blank")
    store = KnowledgeBaseStore()
    created = store.create_knowledge_base(
        knowledge_base_id=f"kb_{uuid.uuid4()}",
        user_id=current_user.username,
        name=name,
        description=(payload.description or "").strip() or None,
    )
    return _serialize_knowledge_base(created)


@router.put("/{knowledge_base_id}")
async def update_knowledge_base(
    knowledge_base_id: str,
    payload: KnowledgeBaseUpdatePayload,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")
    store = KnowledgeBaseStore()
    _ensure_owned_knowledge_base(store, knowledge_base_id=knowledge_base_id, user=current_user)
    provided_fields = set(
        getattr(payload, "model_fields_set", getattr(payload, "__fields_set__", set()))
    )
    changes: dict[str, object] = {}
    if "name" in provided_fields:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="Knowledge base name cannot be blank")
        changes["name"] = name
    if "description" in provided_fields:
        changes["description"] = (
            payload.description.strip() or None if payload.description is not None else None
        )
    updated = store.update_knowledge_base(knowledge_base_id, **changes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return _serialize_knowledge_base(updated)


@router.delete("/{knowledge_base_id}")
async def delete_knowledge_base(
    knowledge_base_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")
    store = KnowledgeBaseStore()
    _ensure_owned_knowledge_base(store, knowledge_base_id=knowledge_base_id, user=current_user)
    deleted = store.delete_knowledge_base(knowledge_base_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"knowledge_base_id": knowledge_base_id, "deleted": True}
