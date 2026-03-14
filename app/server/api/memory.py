import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import PgUserMemoryStore
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.runtime.llm.embeddings import get_embeddings
from app.server.api.auth import get_current_active_user
from app.skills.profile.profile_engine import UserProfileEngine, normalize_profile

router = APIRouter(prefix="/memory", tags=["memory"])


class UpdateProfileRequest(BaseModel):
    profile: dict


class CreateMemoryItemRequest(BaseModel):
    kind: str = Field(min_length=1, max_length=32)
    subkind: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=128)
    text: str = Field(min_length=1, max_length=5000)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    last_verified_at: int | None = None
    metadata_json: dict | None = None


def _item_hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _apply_profile_update_with_compensation(
    *,
    user_id: str,
    old_profile: dict,
    new_profile: dict,
    version: int,
    engine: UserProfileEngine,
    memory_engine: UserMemoryEngine,
) -> None:
    memory_engine.replace_profile_semantic_memory(
        user_id=user_id,
        profile=new_profile,
    )
    try:
        engine.upsert_profile(user_id, new_profile, version=version)
    except Exception as exc:
        try:
            memory_engine.replace_profile_semantic_memory(
                user_id=user_id,
                profile=old_profile,
            )
        except Exception as rollback_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Profile update failed and memory rollback failed: {rollback_exc}",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Profile update failed after memory sync: {exc}",
        ) from exc


@router.get("/profile")
async def get_my_profile(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        return {"user_id": current_user.username, "profile": None}

    engine = UserProfileEngine()
    return {
        "user_id": current_user.username,
        "profile": engine.get_profile(current_user.username),
    }


@router.put("/profile")
async def update_my_profile(
    payload: UpdateProfileRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")

    profile = normalize_profile(dict(payload.profile or {}))
    version = int(time.time())
    engine = UserProfileEngine()
    old_profile = engine.get_profile(current_user.username)
    memory_engine = UserMemoryEngine()
    try:
        _apply_profile_update_with_compensation(
            user_id=current_user.username,
            old_profile=old_profile,
            new_profile=profile,
            version=version,
            engine=engine,
            memory_engine=memory_engine,
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {exc}") from exc
    return {
        "message": "Updated",
        "user_id": current_user.username,
        "profile": profile,
        "version": version,
    }


@router.get("/items")
async def list_memory_items(
    current_user: Annotated[User, Depends(get_current_active_user)],
    kind: str | None = Query(default=None),
    subkind: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
):
    if not ensure_schema_if_possible():
        return {"items": []}

    store = PgUserMemoryStore()
    items = store.list_items(
        user_id=current_user.username,
        kind=kind,
        subkind=subkind,
        session_id=session_id,
        limit=limit,
    )
    return {"items": items}


@router.post("/items")
async def create_memory_item(
    payload: CreateMemoryItemRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")

    text = str(payload.text).strip()
    metadata_json = dict(payload.metadata_json or {})
    embeddings = get_embeddings()
    embedding = embeddings.embed_documents([text])[0]
    now = int(time.time())
    row = {
        "user_id": current_user.username,
        "kind": str(payload.kind).strip(),
        "subkind": str(payload.subkind).strip() if payload.subkind else None,
        "session_id": str(payload.session_id).strip() if payload.session_id else None,
        "text": text,
        "item_hash": _item_hash(
            current_user.username,
            str(payload.kind),
            str(payload.subkind or ""),
            str(payload.session_id or ""),
            text,
        ),
        "confidence_score": payload.confidence_score,
        "last_verified_at": payload.last_verified_at or now,
        "metadata_json": metadata_json,
        "embedding": embedding,
    }
    store = PgUserMemoryStore()
    count = store.upsert_items([row])
    if count != 1:
        raise HTTPException(status_code=500, detail="Failed to create memory item")
    item = store.get_item_by_hash(
        user_id=current_user.username,
        kind=row["kind"],
        item_hash=row["item_hash"],
    )
    if item is None:
        raise HTTPException(status_code=500, detail="Memory item created but could not be reloaded")
    return {"message": "Created", "item": item}


@router.delete("/items/{item_id}")
async def delete_memory_item(
    item_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=404, detail="Memory item not found")

    store = PgUserMemoryStore()
    item = store.get_item(user_id=current_user.username, item_id=item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Memory item not found")

    deleted = store.delete_item(user_id=current_user.username, item_id=item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory item not found")

    return {"message": "Deleted", "item_id": item_id}
