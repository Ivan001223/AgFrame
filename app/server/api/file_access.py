from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.infrastructure.config.settings import settings
from app.infrastructure.database.models import User
from app.infrastructure.storage.object_store import (
    build_document_storage_key,
    build_upload_storage_key,
    get_object_store,
    storage_media_type,
)
from app.server.api.auth import get_current_active_user

router = APIRouter(tags=["file-access"])


def _authorize_owner_access(owner: str, current_user: User) -> None:
    if current_user.role != "admin" and owner != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this file")


def _build_storage_response(storage_reference: str, *, filename: str):
    storage = get_object_store()
    local_path = storage.get_local_path(storage_reference)
    media_type = storage_media_type(filename)
    if local_path:
        try:
            import os

            if not os.path.isfile(local_path):
                raise HTTPException(status_code=404, detail="File not found")
        except OSError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        return FileResponse(local_path, media_type=media_type)
    if not storage.exists(storage_reference):
        raise HTTPException(status_code=404, detail="File not found")
    return StreamingResponse(storage.iter_bytes(storage_reference), media_type=media_type)


@router.get("/uploads/{owner}/{relative_path:path}")
async def get_uploaded_file(
    owner: str,
    relative_path: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    _authorize_owner_access(owner, current_user)
    filename = relative_path.rsplit("/", 1)[-1] or "upload.bin"
    storage_reference = get_object_store().build_uri(
        build_upload_storage_key(owner=owner, filename=relative_path)
    )
    return _build_storage_response(storage_reference, filename=filename)


@router.get("/files/{owner}/{relative_path:path}")
async def get_document_file(
    owner: str,
    relative_path: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    _authorize_owner_access(owner, current_user)
    filename = relative_path.rsplit("/", 1)[-1] or "document"
    storage_reference = get_object_store().build_uri(
        build_document_storage_key(owner=owner, filename=relative_path)
    )
    return _build_storage_response(storage_reference, filename=filename)
