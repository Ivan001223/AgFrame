import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.infrastructure.config.settings import settings
from app.infrastructure.database.models import User
from app.infrastructure.utils.files import resolve_path_within_roots
from app.server.api.auth import get_current_active_user

router = APIRouter(tags=["file-access"])


def _authorize_owner_access(owner: str, current_user: User) -> None:
    if current_user.role != "admin" and owner != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this file")


def _build_user_scoped_file_path(root_dir: str, owner: str, relative_path: str) -> str:
    try:
        return resolve_path_within_roots(
            os.path.join(owner, relative_path),
            default_root=root_dir,
            allowed_roots=(root_dir,),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc


@router.get("/uploads/{owner}/{relative_path:path}")
async def get_uploaded_file(
    owner: str,
    relative_path: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    _authorize_owner_access(owner, current_user)
    file_path = _build_user_scoped_file_path(
        settings.storage_local.uploads_dir,
        owner,
        relative_path,
    )
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@router.get("/files/{owner}/{relative_path:path}")
async def get_document_file(
    owner: str,
    relative_path: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    _authorize_owner_access(owner, current_user)
    file_path = _build_user_scoped_file_path(
        settings.storage_local.documents_dir,
        owner,
        relative_path,
    )
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
