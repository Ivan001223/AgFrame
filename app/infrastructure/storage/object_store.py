from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator
from urllib.parse import urlsplit

from app.infrastructure.config.settings import settings

DOCUMENT_NAMESPACE = "documents"
UPLOAD_NAMESPACE = "uploads"
STREAM_CHUNK_SIZE = 1024 * 1024
CURRENT_SEGMENT = os.curdir
PARENT_SEGMENT = os.pardir


class ObjectStoreError(RuntimeError):
    pass


class ObjectStoreNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class StorageReference:
    scheme: str
    key: str
    bucket: str | None = None
    raw: str = ""


def _normalize_key(*parts: str) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        for token in str(part).replace("\\", "/").split("/"):
            token = token.strip()
            if not token or token == CURRENT_SEGMENT:
                continue
            if token == PARENT_SEGMENT:
                raise ValueError("storage key must stay within the configured namespace")
            normalized_parts.append(token)
    normalized = "/".join(normalized_parts)
    if not normalized:
        raise ValueError("storage key must not be empty")
    return normalized


def build_document_storage_key(*, owner: str, filename: str) -> str:
    return _normalize_key(DOCUMENT_NAMESPACE, owner, filename)


def build_upload_storage_key(*, owner: str, filename: str) -> str:
    return _normalize_key(UPLOAD_NAMESPACE, owner, filename)


def parse_storage_reference(reference: str | None) -> StorageReference | None:
    raw = str(reference or "").strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if parsed.scheme == "local":
        key = _normalize_key(parsed.netloc, parsed.path.lstrip("/"))
        return StorageReference(scheme="local", key=key, raw=raw)
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise ValueError("s3 storage reference is missing the bucket name")
        key = _normalize_key(parsed.path.lstrip("/"))
        return StorageReference(scheme="s3", key=key, bucket=parsed.netloc, raw=raw)
    return StorageReference(scheme="legacy", key=raw, raw=raw)


def storage_display_label(reference: str) -> str:
    parsed = parse_storage_reference(reference)
    if parsed is None:
        return ""
    if parsed.scheme == "legacy":
        candidate = os.path.basename(parsed.raw)
        return candidate or parsed.raw
    if parsed.scheme == "s3" and parsed.bucket:
        return f"{parsed.bucket}/{parsed.key}"
    return parsed.key


def storage_filename(reference: str) -> str:
    label = storage_display_label(reference)
    candidate = PurePosixPath(label).name
    return candidate or label


def storage_media_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def _selected_storage_backend() -> str:
    configured = str(getattr(settings, "storage_backend", "local") or "").strip().lower()
    if configured:
        return configured
    if str(settings.storage_s3.s3_endpoint or "").strip():
        return "s3"
    return "local"


class _BaseObjectStore:
    backend_name = "base"

    def build_uri(self, key: str) -> str:
        raise NotImplementedError

    def store_file(self, *, source_path: str, key: str) -> str:
        raise NotImplementedError

    def delete(self, reference: str) -> None:
        raise NotImplementedError

    def exists(self, reference: str) -> bool:
        raise NotImplementedError

    def get_local_path(self, reference: str) -> str | None:
        return None

    def open_binary_stream(self, reference: str) -> BinaryIO:
        raise NotImplementedError

    @contextmanager
    def materialize_to_local_path(self, reference: str, *, suffix: str | None = None):
        local_path = self.get_local_path(reference)
        if local_path:
            yield local_path
            return

        extension = suffix or Path(storage_filename(reference)).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension or "") as temp_file:
            temp_path = temp_file.name

        try:
            with self.open_binary_stream(reference) as source, open(temp_path, "wb") as output:
                shutil.copyfileobj(source, output, length=STREAM_CHUNK_SIZE)
            yield temp_path
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def iter_bytes(self, reference: str) -> Iterator[bytes]:
        with self.open_binary_stream(reference) as stream:
            while True:
                chunk = stream.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk


class _LocalObjectStore(_BaseObjectStore):
    backend_name = "local"

    def build_uri(self, key: str) -> str:
        return f"local://{_normalize_key(key)}"

    def _resolve_key_path(self, key: str) -> Path:
        normalized = _normalize_key(key)
        namespace, _, remainder = normalized.partition("/")
        if namespace == DOCUMENT_NAMESPACE:
            root = Path(settings.storage_local.documents_dir).resolve()
        elif namespace == UPLOAD_NAMESPACE:
            root = Path(settings.storage_local.uploads_dir).resolve()
        else:
            root = (Path(settings.storage_local.data_dir) / "objects").resolve()
        relative = PurePosixPath(remainder) if remainder else PurePosixPath()
        target = root.joinpath(*relative.parts).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("resolved storage path escaped the configured root") from exc
        return target

    def _resolve_reference_path(self, reference: str) -> Path:
        parsed = parse_storage_reference(reference)
        if parsed is None:
            raise ObjectStoreNotFoundError("storage reference is empty")
        if parsed.scheme == "legacy":
            return Path(parsed.raw)
        if parsed.scheme != "local":
            raise ObjectStoreError(f"unsupported reference scheme for local backend: {parsed.scheme}")
        return self._resolve_key_path(parsed.key)

    def store_file(self, *, source_path: str, key: str) -> str:
        destination = self._resolve_key_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return self.build_uri(key)

    def delete(self, reference: str) -> None:
        path = self._resolve_reference_path(reference)
        if path.exists():
            path.unlink()

    def exists(self, reference: str) -> bool:
        return self._resolve_reference_path(reference).exists()

    def get_local_path(self, reference: str) -> str | None:
        return str(self._resolve_reference_path(reference))

    def open_binary_stream(self, reference: str) -> BinaryIO:
        path = self._resolve_reference_path(reference)
        if not path.exists():
            raise ObjectStoreNotFoundError(f"storage object not found: {reference}")
        return open(path, "rb")


class _S3ObjectStore(_BaseObjectStore):
    backend_name = "s3"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise ObjectStoreError("boto3 is required when STORAGE_BACKEND=s3") from exc

        endpoint = str(settings.storage_s3.s3_endpoint or "").strip() or None
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=str(settings.storage_s3.s3_access_key or "").strip() or None,
            aws_secret_access_key=str(settings.storage_s3.s3_secret_key or "").strip() or None,
        )
        return self._client

    @property
    def _bucket(self) -> str:
        bucket = str(settings.storage_s3.s3_bucket or "").strip()
        if not bucket:
            raise ObjectStoreError("S3 bucket is required when STORAGE_BACKEND=s3")
        return bucket

    def _parse(self, reference: str) -> StorageReference:
        parsed = parse_storage_reference(reference)
        if parsed is None:
            raise ObjectStoreNotFoundError("storage reference is empty")
        if parsed.scheme == "legacy":
            return parsed
        if parsed.scheme != "s3":
            raise ObjectStoreError(f"unsupported reference scheme for s3 backend: {parsed.scheme}")
        return parsed

    def build_uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{_normalize_key(key)}"

    def store_file(self, *, source_path: str, key: str) -> str:
        normalized = _normalize_key(key)
        with open(source_path, "rb") as source:
            self._get_client().upload_fileobj(source, self._bucket, normalized)
        return self.build_uri(normalized)

    def delete(self, reference: str) -> None:
        parsed = self._parse(reference)
        if parsed.scheme == "legacy":
            legacy_path = Path(parsed.raw)
            if legacy_path.exists():
                legacy_path.unlink()
            return
        self._get_client().delete_object(Bucket=parsed.bucket or self._bucket, Key=parsed.key)

    def exists(self, reference: str) -> bool:
        parsed = self._parse(reference)
        if parsed.scheme == "legacy":
            return Path(parsed.raw).exists()
        try:
            self._get_client().head_object(Bucket=parsed.bucket or self._bucket, Key=parsed.key)
            return True
        except Exception:
            return False

    def get_local_path(self, reference: str) -> str | None:
        parsed = self._parse(reference)
        if parsed.scheme == "legacy":
            return str(Path(parsed.raw))
        return None

    def open_binary_stream(self, reference: str) -> BinaryIO:
        parsed = self._parse(reference)
        if parsed.scheme == "legacy":
            legacy_path = Path(parsed.raw)
            if not legacy_path.exists():
                raise ObjectStoreNotFoundError(f"storage object not found: {reference}")
            return open(legacy_path, "rb")
        response = self._get_client().get_object(Bucket=parsed.bucket or self._bucket, Key=parsed.key)
        body = response.get("Body")
        if body is None:
            raise ObjectStoreNotFoundError(f"storage object not found: {reference}")
        return body


_object_store: _BaseObjectStore | None = None


def get_object_store() -> _BaseObjectStore:
    global _object_store
    if _object_store is not None:
        return _object_store

    if _selected_storage_backend() == "s3":
        _object_store = _S3ObjectStore()
    else:
        _object_store = _LocalObjectStore()
    return _object_store
