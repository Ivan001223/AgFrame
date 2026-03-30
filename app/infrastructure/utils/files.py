import hashlib
import os
from collections.abc import Iterable


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    计算文件的 SHA-256 哈希值。
    
    Args:
        path: 文件路径
        chunk_size: 读取块大小 (默认 1MB)
        
    Returns:
        str: 十六进制哈希字符串
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path_within_roots(
    path: str,
    *,
    default_root: str,
    allowed_roots: Iterable[str] | None = None,
) -> str:
    """
    Resolve a user-supplied path and ensure it stays under allowed roots.

    Relative paths are anchored under ``default_root``. Absolute paths are
    accepted only when they are already inside one of the allowed roots.
    """
    candidate = str(path or "").strip()
    if not candidate:
        raise ValueError("path is required")

    resolved_default_root = os.path.abspath(default_root)
    resolved_allowed_roots = [
        os.path.abspath(root)
        for root in (allowed_roots or (resolved_default_root,))
        if str(root or "").strip()
    ]
    if not resolved_allowed_roots:
        raise ValueError("at least one allowed root is required")

    if os.path.isabs(candidate):
        resolved_path = os.path.abspath(candidate)
    else:
        resolved_path = os.path.abspath(os.path.join(resolved_default_root, candidate))

    for root in resolved_allowed_roots:
        try:
            if os.path.commonpath([resolved_path, root]) == root:
                return resolved_path
        except ValueError:
            continue

    allowed_display = ", ".join(resolved_allowed_roots)
    raise ValueError(
        f"path '{candidate}' is outside allowed roots: {allowed_display}"
    )
