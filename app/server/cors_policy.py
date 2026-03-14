from __future__ import annotations


def _normalize_origins(cors_origins: list[str] | None) -> list[str]:
    if not cors_origins:
        return []
    normalized: list[str] = []
    for origin in cors_origins:
        item = str(origin).strip()
        if not item:
            continue
        if item not in normalized:
            normalized.append(item)
    return normalized


def build_cors_options(
    *,
    cors_origins: list[str] | None,
    cors_allow_credentials: bool,
) -> dict[str, object]:
    origins = _normalize_origins(cors_origins)
    if cors_allow_credentials and "*" in origins:
        raise ValueError("CORS 配置错误: allow_credentials=true 时不允许 '*' 来源")
    return {
        "allow_origins": origins,
        "allow_credentials": cors_allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
