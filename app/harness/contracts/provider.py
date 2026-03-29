from __future__ import annotations

from pydantic import BaseModel, Field


class HarnessModelProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str = Field(min_length=1, max_length=512)
    models: list[str] = Field(default_factory=list)
    is_default: bool = False
    enabled: bool = True


class HarnessModelProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    api_key: str | None = Field(default=None, min_length=1, max_length=512)
    models: list[str] | None = None
    is_default: bool | None = None
    enabled: bool | None = None


class HarnessModelProviderSummary(BaseModel):
    provider_id: str
    user_id: str
    name: str
    base_url: str
    models: list[str] = Field(default_factory=list)
    is_default: bool = False
    enabled: bool = True
    created_at: int | None = None
    updated_at: int | None = None
