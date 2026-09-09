"""Embedding step configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedConfig(BaseModel):
    batch_size: int = Field(default=64, ge=1)
    provider: str | None = None
    model: str | None = None
