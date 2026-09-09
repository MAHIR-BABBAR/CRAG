"""Chunking configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    child_max_chars: int = Field(default=1000, ge=1)
    child_overlap_chars: int = Field(default=150, ge=0)
    parent_max_chars: int = Field(default=6000, ge=1)
    min_child_chars: int = Field(default=20, ge=0)
