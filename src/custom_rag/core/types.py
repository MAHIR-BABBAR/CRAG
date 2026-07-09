"""Shared domain models for the parse → chunk → retrieve pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class BlockType(StrEnum):
    DOCUMENT = "document"
    PAGE = "page"
    SLIDE = "slide"
    SECTION = "section"
    MODULE = "module"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    CODE_SYMBOL = "code_symbol"
    TABLE = "table"
    TABLE_ROW = "table_row"
    LIST_ITEM = "list_item"
    SHAPE = "shape"
    NOTE = "note"
    FIGURE = "figure"
    RECORD = "record"


class BlockLocation(BaseModel):
    page_number: int | None = None
    slide_number: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox must satisfy x1 >= x0 and y1 >= y0")
        return value


class DocumentMetadata(BaseModel):
    doc_id: str
    source_path: str
    source_uri: str
    doc_type: str
    mime_type: str | None = None
    content_hash: str
    file_size: int = Field(ge=0)
    modified_at: datetime
    indexed_at: datetime | None = None
    language: str | None = None
    encoding: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ContentBlock(BaseModel):
    block_id: str
    block_type: BlockType
    text: str
    parent_block_id: str | None = None
    order: int = Field(ge=0, default=0)
    hierarchy_path: list[str] = Field(default_factory=list)
    location: BlockLocation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    blocks: list[ContentBlock]
    raw_text: str | None = None

    @model_validator(mode="after")
    def validate_block_graph(self) -> Self:
        block_ids = {block.block_id for block in self.blocks}
        for block in self.blocks:
            if block.parent_block_id is not None and block.parent_block_id not in block_ids:
                msg = (
                    f"block {block.block_id!r} references unknown parent "
                    f"{block.parent_block_id!r}"
                )
                raise ValueError(msg)
        return self

    def block_by_id(self, block_id: str) -> ContentBlock | None:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def children_of(self, parent_block_id: str) -> list[ContentBlock]:
        return [b for b in self.blocks if b.parent_block_id == parent_block_id]
