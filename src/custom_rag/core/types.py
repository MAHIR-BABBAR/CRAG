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
    CODE_CONTAINER = "code_container"
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
    """A hierarchical content unit from parsing.

    ``text`` is local to this block only:
    - containers (document/page/section/module/code_container/table/slide):
      empty string or a title/signature — never duplicated descendant body text
    - leaves (paragraph/code_symbol/etc.): the body content for that unit

    Full-file / full-document content belongs in ``ParsedDocument.raw_text``.
    """

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
    """Parsed file with hierarchical blocks.

    ``raw_text`` holds the complete source or extracted document text.
    Block ``text`` fields follow the local-text contract (see ``ContentBlock``).
    """

    metadata: DocumentMetadata
    blocks: list[ContentBlock]
    raw_text: str | None = None

    @model_validator(mode="after")
    def validate_block_graph(self) -> Self:
        block_ids = {block.block_id for block in self.blocks}
        for block in self.blocks:
            if block.parent_block_id is not None and block.parent_block_id not in block_ids:
                msg = (
                    f"block {block.block_id!r} references unknown parent {block.parent_block_id!r}"
                )
                raise ValueError(msg)
        return self

    def block_by_id(self, block_id: str) -> ContentBlock | None:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def children_of(self, parent_block_id: str) -> list[ContentBlock]:
        children = [b for b in self.blocks if b.parent_block_id == parent_block_id]
        return sorted(children, key=lambda block: block.order)


class ChunkRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"


class Chunk(BaseModel):
    chunk_id: str
    role: ChunkRole
    text: str
    source_block_id: str
    parent_chunk_id: str | None = None
    split_index: int = Field(ge=0, default=0)
    block_type: BlockType
    order: int = Field(ge=0, default=0)
    hierarchy_path: list[str] = Field(default_factory=list)
    location: BlockLocation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ChunkedDocument(BaseModel):
    metadata: DocumentMetadata
    chunks: list[Chunk]
    source_block_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chunk_graph(self) -> Self:
        chunk_ids = {chunk.chunk_id for chunk in self.chunks}
        for chunk in self.chunks:
            if chunk.parent_chunk_id is not None and chunk.parent_chunk_id not in chunk_ids:
                msg = (
                    f"chunk {chunk.chunk_id!r} references unknown parent chunk "
                    f"{chunk.parent_chunk_id!r}"
                )
                raise ValueError(msg)
        return self

    def children(self) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.role == ChunkRole.CHILD]

    def parents(self) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.role == ChunkRole.PARENT]

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def parent_of(self, child_chunk_id: str) -> Chunk | None:
        child = self.chunk_by_id(child_chunk_id)
        if child is None or child.parent_chunk_id is None:
            return None
        return self.chunk_by_id(child.parent_chunk_id)


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]
    model: str
    provider: str
    dimensions: int = Field(ge=1)
    embedded_at: datetime
    text_hash: str

    @model_validator(mode="after")
    def validate_embedding_length(self) -> Self:
        if len(self.embedding) != self.dimensions:
            msg = (
                f"embedding length {len(self.embedding)} does not match "
                f"dimensions {self.dimensions}"
            )
            raise ValueError(msg)
        if self.chunk.role != ChunkRole.CHILD:
            raise ValueError("embedded chunks must come from child role chunks")
        return self


class EmbeddedDocument(BaseModel):
    metadata: DocumentMetadata
    children: list[EmbeddedChunk]
    parents: list[Chunk] = Field(default_factory=list)
    skipped: bool = False

    def embedded_child_by_id(self, chunk_id: str) -> EmbeddedChunk | None:
        for item in self.children:
            if item.chunk.chunk_id == chunk_id:
                return item
        return None
