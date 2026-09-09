"""Storage protocols for the CRAG index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from custom_rag.core.types import Chunk, EmbeddedDocument


@dataclass(frozen=True)
class VectorHit:
    doc_id: str
    chunk_id: str
    score: float
    chunk: Chunk
    parent: Chunk | None


@runtime_checkable
class IndexStore(Protocol):
    def get_content_hash(self, source_path: str) -> str | None: ...

    def set_content_hash(self, source_path: str, content_hash: str) -> None: ...

    def upsert(self, document: EmbeddedDocument, *, collection: str | None = None) -> None: ...

    def search_vector(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[VectorHit]: ...

    def embedding_signatures(self, collection: str | None = None) -> list[tuple[str, str, int]]: ...

    def get_document_text(self, doc_id: str) -> str: ...

    def get_chunk(self, doc_id: str, chunk_id: str) -> Chunk | None: ...

    def get_parent(self, doc_id: str, parent_chunk_id: str) -> Chunk | None: ...

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[VectorHit]: ...

    def close(self) -> None: ...
