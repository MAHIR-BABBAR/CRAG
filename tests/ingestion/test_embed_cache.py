"""Tests for embedding cache."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.types import BlockType, Chunk, ChunkedDocument, ChunkRole, DocumentMetadata
from custom_rag.ingestion.embedders.cache import EmbeddingCache
from custom_rag.ingestion.embedders.embedder import embed_document


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path="/tmp/sample.txt",
        source_uri="file:///tmp/sample.txt",
        doc_type="text",
        content_hash="abc123",
        file_size=128,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _chunked_document() -> ChunkedDocument:
    return ChunkedDocument(
        metadata=_metadata(),
        chunks=[
            Chunk(
                chunk_id="parent_doc_root",
                role=ChunkRole.PARENT,
                text="Document",
                source_block_id="doc_root",
                block_type=BlockType.DOCUMENT,
            ),
            Chunk(
                chunk_id="child_p_0",
                role=ChunkRole.CHILD,
                text="First paragraph.",
                source_block_id="p_0",
                parent_chunk_id="parent_doc_root",
                block_type=BlockType.PARAGRAPH,
            ),
        ],
    )


class _CountingProvider(MockEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return super().embed_texts(texts)


def test_cache_reuses_vectors_for_same_text() -> None:
    provider = _CountingProvider()
    cache = EmbeddingCache()
    document = _chunked_document()

    first = embed_document(document, provider=provider, cache=cache)
    second = embed_document(document, provider=provider, cache=cache)

    assert provider.call_count == 1
    assert first.children[0].embedding == second.children[0].embedding
    assert len(cache) == 1
