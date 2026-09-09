"""Tests for embed_document orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.types import BlockType, Chunk, ChunkedDocument, ChunkRole, DocumentMetadata
from custom_rag.ingestion.embedders.embedder import embed_document


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path="/tmp/sample.md",
        source_uri="file:///tmp/sample.md",
        doc_type="markdown",
        content_hash="abc123",
        file_size=128,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _chunked_document() -> ChunkedDocument:
    return ChunkedDocument(
        metadata=_metadata(),
        chunks=[
            Chunk(
                chunk_id="parent_sec_1",
                role=ChunkRole.PARENT,
                text="Installation\n\nInstall with pip.",
                source_block_id="sec_1",
                block_type=BlockType.SECTION,
            ),
            Chunk(
                chunk_id="child_p_1",
                role=ChunkRole.CHILD,
                text="Install with pip.",
                source_block_id="p_1",
                parent_chunk_id="parent_sec_1",
                block_type=BlockType.PARAGRAPH,
            ),
            Chunk(
                chunk_id="child_p_2",
                role=ChunkRole.CHILD,
                text="Python 3.11 or newer.",
                source_block_id="p_2",
                parent_chunk_id="parent_sec_1",
                block_type=BlockType.PARAGRAPH,
            ),
        ],
    )


def test_embed_document_embeds_children_only() -> None:
    provider = MockEmbeddingProvider()
    embedded = embed_document(_chunked_document(), provider=provider)

    assert embedded.skipped is False
    assert len(embedded.children) == 2
    assert len(embedded.parents) == 1
    assert all(item.provider == "mock" for item in embedded.children)
    assert all(len(item.embedding) == provider.dimensions for item in embedded.children)


def test_embed_document_preserves_parent_chunks_without_vectors() -> None:
    embedded = embed_document(_chunked_document(), provider=MockEmbeddingProvider())
    assert embedded.parents[0].chunk_id == "parent_sec_1"
    assert embedded.embedded_child_by_id("child_p_1") is not None
