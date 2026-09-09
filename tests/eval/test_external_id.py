"""Tests for external_id relevance judgment."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from custom_rag.core.types import (
    BlockType,
    Chunk,
    ChunkRole,
    DocumentMetadata,
    EmbeddedChunk,
    EmbeddedDocument,
)
from custom_rag.eval.runner import RelevantRule, hit_is_relevant
from custom_rag.storage.base import VectorHit
from custom_rag.storage.sqlite_store import SQLiteIndexStore


def test_hit_is_relevant_by_external_id() -> None:
    hit = VectorHit(
        doc_id="internal-1",
        chunk_id="c0",
        score=1.0,
        chunk=Chunk(
            chunk_id="c0",
            role=ChunkRole.CHILD,
            text="body",
            source_block_id="p0",
            block_type=BlockType.PARAGRAPH,
        ),
        parent=None,
    )
    rules = [RelevantRule(external_id="ext-42")]
    assert hit_is_relevant(hit, rules, source_path="x.txt", external_id="ext-42")
    assert not hit_is_relevant(hit, rules, source_path="x.txt", external_id="other")


def test_store_get_external_id(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    parent = Chunk(
        chunk_id="parent_root",
        role=ChunkRole.PARENT,
        text="parent",
        source_block_id="root",
        block_type=BlockType.DOCUMENT,
    )
    child = Chunk(
        chunk_id="child_0",
        role=ChunkRole.CHILD,
        text="hello UniqueKeyword",
        source_block_id="p_0",
        parent_chunk_id="parent_root",
        block_type=BlockType.PARAGRAPH,
    )
    doc = EmbeddedDocument(
        metadata=DocumentMetadata(
            doc_id="doc-1",
            source_path=str(tmp_path / "a.txt"),
            source_uri="file://a.txt",
            doc_type="text",
            content_hash="h1",
            file_size=1,
            modified_at=datetime(2026, 1, 1, tzinfo=UTC),
            extra={"external_id": "scifact-99"},
        ),
        parents=[parent],
        children=[
            EmbeddedChunk(
                chunk=child,
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                model="mock",
                provider="mock",
                dimensions=8,
                embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                text_hash="t1",
            )
        ],
    )
    store.upsert(doc)
    assert store.get_external_id("doc-1") == "scifact-99"
    store.close()
