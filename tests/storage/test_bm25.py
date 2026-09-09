"""Tests for BM25 / FTS5 search."""

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
from custom_rag.storage.sqlite_store import SQLiteIndexStore, sanitize_fts_query


def _doc(
    *,
    doc_id: str,
    source_path: str,
    child_text: str,
    collection: str = "default",
) -> EmbeddedDocument:
    parent = Chunk(
        chunk_id="parent_root",
        role=ChunkRole.PARENT,
        text=f"Parent for {child_text}",
        source_block_id="root",
        block_type=BlockType.DOCUMENT,
    )
    child = Chunk(
        chunk_id="child_0",
        role=ChunkRole.CHILD,
        text=child_text,
        source_block_id="p_0",
        parent_chunk_id="parent_root",
        block_type=BlockType.PARAGRAPH,
    )
    return EmbeddedDocument(
        metadata=DocumentMetadata(
            doc_id=doc_id,
            source_path=source_path,
            source_uri=f"file://{source_path}",
            doc_type="text",
            content_hash=f"hash-{doc_id}",
            file_size=10,
            modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        parents=[parent],
        children=[
            EmbeddedChunk(
                chunk=child,
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                model="mock-embed-v1",
                provider="mock",
                dimensions=8,
                embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                text_hash=f"th-{doc_id}",
            )
        ],
    )


def test_sanitize_fts_query_quotes_tokens() -> None:
    assert sanitize_fts_query('hello "world" AND foo') == '"hello" OR "world" OR "AND" OR "foo"'
    assert sanitize_fts_query("!!!") == ""
    assert sanitize_fts_query("a ab") == '"ab"'


def test_sanitize_fts_or_allows_partial_overlap() -> None:
    # Long AND queries used to return empty on SciFact-style claims.
    q = sanitize_fts_query("alpha beta UniqueKeyword missingtoken")
    assert " OR " in q
    assert '"UniqueKeyword"' in q


def test_search_bm25_finds_known_token(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    store.upsert(_doc(doc_id="doc-a", source_path="a.txt", child_text="alpha beta UniqueKeyword"))
    store.upsert(_doc(doc_id="doc-b", source_path="b.txt", child_text="unrelated noise text"))

    hits = store.search_bm25("UniqueKeyword", top_k=5)
    assert hits
    assert hits[0].doc_id == "doc-a"
    assert hits[0].parent is None
    assert "UniqueKeyword" in hits[0].chunk.text
    store.close()


def test_search_bm25_respects_collection(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3", default_collection="default")
    store.upsert(
        _doc(doc_id="doc-a", source_path="a.txt", child_text="SharedToken in collection A"),
        collection="coll_a",
    )
    store.upsert(
        _doc(doc_id="doc-b", source_path="b.txt", child_text="SharedToken in collection B"),
        collection="coll_b",
    )

    hits_a = store.search_bm25("SharedToken", top_k=5, collection="coll_a")
    hits_b = store.search_bm25("SharedToken", top_k=5, collection="coll_b")
    assert {hit.doc_id for hit in hits_a} == {"doc-a"}
    assert {hit.doc_id for hit in hits_b} == {"doc-b"}
    store.close()


def test_search_bm25_empty_or_invalid_query_returns_empty(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    store.upsert(_doc(doc_id="doc-a", source_path="a.txt", child_text="hello world"))
    assert store.search_bm25("!!!") == []
    assert store.search_bm25("") == []
    store.close()
