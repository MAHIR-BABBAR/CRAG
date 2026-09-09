"""Tests for SQLite index upsert and search."""

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
from custom_rag.storage.sqlite_store import SQLiteIndexStore


def _metadata(source_path: str, content_hash: str = "hash-1") -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path=source_path,
        source_uri=f"file://{source_path}",
        doc_type="text",
        content_hash=content_hash,
        file_size=10,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _embedded(source_path: str, *, content_hash: str = "hash-1") -> EmbeddedDocument:
    parent = Chunk(
        chunk_id="parent_doc_root",
        role=ChunkRole.PARENT,
        text="Parent context with hello world.",
        source_block_id="doc_root",
        block_type=BlockType.DOCUMENT,
    )
    child = Chunk(
        chunk_id="child_p_0",
        role=ChunkRole.CHILD,
        text="hello world",
        source_block_id="p_0",
        parent_chunk_id="parent_doc_root",
        block_type=BlockType.PARAGRAPH,
    )
    return EmbeddedDocument(
        metadata=_metadata(source_path, content_hash=content_hash),
        parents=[parent],
        children=[
            EmbeddedChunk(
                chunk=child,
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                model="mock-embed-v1",
                provider="mock",
                dimensions=8,
                embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                text_hash="text-hash-1",
            )
        ],
    )


def test_upsert_persists_parents_children_and_embeddings(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    source = str(tmp_path / "sample.txt")
    store.upsert(_embedded(source))

    assert store.get_content_hash(source) == "hash-1"
    child = store.get_chunk("doc-1", "child_p_0")
    parent = store.get_parent("doc-1", "parent_doc_root")
    assert child is not None
    assert child.text == "hello world"
    assert parent is not None
    assert "Parent context" in parent.text

    hits = store.search_vector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "child_p_0"
    assert hits[0].parent is None
    parent = store.get_parent(hits[0].doc_id, "parent_doc_root")
    assert parent is not None
    assert parent.chunk_id == "parent_doc_root"
    store.close()


def test_upsert_replaces_previous_document_rows(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    source = str(tmp_path / "sample.txt")
    store.upsert(_embedded(source, content_hash="hash-1"))

    updated = _embedded(source, content_hash="hash-2")
    updated.children[0].chunk.text = "updated hello"
    # Rebuild EmbeddedChunk with new text via model_copy
    child = updated.children[0]
    updated = updated.model_copy(
        update={
            "children": [
                child.model_copy(
                    update={
                        "chunk": child.chunk.model_copy(update={"text": "updated hello"}),
                        "text_hash": "text-hash-2",
                    }
                )
            ]
        }
    )
    store.upsert(updated)

    assert store.get_content_hash(source) == "hash-2"
    child_row = store.get_chunk("doc-1", "child_p_0")
    assert child_row is not None
    assert child_row.text == "updated hello"
    count = store._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    assert count == 1
    store.close()


def test_skipped_document_does_not_write(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    source = str(tmp_path / "sample.txt")
    skipped = EmbeddedDocument(
        metadata=_metadata(source),
        children=[],
        parents=[],
        skipped=True,
    )
    store.upsert(skipped)
    assert store.get_content_hash(source) is None
    store.close()
