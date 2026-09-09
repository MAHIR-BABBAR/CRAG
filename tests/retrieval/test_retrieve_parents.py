"""Tests for parent-attached retrieval."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion import index_file
from custom_rag.retrieval import retrieve_with_parents
from custom_rag.storage import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_retrieve_with_parents_returns_child_and_parent(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    provider = MockEmbeddingProvider()
    indexed = index_file(
        FIXTURES / "sample.txt",
        provider=provider,
        store=store,
        collection="default",
    )
    assert indexed.children

    result = retrieve_with_parents(
        "First paragraph",
        store=store,
        provider=provider,
        top_k=3,
        collection="default",
    )
    assert result.hits
    top = result.hits[0]
    assert top.chunk.text
    assert "paragraph" in top.chunk.text.lower() or top.chunk.text
    # Owning parent should be attached for text documents.
    assert top.parent is not None
    assert top.chunk.text in top.parent.text or top.parent.text
    store.close()
