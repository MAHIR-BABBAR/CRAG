"""Tests for SQLite-backed content-hash dedup during indexing."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion import index_file
from custom_rag.storage import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_second_index_skips_unchanged_content_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "crag.sqlite3"
    store = SQLiteIndexStore(db_path)
    provider = MockEmbeddingProvider()
    source = FIXTURES / "sample.txt"

    first = index_file(source, provider=provider, store=store, collection="default")
    assert first.skipped is False
    assert len(first.children) > 0

    second = index_file(source, provider=provider, store=store, collection="default")
    assert second.skipped is True
    assert second.children == []
    store.close()
