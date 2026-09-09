"""Tests for SQLite schema creation."""

from __future__ import annotations

from pathlib import Path

from custom_rag.storage.sqlite_store import SQLiteIndexStore


def test_schema_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "crag.sqlite3"
    store = SQLiteIndexStore(db_path)
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert "documents" in names
    assert "chunks" in names
    assert "embeddings" in names
    assert "chunks_fts" in names
    store.close()
