"""Tests for recursive directory indexing."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion.dir_index import index_path, iter_indexable_files
from custom_rag.storage.sqlite_store import SQLiteIndexStore


def test_iter_skips_hidden_and_node_modules(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# keep", encoding="utf-8")
    (tmp_path / ".secret.md").write_text("no", encoding="utf-8")
    nested = tmp_path / "node_modules" / "pkg"
    nested.mkdir(parents=True)
    (nested / "x.md").write_text("no", encoding="utf-8")
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")

    files, truncated = iter_indexable_files(tmp_path, recursive=True)
    names = {p.name for p in files}
    assert "keep.md" in names
    assert "nested.txt" in names
    assert "secret.md" not in names
    assert "x.md" not in names
    assert truncated is False


def test_iter_reports_truncation_at_max_files(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"doc{i}.md").write_text(f"doc {i}", encoding="utf-8")

    files, truncated = iter_indexable_files(tmp_path, recursive=True, max_files=3)
    assert len(files) == 3
    assert truncated is True


def test_iter_is_deterministically_ordered(tmp_path: Path) -> None:
    """A capped walk must cut the same files every run, not whatever the OS lists first."""
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.md").write_text("d", encoding="utf-8")

    first, _ = iter_indexable_files(tmp_path, recursive=True)
    second, _ = iter_indexable_files(tmp_path, recursive=True)
    assert first == second
    assert [p.name for p in first][:3] == ["a.md", "b.md", "c.md"]


def test_extensions_track_the_parser_registry(tmp_path: Path) -> None:
    """Formats the parsers support must not be silently skipped by the walker."""
    from custom_rag.ingestion.dir_index import default_index_extensions

    extensions = default_index_extensions()
    assert {".md", ".txt", ".py", ".html", ".json", ".csv"} <= extensions
    # Regression: these have parsers but were missing from the hand-written list.
    assert {".docx", ".pptx", ".rst", ".tsv"} <= extensions


def test_index_path_directory(tmp_path: Path) -> None:
    docs = tmp_path / "corpus"
    docs.mkdir()
    (docs / "a.md").write_text("alpha keyword uniqueA", encoding="utf-8")
    (docs / "b.txt").write_text("beta keyword uniqueB", encoding="utf-8")
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    result = index_path(
        docs,
        store=store,
        provider=MockEmbeddingProvider(),
        collection="demo",
        max_files=50,
    )
    assert result.indexed == 2
    assert result.failed == []
    hits = store.search_bm25("uniqueA", top_k=3, collection="demo")
    assert hits
    store.close()
