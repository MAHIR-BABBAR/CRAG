"""Recursive directory indexing helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.ingestion.embedders import EmbedConfig
from custom_rag.ingestion.pipeline import index_file
from custom_rag.storage.sqlite_store import SQLiteIndexStore


@lru_cache(maxsize=1)
def default_index_extensions() -> frozenset[str]:
    """Extensions the default parser registry can handle.

    Derived from the registry rather than duplicated here, so registering a
    parser is enough to make directory indexing pick up its files.
    """
    from custom_rag.ingestion.parsers.composite import build_default_registry

    return build_default_registry().supported_extensions


_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
    }
)


@dataclass
class IndexFileResult:
    path: str
    skipped: bool
    doc_id: str | None = None
    children: int = 0
    parents: int = 0
    error: str | None = None


@dataclass
class IndexDirectoryResult:
    indexed: int = 0
    skipped: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    files: list[IndexFileResult] = field(default_factory=list)
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "indexed": self.indexed,
            "skipped": self.skipped,
            "failed": self.failed,
            "file_count": len(self.files),
            "truncated": self.truncated,
        }


def iter_indexable_files(
    root: Path,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    max_files: int | None = None,
) -> tuple[list[Path], bool]:
    """Collect indexable files under ``root``.

    Returns the files in a stable, sorted order along with a flag that is true
    when ``max_files`` cut the walk short, so callers can tell a complete
    directory from a partial one.
    """
    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (extensions or default_index_extensions())
    }
    root = root.resolve()

    def _should_skip_dir(name: str) -> bool:
        return name in _SKIP_DIR_NAMES or name.startswith(".")

    if root.is_file():
        return ([root], False) if root.suffix.lower() in allowed else ([], False)

    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not _should_skip_dir(d))
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                path = Path(dirpath) / name
                if path.suffix.lower() in allowed:
                    found.append(path)
    else:
        for path in sorted(root.iterdir()):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in allowed:
                found.append(path)

    if max_files is not None and len(found) > max_files:
        return found[:max_files], True
    return found, False


def index_path(
    path: Path,
    *,
    store: SQLiteIndexStore,
    provider: EmbeddingProvider | None = None,
    embed_config: EmbedConfig | None = None,
    collection: str | None = None,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    max_files: int = 500,
) -> IndexDirectoryResult:
    """Index a file or directory; returns aggregate counts."""
    path = path.expanduser().resolve()
    result = IndexDirectoryResult()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files, result.truncated = iter_indexable_files(
            path,
            recursive=recursive,
            extensions=extensions,
            max_files=max_files,
        )
    else:
        result.failed.append({"path": str(path), "error": "path not found"})
        return result

    for file_path in files:
        try:
            embedded = index_file(
                file_path,
                embed_config=embed_config,
                provider=provider,
                store=store,
                collection=collection,
            )
            children = len(embedded.children)
            parents = len(embedded.parents)
            doc_id = embedded.metadata.doc_id
            if embedded.skipped:
                counts = store.get_indexed_counts(str(file_path))
                if counts is not None:
                    doc_id, children, parents = counts
                result.skipped += 1
            else:
                result.indexed += 1
            result.files.append(
                IndexFileResult(
                    path=str(file_path),
                    skipped=embedded.skipped,
                    doc_id=doc_id,
                    children=children,
                    parents=parents,
                )
            )
        except Exception as exc:
            result.failed.append({"path": str(file_path), "error": str(exc)})
            result.files.append(IndexFileResult(path=str(file_path), skipped=False, error=str(exc)))
    return result
