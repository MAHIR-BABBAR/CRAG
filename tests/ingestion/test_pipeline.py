"""Tests for ingestion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_file_merges_file_metadata() -> None:
    path = FIXTURES / "sample.txt"
    document = parse_file(path)

    assert document.metadata.source_path == str(path.resolve())
    assert document.metadata.source_uri.startswith("file://")
    assert document.metadata.file_size > 0
    assert document.metadata.mime_type == "text/plain"
