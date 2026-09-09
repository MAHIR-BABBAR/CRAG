"""Integration tests for parse + chunk pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_rag.ingestion import chunk_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "filename",
    ["sample.md", "sample.txt", "sample.py"],
)
def test_chunk_file_on_fixtures(filename: str) -> None:
    chunked = chunk_file(FIXTURES / filename)
    assert chunked.metadata.doc_type
    assert chunked.children()
    assert chunked.parents()
    assert chunked.validate_chunk_graph() is chunked
    for child in chunked.children():
        assert child.parent_chunk_id is not None


def test_chunk_file_markdown_links_children_to_section_parents() -> None:
    chunked = chunk_file(FIXTURES / "sample.md")
    parents = {chunk.chunk_id for chunk in chunked.parents()}
    for child in chunked.children():
        assert child.parent_chunk_id in parents


def test_chunk_file_code_produces_module_and_container_parents() -> None:
    chunked = chunk_file(FIXTURES / "sample.py")
    parent_ids = {chunk.chunk_id for chunk in chunked.parents()}
    assert "parent_module" in parent_ids
    assert any(chunk.block_type.value == "code_container" for chunk in chunked.parents())
    assert chunked.children()
    for child in chunked.children():
        assert child.parent_chunk_id in parent_ids


def test_chunk_file_text_parent_has_no_duplicated_paragraphs() -> None:
    chunked = chunk_file(FIXTURES / "sample.txt")
    parent = chunked.parents()[0]
    assert parent.text.count("First paragraph.") == 1
    assert parent.text.count("Second paragraph.") == 1
