"""Tests for text parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.chunkers import chunk_document
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_text_parser_builds_document_parent_and_paragraph_children() -> None:
    document = parse_file(FIXTURES / "sample.txt")

    assert document.metadata.doc_type == "text"
    assert document.metadata.content_hash
    assert document.raw_text is not None
    assert document.raw_text.strip() == "First paragraph.\n\nSecond paragraph."

    root = document.block_by_id("doc_root")
    assert root is not None
    assert root.block_type == BlockType.DOCUMENT
    assert root.text == ""

    paragraphs = document.children_of("doc_root")
    assert len(paragraphs) == 2
    assert all(block.block_type == BlockType.PARAGRAPH for block in paragraphs)


def test_text_parent_chunk_does_not_duplicate_paragraphs() -> None:
    document = parse_file(FIXTURES / "sample.txt")
    chunked = chunk_document(document)
    parent = next(chunk for chunk in chunked.parents() if chunk.source_block_id == "doc_root")
    assert parent.text.count("First paragraph.") == 1
    assert parent.text.count("Second paragraph.") == 1
    assert parent.text == "First paragraph.\n\nSecond paragraph."
