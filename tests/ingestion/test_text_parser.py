"""Tests for text parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_text_parser_builds_document_parent_and_paragraph_children() -> None:
    document = parse_file(FIXTURES / "sample.txt")

    assert document.metadata.doc_type == "text"
    assert document.metadata.content_hash
    assert document.raw_text

    root = document.block_by_id("doc_root")
    assert root is not None
    assert root.block_type == BlockType.DOCUMENT

    paragraphs = document.children_of("doc_root")
    assert len(paragraphs) == 2
    assert all(block.block_type == BlockType.PARAGRAPH for block in paragraphs)
