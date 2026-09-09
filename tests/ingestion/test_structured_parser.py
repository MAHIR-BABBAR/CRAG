"""Tests for structured parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.chunkers import chunk_document
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_json_parser_emits_record_blocks() -> None:
    document = parse_file(FIXTURES / "sample.json")

    assert document.metadata.doc_type == "json"
    root = document.block_by_id("doc_root")
    assert root is not None
    assert root.text == ""
    assert document.raw_text
    records = [block for block in document.blocks if block.block_type == BlockType.RECORD]
    assert len(records) == 3
    assert all(record.parent_block_id == "doc_root" for record in records)


def test_csv_parser_emits_table_row_blocks() -> None:
    document = parse_file(FIXTURES / "sample.csv")

    assert document.metadata.doc_type == "csv"
    root = document.block_by_id("doc_root")
    assert root is not None
    assert root.text == ""
    assert document.raw_text
    rows = [block for block in document.blocks if block.block_type == BlockType.TABLE_ROW]
    assert len(rows) == 2
    assert "Alice" in rows[0].text
    assert rows[0].metadata["row_index"] == 0


def test_structured_parent_chunk_does_not_repeat_raw_source() -> None:
    document = parse_file(FIXTURES / "sample.csv")
    chunked = chunk_document(document)
    parent = next(chunk for chunk in chunked.parents() if chunk.source_block_id == "doc_root")
    assert parent.text.count("Alice") == 1
    assert parent.text.count("Bob") == 1
    assert document.raw_text is not None
    # Parent uses normalized row text, not the raw CSV bytes.
    assert "name,role" not in parent.text.lower() or parent.text != document.raw_text.strip()
