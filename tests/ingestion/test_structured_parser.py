"""Tests for structured parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_json_parser_emits_record_blocks() -> None:
    document = parse_file(FIXTURES / "sample.json")

    assert document.metadata.doc_type == "json"
    records = [block for block in document.blocks if block.block_type == BlockType.RECORD]
    assert len(records) == 3
    assert all(record.parent_block_id == "doc_root" for record in records)


def test_csv_parser_emits_table_row_blocks() -> None:
    document = parse_file(FIXTURES / "sample.csv")

    assert document.metadata.doc_type == "csv"
    rows = [block for block in document.blocks if block.block_type == BlockType.TABLE_ROW]
    assert len(rows) == 2
    assert "Alice" in rows[0].text
    assert rows[0].metadata["row_index"] == 0
