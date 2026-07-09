"""Tests for code parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_code_parser_builds_module_parent_and_symbol_children() -> None:
    document = parse_file(FIXTURES / "sample.py")

    assert document.metadata.doc_type == "code"
    assert document.metadata.language == "python"

    module = document.block_by_id("module")
    assert module is not None
    assert module.block_type == BlockType.MODULE

    symbols = [block for block in document.blocks if block.block_type == BlockType.CODE_SYMBOL]
    assert symbols
    assert all(symbol.parent_block_id == "module" for symbol in symbols)
    kinds = {symbol.metadata.get("symbol_kind") for symbol in symbols}
    names = {symbol.metadata.get("symbol_name") for symbol in symbols}
    assert "greet" in names or "Greeter" in names or "line_window" in kinds
