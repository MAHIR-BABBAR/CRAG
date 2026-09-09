"""Tests for code parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.chunkers import chunk_document
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_code_parser_builds_module_container_and_symbols() -> None:
    document = parse_file(FIXTURES / "sample.py")

    assert document.metadata.doc_type == "code"
    assert document.metadata.language == "python"
    assert document.raw_text is not None
    assert "def greet(name: str) -> str:" in document.raw_text

    module = document.block_by_id("module")
    assert module is not None
    assert module.block_type == BlockType.MODULE
    assert module.text == ""

    containers = [
        block for block in document.blocks if block.block_type == BlockType.CODE_CONTAINER
    ]
    symbols = [block for block in document.blocks if block.block_type == BlockType.CODE_SYMBOL]
    gaps = [block for block in document.blocks if block.block_type == BlockType.CODE_BLOCK]

    assert containers
    greeter = next(block for block in containers if block.metadata.get("symbol_name") == "Greeter")
    assert greeter.text.startswith("class Greeter")
    assert greeter.parent_block_id == "module"

    method_names = {block.metadata.get("symbol_name") for block in symbols}
    assert "greet" in method_names
    assert any(block.metadata.get("qualified_name") == "Greeter.greet" for block in symbols)
    assert any(block.parent_block_id == greeter.block_id for block in symbols)
    assert gaps  # module docstring / imports-style gaps


def test_code_parent_chunks_do_not_duplicate_methods() -> None:
    document = parse_file(FIXTURES / "sample.py")
    chunked = chunk_document(document)

    greeter_parent = next(
        chunk
        for chunk in chunked.parents()
        if chunk.block_type == BlockType.CODE_CONTAINER
        and chunk.metadata.get("symbol_name") == "Greeter"
    )
    assert greeter_parent.text.count("def greet(self, name: str) -> str:") == 1
    assert greeter_parent.text.count("class Greeter") == 1

    module_parent = next(chunk for chunk in chunked.parents() if chunk.source_block_id == "module")
    assert "def greet(self, name: str) -> str:" not in module_parent.text
    assert module_parent.text.count("def greet(name: str) -> str:") == 1
