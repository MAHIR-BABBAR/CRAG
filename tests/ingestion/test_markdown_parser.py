"""Tests for markdown parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_markdown_parser_extracts_frontmatter_and_hierarchy() -> None:
    document = parse_file(FIXTURES / "sample.md")

    assert document.metadata.doc_type == "markdown"
    assert document.metadata.extra["title"] == "CRAG Guide"
    assert document.metadata.extra["version"] == 1

    sections = [block for block in document.blocks if block.block_type == BlockType.SECTION]
    assert len(sections) >= 2

    requirements = next(
        block for block in sections if block.metadata.get("heading_text") == "Requirements"
    )
    assert requirements.parent_block_id is not None

    code_blocks = [block for block in document.blocks if block.block_type == BlockType.CODE_BLOCK]
    assert len(code_blocks) == 1
    assert "pip install" in code_blocks[0].text
