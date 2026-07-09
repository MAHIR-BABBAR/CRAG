"""Tests for HTML parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_rag.core.types import BlockType
from custom_rag.ingestion.pipeline import parse_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_html_parser_builds_section_hierarchy() -> None:
    pytest.importorskip("bs4")
    document = parse_file(FIXTURES / "sample.html")

    assert document.metadata.doc_type == "html"
    sections = [block for block in document.blocks if block.block_type == BlockType.SECTION]
    assert len(sections) >= 2

    paragraphs = [block for block in document.blocks if block.block_type == BlockType.PARAGRAPH]
    assert paragraphs
    assert paragraphs[0].parent_block_id is not None

    code_blocks = [block for block in document.blocks if block.block_type == BlockType.CODE_BLOCK]
    assert len(code_blocks) == 1
