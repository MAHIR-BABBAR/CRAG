"""Tests for PDF element mapper."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.pdf.mapper import map_elements_to_blocks


def test_pdf_mapper_builds_page_parent_and_paragraph_children() -> None:
    elements = [
        {
            "type": "Title",
            "text": "Annual Report",
            "metadata": {"page_number": 1},
        },
        {
            "type": "NarrativeText",
            "text": "Revenue increased 12 percent.",
            "metadata": {"page_number": 1},
        },
        {
            "type": "NarrativeText",
            "text": "Operating costs decreased.",
            "metadata": {"page_number": 2},
        },
    ]

    blocks = map_elements_to_blocks(elements)
    metadata = DocumentMetadata(
        doc_id="1",
        source_path="/tmp/report.pdf",
        source_uri="file:///tmp/report.pdf",
        doc_type="pdf",
        content_hash="abc",
        file_size=1,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    document = ParsedDocument(metadata=metadata, blocks=blocks)

    page_one = document.block_by_id("page_1")
    assert page_one is not None
    assert page_one.block_type == BlockType.PAGE
    assert "Revenue increased" in page_one.text

    sections = [block for block in blocks if block.block_type == BlockType.SECTION]
    assert len(sections) == 1
    assert sections[0].parent_block_id == "page_1"

    paragraphs = [block for block in blocks if block.block_type == BlockType.PARAGRAPH]
    assert len(paragraphs) == 2
    assert paragraphs[0].parent_block_id == sections[0].block_id
