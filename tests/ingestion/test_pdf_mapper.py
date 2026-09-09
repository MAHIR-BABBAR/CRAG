"""Tests for PDF element mapper."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.chunkers import chunk_document
from custom_rag.ingestion.parsers.pdf.mapper import accepted_element_texts, map_elements_to_blocks


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
    assert page_one.text == ""

    sections = [block for block in blocks if block.block_type == BlockType.SECTION]
    assert len(sections) == 1
    assert sections[0].parent_block_id == "page_1"

    paragraphs = [block for block in blocks if block.block_type == BlockType.PARAGRAPH]
    assert len(paragraphs) == 2
    assert paragraphs[0].parent_block_id == sections[0].block_id


def test_pdf_mapper_table_emits_table_parent_and_row_child() -> None:
    elements = [
        {
            "type": "Table",
            "text": "Region | Revenue",
            "metadata": {"page_number": 1},
        }
    ]
    blocks = map_elements_to_blocks(elements)
    tables = [block for block in blocks if block.block_type == BlockType.TABLE]
    rows = [block for block in blocks if block.block_type == BlockType.TABLE_ROW]
    assert len(tables) == 1
    assert tables[0].text == ""
    assert len(rows) == 1
    assert rows[0].parent_block_id == tables[0].block_id
    assert rows[0].text == "Region | Revenue"


def test_pdf_raw_text_from_accepted_elements_is_non_duplicating() -> None:
    elements = [
        {"type": "Title", "text": "Heading", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "Body text.", "metadata": {"page_number": 1}},
        {"type": "Footer", "text": "page footer", "metadata": {"page_number": 1}},
    ]
    texts = accepted_element_texts(elements)
    assert texts == ["Heading", "Body text."]
    raw = "\n\n".join(texts)
    assert raw.count("Heading") == 1
    assert raw.count("Body text.") == 1


def test_pdf_chunk_ownership_avoids_page_section_duplication() -> None:
    elements = [
        {"type": "Title", "text": "Annual Report", "metadata": {"page_number": 1}},
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
    chunked = chunk_document(ParsedDocument(metadata=metadata, blocks=blocks))
    section_parent = next(
        chunk for chunk in chunked.parents() if chunk.block_type == BlockType.SECTION
    )
    assert section_parent.text.count("Revenue increased 12 percent.") == 1
    page_parents = [chunk for chunk in chunked.parents() if chunk.block_type == BlockType.PAGE]
    assert all("Revenue increased 12 percent." not in chunk.text for chunk in page_parents)
