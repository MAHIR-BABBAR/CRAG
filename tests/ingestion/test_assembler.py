"""Tests for parent chunk text assembly."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.types import BlockType, ContentBlock, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.chunkers.assembler import assemble_parent_text


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path="/tmp/sample.md",
        source_uri="file:///tmp/sample.md",
        doc_type="markdown",
        content_hash="abc123",
        file_size=128,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_assemble_parent_text_exact_title_and_owned_children() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="section_1",
                block_type=BlockType.SECTION,
                text="Installation",
            ),
            ContentBlock(
                block_id="p_1",
                block_type=BlockType.PARAGRAPH,
                text="Install with pip.",
                parent_block_id="section_1",
                order=1,
            ),
        ],
    )
    blocks_by_id = {block.block_id: block for block in document.blocks}
    section = document.block_by_id("section_1")
    assert section is not None

    assembled = assemble_parent_text(
        document,
        section,
        blocks_by_id,
        parent_max_chars=6000,
    )
    assert assembled == "Installation\n\nInstall with pip."
    assert assembled.count("Install with pip.") == 1


def test_assemble_parent_text_skips_nested_parent_subtree() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="page_1",
                block_type=BlockType.PAGE,
                text="",
            ),
            ContentBlock(
                block_id="p_page",
                block_type=BlockType.PARAGRAPH,
                text="Page intro.",
                parent_block_id="page_1",
                order=1,
            ),
            ContentBlock(
                block_id="section_1",
                block_type=BlockType.SECTION,
                text="Details",
                parent_block_id="page_1",
                order=2,
            ),
            ContentBlock(
                block_id="p_section",
                block_type=BlockType.PARAGRAPH,
                text="Section body.",
                parent_block_id="section_1",
                order=3,
            ),
        ],
    )
    blocks_by_id = {block.block_id: block for block in document.blocks}
    page = document.block_by_id("page_1")
    assert page is not None

    assembled = assemble_parent_text(document, page, blocks_by_id, parent_max_chars=6000)
    assert assembled == "Page intro."
    assert "Section body." not in assembled
    assert assembled.count("Page intro.") == 1


def test_assemble_parent_text_empty_container_title_only() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="doc_root",
                block_type=BlockType.DOCUMENT,
                text="",
            ),
            ContentBlock(
                block_id="p_1",
                block_type=BlockType.PARAGRAPH,
                text="Only child.",
                parent_block_id="doc_root",
                order=1,
            ),
        ],
    )
    blocks_by_id = {block.block_id: block for block in document.blocks}
    root = document.block_by_id("doc_root")
    assert root is not None
    assembled = assemble_parent_text(document, root, blocks_by_id, parent_max_chars=6000)
    assert assembled == "Only child."


def test_assemble_parent_text_truncates_at_parent_max_chars() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="section_1",
                block_type=BlockType.SECTION,
                text="Title",
            ),
            ContentBlock(
                block_id="p_1",
                block_type=BlockType.PARAGRAPH,
                text="x" * 200,
                parent_block_id="section_1",
                order=1,
            ),
        ],
    )
    blocks_by_id = {block.block_id: block for block in document.blocks}
    section = document.block_by_id("section_1")
    assert section is not None

    assembled = assemble_parent_text(
        document,
        section,
        blocks_by_id,
        parent_max_chars=50,
    )
    assert len(assembled) <= 50
    assert assembled.endswith("[...]")
