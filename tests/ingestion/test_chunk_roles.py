"""Tests for chunk role mapping helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.types import BlockType, ContentBlock, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.chunkers.roles import (
    CHILD_TYPES,
    PARENT_TYPES,
    descendant_leaf_blocks,
    has_descendant_leaf,
    is_child_type,
    is_parent_type,
    nearest_parent_block,
    owned_leaf_blocks,
    owning_parent_ids,
)


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


def _document() -> ParsedDocument:
    blocks = [
        ContentBlock(
            block_id="section_1",
            block_type=BlockType.SECTION,
            text="Introduction",
        ),
        ContentBlock(
            block_id="p_1",
            block_type=BlockType.PARAGRAPH,
            text="First paragraph.",
            parent_block_id="section_1",
            order=1,
        ),
        ContentBlock(
            block_id="p_2",
            block_type=BlockType.PARAGRAPH,
            text="Second paragraph.",
            parent_block_id="section_1",
            order=2,
        ),
    ]
    return ParsedDocument(metadata=_metadata(), blocks=blocks)


def test_parent_and_child_type_sets() -> None:
    assert BlockType.SECTION in PARENT_TYPES
    assert BlockType.CODE_CONTAINER in PARENT_TYPES
    assert BlockType.PARAGRAPH in CHILD_TYPES
    assert is_parent_type(BlockType.MODULE)
    assert is_child_type(BlockType.CODE_SYMBOL)
    assert not is_parent_type(BlockType.PARAGRAPH)


def test_nearest_parent_block_walks_up_chain() -> None:
    document = _document()
    blocks_by_id = {block.block_id: block for block in document.blocks}
    leaf = document.block_by_id("p_2")
    assert leaf is not None

    ancestor = nearest_parent_block(document, leaf, blocks_by_id)
    assert ancestor is not None
    assert ancestor.block_id == "section_1"


def test_descendant_leaf_blocks_and_has_descendant_leaf() -> None:
    document = _document()
    blocks_by_id = {block.block_id: block for block in document.blocks}

    leaves = descendant_leaf_blocks(document, "section_1", blocks_by_id)
    assert [leaf.block_id for leaf in leaves] == ["p_1", "p_2"]
    assert has_descendant_leaf(document, "section_1", blocks_by_id)


def test_owned_leaf_blocks_stop_at_nested_parents() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(block_id="page_1", block_type=BlockType.PAGE, text=""),
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
    owned = owned_leaf_blocks(document, "page_1", blocks_by_id)
    assert [leaf.block_id for leaf in owned] == ["p_page"]
    assert owning_parent_ids(document, blocks_by_id) == {"page_1", "section_1"}
