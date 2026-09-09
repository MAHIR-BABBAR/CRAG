"""Tests for document chunker orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.types import BlockType, ContentBlock, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.chunkers import ChunkConfig, chunk_document


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


def _markdown_like_document() -> ParsedDocument:
    return ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="section_1",
                block_type=BlockType.SECTION,
                text="Introduction",
                order=0,
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
            ContentBlock(
                block_id="section_2",
                block_type=BlockType.SECTION,
                text="Details",
                order=3,
            ),
            ContentBlock(
                block_id="p_3",
                block_type=BlockType.PARAGRAPH,
                text="Third paragraph.",
                parent_block_id="section_2",
                order=4,
            ),
        ],
    )


def test_chunk_document_emits_parent_and_child_chunks() -> None:
    chunked = chunk_document(_markdown_like_document())

    parents = chunked.parents()
    children = chunked.children()
    assert len(parents) == 2
    assert len(children) == 3
    assert chunked.validate_chunk_graph() is chunked


def test_child_chunks_link_to_nearest_parent_chunk() -> None:
    chunked = chunk_document(_markdown_like_document())
    child = next(chunk for chunk in chunked.children() if chunk.source_block_id == "p_2")
    parent = chunked.parent_of(child.chunk_id)
    assert parent is not None
    assert parent.chunk_id == "parent_section_1"
    assert "Introduction" in parent.text
    assert "Second paragraph." in parent.text


def test_oversized_child_block_is_split_with_shared_parent() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="doc_root",
                block_type=BlockType.DOCUMENT,
                text="Document",
            ),
            ContentBlock(
                block_id="p_big",
                block_type=BlockType.PARAGRAPH,
                text="word " * 400,
                parent_block_id="doc_root",
                order=1,
            ),
        ],
    )
    chunked = chunk_document(
        document,
        config=ChunkConfig(child_max_chars=100, child_overlap_chars=10, min_child_chars=0),
    )
    split_children = [chunk for chunk in chunked.children() if chunk.source_block_id == "p_big"]
    assert len(split_children) > 1
    assert {chunk.parent_chunk_id for chunk in split_children} == {"parent_doc_root"}
    assert all(chunk.split_index > 0 for chunk in split_children)
    assert all(chunk.metadata["is_split"] for chunk in split_children)


def test_empty_leaf_blocks_produce_no_child_chunks() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="section_1",
                block_type=BlockType.SECTION,
                text="Only title",
            ),
            ContentBlock(
                block_id="p_empty",
                block_type=BlockType.PARAGRAPH,
                text="   ",
                parent_block_id="section_1",
            ),
        ],
    )
    chunked = chunk_document(document)
    assert chunked.children() == []


def test_chunker_emits_only_owning_parents_for_nested_page_section() -> None:
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
    chunked = chunk_document(document)
    parent_ids = {chunk.chunk_id for chunk in chunked.parents()}
    assert parent_ids == {"parent_page_1", "parent_section_1"}

    page_parent = chunked.chunk_by_id("parent_page_1")
    section_parent = chunked.chunk_by_id("parent_section_1")
    assert page_parent is not None
    assert section_parent is not None
    assert page_parent.text == "Page intro."
    assert section_parent.text == "Details\n\nSection body."
    assert page_parent.text.count("Page intro.") == 1
    assert "Section body." not in page_parent.text

    child_page = next(c for c in chunked.children() if c.source_block_id == "p_page")
    child_section = next(c for c in chunked.children() if c.source_block_id == "p_section")
    assert child_page.parent_chunk_id == "parent_page_1"
    assert child_section.parent_chunk_id == "parent_section_1"
