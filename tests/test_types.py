"""Tests for core domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from custom_rag.core.types import BlockType, ContentBlock, DocumentMetadata, ParsedDocument


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path="/tmp/sample.txt",
        source_uri="file:///tmp/sample.txt",
        doc_type="text",
        content_hash="abc123",
        file_size=128,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_parsed_document_rejects_unknown_parent() -> None:
    with pytest.raises(ValidationError):
        ParsedDocument(
            metadata=_metadata(),
            blocks=[
                ContentBlock(
                    block_id="child",
                    block_type=BlockType.PARAGRAPH,
                    text="hello",
                    parent_block_id="missing",
                )
            ],
        )


def test_parsed_document_children_of() -> None:
    document = ParsedDocument(
        metadata=_metadata(),
        blocks=[
            ContentBlock(
                block_id="doc_root",
                block_type=BlockType.DOCUMENT,
                text="full",
            ),
            ContentBlock(
                block_id="p_0",
                block_type=BlockType.PARAGRAPH,
                text="hello",
                parent_block_id="doc_root",
            ),
        ],
    )

    children = document.children_of("doc_root")
    assert len(children) == 1
    assert children[0].block_id == "p_0"
