"""Integration tests for parse + chunk + embed pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion import embed_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "filename",
    ["sample.md", "sample.txt", "sample.py"],
)
def test_embed_file_on_fixtures_with_mock_provider(filename: str) -> None:
    embedded = embed_file(FIXTURES / filename, provider=MockEmbeddingProvider())
    assert embedded.metadata.doc_type
    assert embedded.skipped is False
    assert embedded.children
    assert embedded.parents
    assert all(len(item.embedding) == 8 for item in embedded.children)
    assert embedded.embedded_child_by_id(embedded.children[0].chunk.chunk_id) is not None
