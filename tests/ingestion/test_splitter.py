"""Tests for recursive character splitter."""

from __future__ import annotations

from custom_rag.ingestion.chunkers.splitter import split_text


def test_split_text_returns_single_chunk_for_short_text() -> None:
    parts = split_text("short text", max_chars=100, overlap_chars=10)
    assert parts == ["short text"]


def test_split_text_splits_on_paragraph_boundary() -> None:
    text = "alpha\n\nbeta\n\ngamma"
    parts = split_text(text, max_chars=10, overlap_chars=0)
    assert len(parts) >= 2
    assert all(len(part) <= 10 for part in parts)


def test_split_text_applies_overlap_between_parts() -> None:
    text = "abcdefghij" * 5
    parts = split_text(text, max_chars=20, overlap_chars=5, min_chars=0)
    assert len(parts) > 1
    for index in range(1, len(parts)):
        assert parts[index - 1][-5:] in parts[index] or parts[index] in text


def test_split_text_hard_splits_when_no_separator_fits() -> None:
    text = "x" * 50
    parts = split_text(text, max_chars=20, overlap_chars=0, min_chars=0)
    assert parts == ["x" * 20, "x" * 20, "x" * 10]
