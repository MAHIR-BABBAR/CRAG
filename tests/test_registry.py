"""Tests for parser registry routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_rag.core.exceptions import UnsupportedFormatError
from custom_rag.ingestion.parsers.code import CodeParser
from custom_rag.ingestion.parsers.composite import build_default_registry
from custom_rag.ingestion.parsers.markdown import MarkdownParser
from custom_rag.ingestion.parsers.text import TextParser

FIXTURES = Path(__file__).parent / "fixtures"


def test_registry_routes_markdown_before_text() -> None:
    registry = build_default_registry()
    parser = registry.get_parser(FIXTURES / "sample.md")
    assert isinstance(parser, MarkdownParser)


def test_registry_routes_code_before_text() -> None:
    registry = build_default_registry()
    parser = registry.get_parser(FIXTURES / "sample.py")
    assert isinstance(parser, CodeParser)


def test_registry_routes_text_file() -> None:
    registry = build_default_registry()
    parser = registry.get_parser(FIXTURES / "sample.txt")
    assert isinstance(parser, TextParser)


def test_registry_raises_for_unknown_extension() -> None:
    registry = build_default_registry()
    with pytest.raises(UnsupportedFormatError):
        registry.get_parser(FIXTURES / "sample.unknown")
