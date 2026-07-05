"""Default parser registry wiring."""

from __future__ import annotations

from custom_rag.ingestion.parsers.markdown import MarkdownParser
from custom_rag.ingestion.parsers.registry import ParserRegistry
from custom_rag.ingestion.parsers.structured import StructuredParser
from custom_rag.ingestion.parsers.text import TextParser


def build_default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    # Specific parsers before generic text fallback.
    registry.register(MarkdownParser())
    registry.register(StructuredParser())
    registry.register(TextParser())
    return registry
