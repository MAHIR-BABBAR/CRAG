"""Format-specific document parsers."""

from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder, Parser, content_hash
from custom_rag.ingestion.parsers.code import CodeParser
from custom_rag.ingestion.parsers.composite import build_default_registry
from custom_rag.ingestion.parsers.docx import DOCXParser
from custom_rag.ingestion.parsers.html import HTMLParser
from custom_rag.ingestion.parsers.markdown import MarkdownParser
from custom_rag.ingestion.parsers.pdf import UnstructuredPDFParser, map_elements_to_blocks
from custom_rag.ingestion.parsers.pptx import PPTXParser
from custom_rag.ingestion.parsers.registry import ParserRegistry
from custom_rag.ingestion.parsers.structured import StructuredParser
from custom_rag.ingestion.parsers.text import TextParser

__all__ = [
    "BaseParser",
    "BlockBuilder",
    "CodeParser",
    "DOCXParser",
    "HTMLParser",
    "MarkdownParser",
    "Parser",
    "ParserRegistry",
    "PPTXParser",
    "StructuredParser",
    "TextParser",
    "UnstructuredPDFParser",
    "build_default_registry",
    "content_hash",
    "map_elements_to_blocks",
]
