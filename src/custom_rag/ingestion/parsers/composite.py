"""Default parser registry wiring."""

from __future__ import annotations

import os
from typing import cast

from custom_rag.ingestion.parsers.code import CodeParser
from custom_rag.ingestion.parsers.docx import DOCXParser
from custom_rag.ingestion.parsers.html import HTMLParser
from custom_rag.ingestion.parsers.markdown import MarkdownParser
from custom_rag.ingestion.parsers.pdf.unstructured import PdfStrategy, UnstructuredPDFParser
from custom_rag.ingestion.parsers.pptx import PPTXParser
from custom_rag.ingestion.parsers.registry import ParserRegistry
from custom_rag.ingestion.parsers.structured import StructuredParser
from custom_rag.ingestion.parsers.text import TextParser


def build_default_registry() -> ParserRegistry:
    registry = ParserRegistry()

    strategy_value = os.getenv("UNSTRUCTURED_PDF_STRATEGY", "hi_res")
    if strategy_value not in {"vlm", "hi_res", "auto"}:
        strategy_value = "hi_res"
    pdf_strategy = cast(PdfStrategy, strategy_value)

    registry.register(UnstructuredPDFParser(strategy=pdf_strategy))
    registry.register(PPTXParser())
    registry.register(DOCXParser())
    registry.register(CodeParser())
    registry.register(HTMLParser())
    registry.register(MarkdownParser())
    registry.register(StructuredParser())
    registry.register(TextParser())
    return registry
