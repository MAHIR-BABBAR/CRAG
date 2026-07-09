"""Plain-text document parser."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder


class TextParser(BaseParser):
    name = "text"
    supported_extensions = frozenset({".txt", ".log", ".rst", ".text"})
    supported_mimes = frozenset({"text/plain"})

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        raw = self.read_bytes(path)
        text, encoding = self.decode_text(raw, path, metadata.encoding)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = _split_paragraphs(text)

        builder = BlockBuilder()
        builder.add(
            block_id="doc_root",
            block_type=BlockType.DOCUMENT,
            text=text,
            hierarchy_path=["doc"],
        )

        for index, paragraph in enumerate(paragraphs):
            builder.add(
                block_id=f"p_{index}",
                block_type=BlockType.PARAGRAPH,
                text=paragraph,
                parent_block_id="doc_root",
                order=index,
                hierarchy_path=["doc", f"p_{index}"],
            )

        doc_metadata = metadata.model_copy(update={"doc_type": "text", "encoding": encoding})
        return ParsedDocument(metadata=doc_metadata, blocks=builder.blocks, raw_text=text)


def _split_paragraphs(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in text.split("\n\n")]
    return [chunk for chunk in chunks if chunk]
