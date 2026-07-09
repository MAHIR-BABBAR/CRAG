"""HTML document parser."""

from __future__ import annotations

import re
from pathlib import Path

from custom_rag.core.exceptions import ParseFailedError
from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_STRIP_TAGS = {"script", "style", "nav", "footer", "noscript"}


class HTMLParser(BaseParser):
    name = "html"
    supported_extensions = frozenset({".html", ".htm"})
    supported_mimes = frozenset({"text/html"})

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ParseFailedError(
                str(path),
                "beautifulsoup4 is not installed; use pip install 'custom-rag[ingestion]'",
                cause=exc,
            ) from exc

        raw = self.read_bytes(path)
        text, encoding = self.decode_text(raw, path, metadata.encoding)

        try:
            soup = BeautifulSoup(text, "lxml")
        except Exception as exc:
            raise ParseFailedError(str(path), "unable to parse HTML", cause=exc) from exc

        for tag_name in _STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        builder = BlockBuilder()
        section_stack: list[tuple[str, int]] = []
        section_counter = 0
        paragraph_counter = 0
        code_counter = 0
        raw_parts: list[str] = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "li"]):
            content = _clean_text(element.get_text("\n", strip=True))
            if not content:
                continue

            tag = element.name.lower()
            if tag in _HEADING_TAGS:
                section_counter += 1
                block_id = f"sec_{section_counter}"
                level = int(tag[1])
                while section_stack and section_stack[-1][1] >= level:
                    section_stack.pop()
                parent_id = section_stack[-1][0] if section_stack else None
                hierarchy = ["doc", *[item[0] for item in section_stack], block_id]
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.SECTION,
                    text=content,
                    parent_block_id=parent_id,
                    order=section_counter,
                    hierarchy_path=hierarchy,
                    metadata={"heading_tag": tag, "heading_level": level},
                )
                section_stack.append((block_id, level))
                raw_parts.append(content)
                continue

            parent_id, hierarchy = _current_parent(section_stack)
            if tag == "pre":
                code_counter += 1
                block_id = f"code_{code_counter}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.CODE_BLOCK,
                    text=content,
                    parent_block_id=parent_id,
                    order=code_counter,
                    hierarchy_path=[*hierarchy, block_id],
                )
            elif tag == "li":
                paragraph_counter += 1
                block_id = f"li_{paragraph_counter}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.LIST_ITEM,
                    text=content,
                    parent_block_id=parent_id,
                    order=paragraph_counter,
                    hierarchy_path=[*hierarchy, block_id],
                )
            else:
                paragraph_counter += 1
                block_id = f"p_{paragraph_counter}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.PARAGRAPH,
                    text=content,
                    parent_block_id=parent_id,
                    order=paragraph_counter,
                    hierarchy_path=[*hierarchy, block_id],
                )
            raw_parts.append(content)

        doc_metadata = metadata.model_copy(update={"doc_type": "html", "encoding": encoding})
        return ParsedDocument(
            metadata=doc_metadata,
            blocks=builder.blocks,
            raw_text="\n\n".join(raw_parts) if raw_parts else None,
        )


def _current_parent(section_stack: list[tuple[str, int]]) -> tuple[str | None, list[str]]:
    if not section_stack:
        return None, ["doc"]
    hierarchy = ["doc", *[item[0] for item in section_stack]]
    return section_stack[-1][0], hierarchy


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
