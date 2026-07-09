"""Markdown document parser with heading hierarchy and frontmatter."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```")


class MarkdownParser(BaseParser):
    name = "markdown"
    supported_extensions = frozenset({".md", ".mdx", ".markdown"})
    supported_mimes = frozenset({"text/markdown"})

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        raw = self.read_bytes(path)
        source, encoding = self.decode_text(raw, path, metadata.encoding)
        post = frontmatter.loads(source)
        body = post.content

        builder = BlockBuilder()
        section_stack: list[tuple[str, int]] = []
        section_counter = 0
        paragraph_counter = 0
        code_counter = 0
        in_code_fence = False
        code_lines: list[str] = []

        doc_metadata = metadata.model_copy(
            update={
                "doc_type": "markdown",
                "encoding": encoding,
                "extra": {**metadata.extra, **dict(post.metadata)},
            }
        )

        for line in body.splitlines():
            if _FENCE_RE.match(line):
                if in_code_fence:
                    code_counter += 1
                    parent_id, hierarchy = _current_parent(section_stack)
                    block_id = f"code_{code_counter}"
                    builder.add(
                        block_id=block_id,
                        block_type=BlockType.CODE_BLOCK,
                        text="\n".join(code_lines).strip(),
                        parent_block_id=parent_id,
                        order=code_counter,
                        hierarchy_path=[*hierarchy, block_id],
                    )
                    code_lines = []
                    in_code_fence = False
                else:
                    in_code_fence = True
                continue

            if in_code_fence:
                code_lines.append(line)
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                section_counter += 1
                block_id = f"sec_{section_counter}"
                slug = BlockBuilder.slugify(title)

                while section_stack and section_stack[-1][1] >= level:
                    section_stack.pop()

                parent_id = section_stack[-1][0] if section_stack else None
                hierarchy = ["doc"]
                hierarchy.extend(item[0] for item in section_stack)
                hierarchy.append(block_id)

                builder.add(
                    block_id=block_id,
                    block_type=BlockType.SECTION,
                    text=title,
                    parent_block_id=parent_id,
                    order=section_counter,
                    hierarchy_path=hierarchy,
                    metadata={"heading_level": level, "heading_text": title, "slug": slug},
                )
                section_stack.append((block_id, level))
                continue

            if not line.strip():
                continue

            paragraph_counter += 1
            parent_id, hierarchy = _current_parent(section_stack)
            block_id = f"p_{paragraph_counter}"
            builder.add(
                block_id=block_id,
                block_type=BlockType.PARAGRAPH,
                text=line.strip(),
                parent_block_id=parent_id,
                order=paragraph_counter,
                hierarchy_path=[*hierarchy, block_id],
            )

        if in_code_fence and code_lines:
            code_counter += 1
            parent_id, hierarchy = _current_parent(section_stack)
            block_id = f"code_{code_counter}"
            builder.add(
                block_id=block_id,
                block_type=BlockType.CODE_BLOCK,
                text="\n".join(code_lines).strip(),
                parent_block_id=parent_id,
                order=code_counter,
                hierarchy_path=[*hierarchy, block_id],
            )

        return ParsedDocument(metadata=doc_metadata, blocks=builder.blocks, raw_text=body)


def _current_parent(section_stack: list[tuple[str, int]]) -> tuple[str | None, list[str]]:
    if not section_stack:
        return None, ["doc"]
    hierarchy = ["doc", *[item[0] for item in section_stack]]
    return section_stack[-1][0], hierarchy
