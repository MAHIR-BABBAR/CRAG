"""Source code parser with tree-sitter AST boundaries and non-overlapping spans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from custom_rag.core.types import BlockLocation, BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder

_EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
}

_SYMBOL_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "async_function_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "interface_declaration",
        "type_alias_declaration",
    },
}

_CONTAINER_KINDS = frozenset(
    {
        "class_definition",
        "class_declaration",
        "interface_declaration",
    }
)


@dataclass
class _AstSymbol:
    name: str
    kind: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    children: list[_AstSymbol] = field(default_factory=list)


class CodeParser(BaseParser):
    name = "code"
    supported_extensions = frozenset(_EXTENSION_LANGUAGE.keys())
    supported_mimes = frozenset({"text/x-python", "application/javascript", "text/javascript"})

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        raw = self.read_bytes(path)
        source, encoding = self.decode_text(raw, path, metadata.encoding)
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        language = _EXTENSION_LANGUAGE.get(path.suffix.lower(), "unknown")
        source_bytes = source.encode("utf-8")

        builder = BlockBuilder()
        builder.add(
            block_id="module",
            block_type=BlockType.MODULE,
            text="",
            hierarchy_path=["doc", "module"],
            metadata={"language": language},
        )

        roots = _extract_symbol_tree(source, language)
        if roots:
            _emit_symbol_forest(
                builder,
                source_bytes=source_bytes,
                symbols=roots,
                parent_id="module",
                hierarchy=["doc", "module"],
                language=language,
                qualified_prefix="",
                order_start=0,
            )
            _emit_gaps(
                builder,
                source_bytes=source_bytes,
                covered={(symbol.start_byte, symbol.end_byte) for symbol in roots},
                parent_id="module",
                hierarchy=["doc", "module"],
                language=language,
                range_start=0,
                range_end=len(source_bytes),
                order_base=10_000,
            )
        else:
            _add_line_windows(builder, source)

        doc_metadata = metadata.model_copy(
            update={"doc_type": "code", "language": language, "encoding": encoding}
        )
        return ParsedDocument(metadata=doc_metadata, blocks=builder.blocks, raw_text=source)


def _extract_symbol_tree(source: str, language: str) -> list[_AstSymbol]:
    parser = _load_tree_sitter(language)
    if parser is None:
        return []

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return []

    node_types = _SYMBOL_NODE_TYPES.get(language, set())
    source_bytes = source.encode("utf-8")
    flat: list[_AstSymbol] = []

    def walk(node: object) -> None:
        node_type = getattr(node, "type", "")
        if node_type in node_types:
            symbol = _node_to_symbol(node, source_bytes)
            if symbol is not None:
                flat.append(symbol)
        for child in getattr(node, "children", []):
            walk(child)

    walk(tree.root_node)
    return _nest_symbols(flat)


def _nest_symbols(flat: list[_AstSymbol]) -> list[_AstSymbol]:
    if not flat:
        return []
    ordered = sorted(flat, key=lambda item: (item.start_byte, -item.end_byte))
    roots: list[_AstSymbol] = []
    stack: list[_AstSymbol] = []

    for symbol in ordered:
        while stack and not (
            stack[-1].start_byte <= symbol.start_byte and symbol.end_byte <= stack[-1].end_byte
        ):
            stack.pop()
        if stack:
            stack[-1].children.append(symbol)
        else:
            roots.append(symbol)
        stack.append(symbol)
    return roots


def _emit_symbol_forest(
    builder: BlockBuilder,
    *,
    source_bytes: bytes,
    symbols: list[_AstSymbol],
    parent_id: str,
    hierarchy: list[str],
    language: str,
    qualified_prefix: str,
    order_start: int,
) -> int:
    order = order_start
    for symbol in symbols:
        qualified = f"{qualified_prefix}.{symbol.name}" if qualified_prefix else symbol.name
        block_id = f"sym_{order}"
        has_nested = bool(symbol.children) or symbol.kind in _CONTAINER_KINDS

        if has_nested and symbol.children:
            header = _header_text(source_bytes, symbol)
            header_end = _header_end_byte(source_bytes, symbol)
            builder.add(
                block_id=block_id,
                block_type=BlockType.CODE_CONTAINER,
                text=header,
                parent_block_id=parent_id,
                order=order,
                hierarchy_path=[*hierarchy, block_id],
                location=BlockLocation(start_line=symbol.start_line, end_line=symbol.end_line),
                metadata={
                    "symbol_name": symbol.name,
                    "qualified_name": qualified,
                    "symbol_kind": symbol.kind,
                    "language": language,
                },
            )
            order += 1
            header_end = symbol.start_byte + len(header.encode("utf-8"))
            child_order = _emit_symbol_forest(
                builder,
                source_bytes=source_bytes,
                symbols=symbol.children,
                parent_id=block_id,
                hierarchy=[*hierarchy, block_id],
                language=language,
                qualified_prefix=qualified,
                order_start=order,
            )
            order = child_order
            interior_covered = {(symbol.start_byte, header_end)}
            interior_covered |= {(child.start_byte, child.end_byte) for child in symbol.children}
            _emit_gaps(
                builder,
                source_bytes=source_bytes,
                covered=interior_covered,
                parent_id=block_id,
                hierarchy=[*hierarchy, block_id],
                language=language,
                range_start=header_end,
                range_end=symbol.end_byte,
                order_base=order + 1000,
            )
        else:
            text = (
                source_bytes[symbol.start_byte : symbol.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )
            if text:
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.CODE_SYMBOL,
                    text=text,
                    parent_block_id=parent_id,
                    order=order,
                    hierarchy_path=[*hierarchy, block_id],
                    location=BlockLocation(
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                    ),
                    metadata={
                        "symbol_name": symbol.name,
                        "qualified_name": qualified,
                        "symbol_kind": symbol.kind,
                        "language": language,
                    },
                )
                order += 1
    return order


def _emit_gaps(
    builder: BlockBuilder,
    *,
    source_bytes: bytes,
    covered: set[tuple[int, int]],
    parent_id: str,
    hierarchy: list[str],
    language: str,
    range_start: int,
    range_end: int,
    order_base: int,
) -> None:
    if range_end <= range_start:
        return

    intervals = sorted(
        (start, end)
        for start, end in covered
        if end > start and end > range_start and start < range_end
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        start = max(start, range_start)
        end = min(end, range_end)
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    cursor = range_start
    gap_index = 0
    for start, end in merged:
        if cursor < start:
            _add_gap_block(
                builder,
                source_bytes=source_bytes,
                start=cursor,
                end=start,
                parent_id=parent_id,
                hierarchy=hierarchy,
                language=language,
                gap_index=gap_index,
                order=order_base + gap_index,
            )
            gap_index += 1
        cursor = max(cursor, end)
    if cursor < range_end:
        _add_gap_block(
            builder,
            source_bytes=source_bytes,
            start=cursor,
            end=range_end,
            parent_id=parent_id,
            hierarchy=hierarchy,
            language=language,
            gap_index=gap_index,
            order=order_base + gap_index,
        )


def _add_gap_block(
    builder: BlockBuilder,
    *,
    source_bytes: bytes,
    start: int,
    end: int,
    parent_id: str,
    hierarchy: list[str],
    language: str,
    gap_index: int,
    order: int,
) -> None:
    text = source_bytes[start:end].decode("utf-8", errors="replace").strip()
    if not text:
        return
    block_id = f"{parent_id}_gap_{gap_index}"
    start_line = source_bytes[:start].count(b"\n") + 1
    end_line = source_bytes[:end].count(b"\n") + 1
    builder.add(
        block_id=block_id,
        block_type=BlockType.CODE_BLOCK,
        text=text,
        parent_block_id=parent_id,
        order=order,
        hierarchy_path=[*hierarchy, block_id],
        location=BlockLocation(start_line=start_line, end_line=end_line),
        metadata={"symbol_kind": "gap", "language": language},
    )


def _header_text(source_bytes: bytes, symbol: _AstSymbol) -> str:
    segment = source_bytes[symbol.start_byte : symbol.end_byte].decode("utf-8", errors="replace")
    first_line = segment.splitlines()[0].strip() if segment else ""
    if first_line:
        return first_line
    return symbol.name


def _header_end_byte(source_bytes: bytes, symbol: _AstSymbol) -> int:
    newline = source_bytes.find(b"\n", symbol.start_byte, symbol.end_byte)
    if newline == -1:
        return min(symbol.start_byte + 1, symbol.end_byte)
    return newline + 1


def _load_tree_sitter(language: str) -> Any | None:
    """A tree-sitter ``Parser``, or None when the grammar is not installed."""
    try:
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    grammar = _grammar_for_language(language)
    if grammar is None:
        return None

    try:
        parser = Parser(Language(grammar))
    except Exception:
        return None
    return parser


def _grammar_for_language(language: str) -> object | None:
    try:
        if language == "python":
            import tree_sitter_python as tspython

            return tspython.language()
        if language == "javascript":
            import tree_sitter_javascript as tsjavascript

            return tsjavascript.language()
        if language == "typescript":
            import tree_sitter_typescript as tstypescript

            return tstypescript.language_typescript()
    except ImportError:
        return None
    return None


def _node_to_symbol(node: object, source_bytes: bytes) -> _AstSymbol | None:
    start_point = getattr(node, "start_point", None)
    end_point = getattr(node, "end_point", None)
    start_byte = getattr(node, "start_byte", None)
    end_byte = getattr(node, "end_byte", None)
    if start_point is None or end_point is None or start_byte is None or end_byte is None:
        return None

    text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace").strip()
    if not text:
        return None

    name = _symbol_name(node, source_bytes, text)
    kind = str(getattr(node, "type", "symbol"))
    return _AstSymbol(
        name=name,
        kind=kind,
        start_byte=int(start_byte),
        end_byte=int(end_byte),
        start_line=int(start_point[0]) + 1,
        end_line=int(end_point[0]) + 1,
    )


def _symbol_name(node: object, source_bytes: bytes, fallback_text: str) -> str:
    for child in getattr(node, "children", []):
        if getattr(child, "type", "") == "identifier":
            start = getattr(child, "start_byte", None)
            end = getattr(child, "end_byte", None)
            if start is not None and end is not None:
                return source_bytes[start:end].decode("utf-8", errors="replace")
    first_line = fallback_text.splitlines()[0] if fallback_text else "symbol"
    return first_line[:80]


def _add_line_windows(builder: BlockBuilder, source: str, *, window_size: int = 50) -> None:
    lines = source.splitlines()
    for index, start in enumerate(range(0, len(lines), window_size)):
        chunk = "\n".join(lines[start : start + window_size]).strip()
        if not chunk:
            continue
        start_line = start + 1
        end_line = min(start + window_size, len(lines))
        builder.add(
            block_id=f"win_{index}",
            block_type=BlockType.CODE_SYMBOL,
            text=chunk,
            parent_block_id="module",
            order=index,
            hierarchy_path=["doc", "module", f"win_{index}"],
            location=BlockLocation(start_line=start_line, end_line=end_line),
            metadata={"symbol_kind": "line_window"},
        )
