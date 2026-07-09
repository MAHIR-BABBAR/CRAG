"""Source code parser with tree-sitter AST boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class _SymbolSpan:
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str


class CodeParser(BaseParser):
    name = "code"
    supported_extensions = frozenset(_EXTENSION_LANGUAGE.keys())
    supported_mimes = frozenset({"text/x-python", "application/javascript", "text/javascript"})

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        raw = self.read_bytes(path)
        source, encoding = self.decode_text(raw, path, metadata.encoding)
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        language = _EXTENSION_LANGUAGE.get(path.suffix.lower(), "unknown")

        builder = BlockBuilder()
        builder.add(
            block_id="module",
            block_type=BlockType.MODULE,
            text=source,
            hierarchy_path=["doc", "module"],
            metadata={"language": language},
        )

        symbols = _extract_symbols(source, language)
        if symbols:
            for index, symbol in enumerate(symbols):
                builder.add(
                    block_id=f"sym_{index}",
                    block_type=BlockType.CODE_SYMBOL,
                    text=symbol.text,
                    parent_block_id="module",
                    order=index,
                    hierarchy_path=["doc", "module", symbol.name],
                    location=BlockLocation(
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                    ),
                    metadata={
                        "symbol_name": symbol.name,
                        "symbol_kind": symbol.kind,
                        "language": language,
                    },
                )
        else:
            _add_line_windows(builder, source)

        doc_metadata = metadata.model_copy(
            update={"doc_type": "code", "language": language, "encoding": encoding}
        )
        return ParsedDocument(metadata=doc_metadata, blocks=builder.blocks, raw_text=source)


def _extract_symbols(source: str, language: str) -> list[_SymbolSpan]:
    parser = _load_tree_sitter(language)
    if parser is None:
        return []

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return []

    node_types = _SYMBOL_NODE_TYPES.get(language, set())
    symbols: list[_SymbolSpan] = []

    def walk(node: object) -> None:
        node_type = getattr(node, "type", "")
        if node_type in node_types:
            span = _node_span(node, source)
            if span is not None:
                symbols.append(span)
        for child in getattr(node, "children", []):
            walk(child)

    walk(tree.root_node)
    return symbols


def _load_tree_sitter(language: str) -> object | None:
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


def _node_span(node: object, source: str) -> _SymbolSpan | None:
    start_point = getattr(node, "start_point", None)
    end_point = getattr(node, "end_point", None)
    start_byte = getattr(node, "start_byte", None)
    end_byte = getattr(node, "end_byte", None)
    if start_point is None or end_point is None or start_byte is None or end_byte is None:
        return None

    start_line = int(start_point[0]) + 1
    end_line = int(end_point[0]) + 1
    source_bytes = source.encode("utf-8")
    text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace").strip()
    if not text:
        return None

    name = _symbol_name(node, source_bytes, text)
    kind = str(getattr(node, "type", "symbol"))
    return _SymbolSpan(name=name, kind=kind, start_line=start_line, end_line=end_line, text=text)


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
