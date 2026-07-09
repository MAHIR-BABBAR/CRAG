"""Parser registration and format routing."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.exceptions import UnsupportedFormatError
from custom_rag.ingestion.parsers.base import Parser


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: list[Parser] = []

    def register(self, parser: Parser, *, prepend: bool = False) -> None:
        if prepend:
            self._parsers.insert(0, parser)
        else:
            self._parsers.append(parser)

    def get_parser(self, path: Path, mime_type: str | None = None) -> Parser:
        resolved = path.expanduser().resolve()
        for parser in self._parsers:
            if parser.can_parse(resolved, mime_type):
                return parser
        raise UnsupportedFormatError(str(resolved), resolved.suffix.lower() or None)

    @property
    def parsers(self) -> tuple[Parser, ...]:
        return tuple(self._parsers)
