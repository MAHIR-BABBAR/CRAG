"""Parser protocol, shared helpers, and block construction utilities."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from charset_normalizer import from_bytes

from custom_rag.core.exceptions import ParseFailedError
from custom_rag.core.types import (
    BlockLocation,
    BlockType,
    ContentBlock,
    DocumentMetadata,
    ParsedDocument,
)


@runtime_checkable
class Parser(Protocol):
    name: str
    supported_extensions: frozenset[str]
    supported_mimes: frozenset[str]

    def can_parse(self, path: Path, mime_type: str | None = None) -> bool: ...

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument: ...


class BaseParser(ABC):
    name: ClassVar[str]
    supported_extensions: ClassVar[frozenset[str]] = frozenset()
    supported_mimes: ClassVar[frozenset[str]] = frozenset()

    def can_parse(self, path: Path, mime_type: str | None = None) -> bool:
        extension = path.suffix.lower()
        if extension in self.supported_extensions:
            return True
        if mime_type and mime_type in self.supported_mimes:
            return True
        return False

    @abstractmethod
    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument: ...

    @staticmethod
    def read_bytes(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ParseFailedError(str(path), "unable to read file", cause=exc) from exc

    @staticmethod
    def decode_text(raw: bytes, path: Path, encoding: str | None = None) -> tuple[str, str]:
        if encoding:
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError as exc:
                raise ParseFailedError(
                    str(path), f"invalid {encoding} encoding", cause=exc
                ) from exc

        detected = from_bytes(raw).best()
        if detected is None:
            raise ParseFailedError(str(path), "unable to detect text encoding")

        return str(detected), str(detected.encoding)


class BlockBuilder:
    """Construct validated ContentBlock sequences with stable hierarchy paths."""

    _SLUG_RE = re.compile(r"[^a-z0-9]+")

    def __init__(self) -> None:
        self._blocks: list[ContentBlock] = []

    @property
    def blocks(self) -> list[ContentBlock]:
        return list(self._blocks)

    def add(
        self,
        *,
        block_id: str,
        block_type: BlockType,
        text: str,
        parent_block_id: str | None = None,
        order: int = 0,
        hierarchy_path: list[str] | None = None,
        location: BlockLocation | None = None,
        metadata: dict | None = None,
    ) -> ContentBlock:
        path = hierarchy_path or ([block_id] if parent_block_id is None else [block_id])
        block = ContentBlock(
            block_id=block_id,
            block_type=block_type,
            text=text,
            parent_block_id=parent_block_id,
            order=order,
            hierarchy_path=path,
            location=location,
            metadata=metadata or {},
        )
        self._blocks.append(block)
        return block

    @classmethod
    def slugify(cls, value: str) -> str:
        slug = cls._SLUG_RE.sub("_", value.lower()).strip("_")
        return slug or "block"


def content_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
