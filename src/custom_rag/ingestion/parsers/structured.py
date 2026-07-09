"""Structured data parsers for JSON, YAML, and CSV."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import yaml

from custom_rag.core.exceptions import ParseFailedError
from custom_rag.core.types import BlockType, DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser, BlockBuilder


class StructuredParser(BaseParser):
    name = "structured"
    supported_extensions = frozenset({".json", ".yaml", ".yml", ".csv", ".tsv"})
    supported_mimes = frozenset(
        {
            "application/json",
            "application/yaml",
            "text/yaml",
            "text/csv",
            "text/tab-separated-values",
        }
    )

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        extension = path.suffix.lower()
        raw = self.read_bytes(path)
        text, encoding = self.decode_text(raw, path, metadata.encoding)

        builder = BlockBuilder()
        builder.add(
            block_id="doc_root",
            block_type=BlockType.DOCUMENT,
            text=text,
            hierarchy_path=["doc"],
        )

        if extension == ".json":
            doc_type = "json"
            self._parse_json(text, path, builder)
        elif extension in {".yaml", ".yml"}:
            doc_type = "yaml"
            self._parse_yaml(text, path, builder)
        else:
            doc_type = "csv"
            delimiter = "\t" if extension == ".tsv" else ","
            self._parse_csv(text, path, builder, delimiter=delimiter)

        doc_metadata = metadata.model_copy(update={"doc_type": doc_type, "encoding": encoding})
        return ParsedDocument(metadata=doc_metadata, blocks=builder.blocks, raw_text=text)

    @staticmethod
    def _parse_json(text: str, path: Path, builder: BlockBuilder) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseFailedError(str(path), "invalid JSON", cause=exc) from exc

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                block_id = f"record_{index}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.RECORD,
                    text=json.dumps(item, ensure_ascii=False),
                    parent_block_id="doc_root",
                    order=index,
                    hierarchy_path=["doc", block_id],
                    metadata={"record_index": index},
                )
            return

        if isinstance(payload, dict):
            for index, (key, value) in enumerate(payload.items()):
                block_id = f"record_{index}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.RECORD,
                    text=json.dumps({key: value}, ensure_ascii=False),
                    parent_block_id="doc_root",
                    order=index,
                    hierarchy_path=["doc", block_id],
                    metadata={"record_key": key, "record_index": index},
                )
            return

        builder.add(
            block_id="record_0",
            block_type=BlockType.RECORD,
            text=json.dumps(payload, ensure_ascii=False),
            parent_block_id="doc_root",
            order=0,
            hierarchy_path=["doc", "record_0"],
        )

    @staticmethod
    def _parse_yaml(text: str, path: Path, builder: BlockBuilder) -> None:
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ParseFailedError(str(path), "invalid YAML", cause=exc) from exc

        if payload is None:
            return

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                block_id = f"record_{index}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.RECORD,
                    text=yaml.safe_dump(item, sort_keys=False).strip(),
                    parent_block_id="doc_root",
                    order=index,
                    hierarchy_path=["doc", block_id],
                    metadata={"record_index": index},
                )
            return

        if isinstance(payload, dict):
            for index, (key, value) in enumerate(payload.items()):
                block_id = f"record_{index}"
                builder.add(
                    block_id=block_id,
                    block_type=BlockType.RECORD,
                    text=yaml.safe_dump({key: value}, sort_keys=False).strip(),
                    parent_block_id="doc_root",
                    order=index,
                    hierarchy_path=["doc", block_id],
                    metadata={"record_key": key, "record_index": index},
                )

    @staticmethod
    def _parse_csv(text: str, path: Path, builder: BlockBuilder, *, delimiter: str) -> None:
        reader = csv.DictReader(StringIO(text), delimiter=delimiter)
        if reader.fieldnames is None:
            raise ParseFailedError(str(path), "CSV file has no header row")

        for index, row in enumerate(reader):
            block_id = f"row_{index}"
            rendered = ", ".join(f"{key}={value}" for key, value in row.items())
            builder.add(
                block_id=block_id,
                block_type=BlockType.TABLE_ROW,
                text=rendered,
                parent_block_id="doc_root",
                order=index,
                hierarchy_path=["doc", block_id],
                metadata={"row_index": index, "columns": list(reader.fieldnames)},
            )
