"""PDF parser using Unstructured API (vlm / hi_res strategies)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from custom_rag.core.exceptions import ConfigError, ParseFailedError
from custom_rag.core.types import DocumentMetadata, ParsedDocument
from custom_rag.ingestion.parsers.base import BaseParser
from custom_rag.ingestion.parsers.pdf.mapper import map_elements_to_blocks

PdfStrategy = Literal["vlm", "hi_res", "auto"]


class UnstructuredPDFParser(BaseParser):
    name = "pdf_unstructured"
    supported_extensions = frozenset({".pdf"})
    supported_mimes = frozenset({"application/pdf"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        strategy: PdfStrategy = "hi_res",
        vlm_model: str | None = None,
        vlm_model_provider: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._strategy = strategy
        self._vlm_model = vlm_model
        self._vlm_model_provider = vlm_model_provider

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        api_key = self._api_key or os.getenv("UNSTRUCTURED_API_KEY")
        if not api_key:
            raise ConfigError(
                "UNSTRUCTURED_API_KEY is required for PDF parsing. "
                "Set it in the environment or pass api_key to UnstructuredPDFParser."
            )

        elements = self._partition(path, api_key)
        blocks = map_elements_to_blocks(elements)
        raw_text = "\n\n".join(block.text for block in blocks if block.text)

        doc_metadata = metadata.model_copy(
            update={
                "doc_type": "pdf",
                "extra": {
                    **metadata.extra,
                    "parser": self.name,
                    "strategy": self._strategy,
                    "element_count": len(elements),
                    "block_count": len(blocks),
                },
            }
        )
        return ParsedDocument(metadata=doc_metadata, blocks=blocks, raw_text=raw_text or None)

    def _partition(self, path: Path, api_key: str) -> list[dict[str, Any]]:
        try:
            import unstructured_client
            from unstructured_client.models import operations, shared
        except ImportError as exc:
            raise ParseFailedError(
                str(path),
                "unstructured-client is not installed; use pip install 'custom-rag[ingestion]'",
                cause=exc,
            ) from exc

        strategy_map = {
            "vlm": shared.Strategy.VLM,
            "hi_res": shared.Strategy.HI_RES,
            "auto": shared.Strategy.AUTO,
        }
        params: dict[str, Any] = {
            "files": shared.Files(content=self.read_bytes(path), file_name=path.name),
            "strategy": strategy_map[self._strategy],
            "coordinates": True,
            "pdf_infer_table_structure": True,
        }
        if self._strategy == "vlm":
            params["vlm_model"] = self._vlm_model or os.getenv("UNSTRUCTURED_VLM_MODEL", "gpt-4o")
            params["vlm_model_provider"] = self._vlm_model_provider or os.getenv(
                "UNSTRUCTURED_VLM_PROVIDER", "openai"
            )

        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(**params)
        )

        try:
            with unstructured_client.UnstructuredClient(api_key_auth=api_key) as client:
                response = client.general.partition(request=request)
        except Exception as exc:
            raise ParseFailedError(
                str(path), "Unstructured API partition failed", cause=exc
            ) from exc

        if response.elements is None:
            raise ParseFailedError(str(path), "Unstructured API returned no elements")

        return [_element_to_dict(element) for element in response.elements]


def _element_to_dict(element: object) -> dict[str, Any]:
    if isinstance(element, dict):
        return element
    model_dump = getattr(element, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return dict(element)  # type: ignore[arg-type]
