"""Child-chunk embedding pipeline."""

from custom_rag.ingestion.embedders.config import EmbedConfig
from custom_rag.ingestion.embedders.embedder import embed_document

__all__ = ["EmbedConfig", "embed_document"]
