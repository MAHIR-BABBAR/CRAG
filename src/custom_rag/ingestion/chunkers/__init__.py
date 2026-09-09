"""Parent-child document chunking."""

from custom_rag.ingestion.chunkers.chunker import chunk_document
from custom_rag.ingestion.chunkers.config import ChunkConfig

__all__ = ["ChunkConfig", "chunk_document"]
