"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from custom_rag.core.exceptions import EmbeddingError

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


class LocalEmbeddingProvider:
    name = "local"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions or _DEFAULT_DIMENSIONS.get(model, 384)
        self._model_instance: object | None = None

    def _get_model(self) -> object:
        if self._model_instance is not None:
            return self._model_instance
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is required; "
                'install with: pip install "custom-rag[embeddings]"',
                cause=exc,
            ) from exc
        try:
            self._model_instance = SentenceTransformer(self.model)
        except Exception as exc:
            raise EmbeddingError(
                f"failed to load sentence-transformers model {self.model!r}: {exc}",
                cause=exc,
            ) from exc
        return self._model_instance

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(texts, convert_to_numpy=True)  # type: ignore[attr-defined]
        except Exception as exc:
            raise EmbeddingError(f"local embedding encode failed: {exc}", cause=exc) from exc
        return [list(map(float, vector)) for vector in vectors]
