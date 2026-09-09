"""OpenAI embedding provider."""

from __future__ import annotations

from custom_rag.core.exceptions import EmbeddingError

_DEFAULT_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Only the v3 models support Matryoshka truncation via the `dimensions` request
# parameter. Sending it to ada-002 is an API error.
_TRUNCATABLE_MODELS: frozenset[str] = frozenset(
    {"text-embedding-3-small", "text-embedding-3-large"}
)


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAI embeddings")
        native = _DEFAULT_DIMENSIONS.get(model, 1536)
        if dimensions is not None and model not in _TRUNCATABLE_MODELS:
            raise EmbeddingError(
                f"{model!r} always returns {native} dimensions and cannot be truncated; "
                "drop embedding.dimensions or switch to text-embedding-3-small/large"
            )
        self.model = model
        self._api_key = api_key
        self._requested_dimensions = dimensions
        self.dimensions = dimensions or native
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "openai package is required; install with pip install custom-rag[embeddings]",
                cause=exc,
            ) from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        kwargs: dict[str, object] = {"input": texts, "model": self.model}
        if self._requested_dimensions is not None:
            kwargs["dimensions"] = self._requested_dimensions
        try:
            response = client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding request failed: {exc}", cause=exc) from exc
        vectors = [list(item.embedding) for item in response.data]
        if vectors and len(vectors[0]) != self.dimensions:
            raise EmbeddingError(
                f"OpenAI returned {len(vectors[0])}-dimensional vectors but "
                f"{self.dimensions} were configured"
            )
        return vectors
