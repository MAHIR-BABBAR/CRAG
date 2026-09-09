"""OpenAI provider configuration checks that do not require the network."""

from __future__ import annotations

from typing import Any

import pytest

from custom_rag.core.exceptions import EmbeddingError
from custom_rag.core.providers.openai import OpenAIEmbeddingProvider


class _FakeEmbeddings:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        count = len(kwargs["input"])

        class _Item:
            def __init__(self, dims: int) -> None:
                self.embedding = [0.1] * dims

        class _Response:
            def __init__(self, dims: int) -> None:
                self.data = [_Item(dims) for _ in range(count)]

        return _Response(self.dimensions)


class _FakeClient:
    def __init__(self, dimensions: int) -> None:
        self.embeddings = _FakeEmbeddings(dimensions)


def test_api_key_is_required() -> None:
    with pytest.raises(EmbeddingError):
        OpenAIEmbeddingProvider(api_key="")


def test_known_models_get_their_native_dimensions() -> None:
    provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-large")
    assert provider.dimensions == 3072


def test_ada_cannot_be_truncated() -> None:
    """ada-002 ignores the `dimensions` parameter, so asking for it is a config error."""
    with pytest.raises(EmbeddingError, match="cannot be truncated"):
        OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-ada-002", dimensions=256)


def test_requested_dimensions_are_sent_to_the_api() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test", model="text-embedding-3-small", dimensions=256
    )
    client = _FakeClient(256)
    provider._client = client
    vectors = provider.embed_texts(["hello"])
    assert client.embeddings.last_kwargs["dimensions"] == 256
    assert len(vectors[0]) == 256


def test_unrequested_dimensions_are_not_sent() -> None:
    provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")
    client = _FakeClient(1536)
    provider._client = client
    provider.embed_texts(["hello"])
    assert "dimensions" not in client.embeddings.last_kwargs


def test_unexpected_vector_width_is_rejected() -> None:
    """A silent width change would corrupt an index rather than fail."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")
    provider._client = _FakeClient(512)
    with pytest.raises(EmbeddingError, match="512"):
        provider.embed_texts(["hello"])


def test_empty_input_short_circuits() -> None:
    provider = OpenAIEmbeddingProvider(api_key="sk-test")
    assert provider.embed_texts([]) == []
