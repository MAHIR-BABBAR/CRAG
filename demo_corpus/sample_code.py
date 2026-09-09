"""A small client an agent might use to pull context out of CRAG.

Included in the demo corpus so the code-aware chunker has real Python to split:
each function and class becomes its own child chunk under a module-level parent.
"""

from __future__ import annotations

import httpx


class ContextClient:
    """Fetches packed context from a running CRAG socket."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=30.0)

    def fetch(self, query: str, *, collection: str = "demo", top_k: int = 5) -> str:
        """Return packed context text, or an empty string when nothing matched."""
        response = self._client.post(
            "/v1/context",
            json={
                "query": query,
                "mode": "hybrid",
                "collection": collection,
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload["retrieval_quality"] == "empty":
            return ""
        return str(payload["context"])

    def close(self) -> None:
        self._client.close()


def build_prompt(question: str, context: str) -> str:
    """Compose a grounded prompt. Generation stays on the agent's side."""
    if not context:
        return f"Answer from your own knowledge and say so explicitly:\n{question}"
    return (
        "Answer using only the context below. Cite the bracketed numbers.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
