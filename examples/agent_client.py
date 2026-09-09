#!/usr/bin/env python3
"""Minimal agent-side client: health, index, then fetch packed context.

The provider must match the one the collection was indexed with, because a
query vector is only comparable to vectors from the same embedding model.

    # against `crag serve` with the mock provider (no model download)
    python examples/agent_client.py --provider mock

    # against `docker compose up`, which indexes with all-MiniLM-L6-v2
    python examples/agent_client.py --provider local --api-key crag-demo-key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo_corpus"


def _print_section(title: str, payload: Any, limit: int = 800) -> None:
    rendered = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    print(f"\n--- {title} ---")
    print(rendered[:limit] or "(empty)")


def main() -> int:
    parser = argparse.ArgumentParser(description="CRAG agent client demo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=os.getenv("CRAG_API_KEY"),
        help="Defaults to $CRAG_API_KEY. Required when the server sets one.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "local", "openai"],
        help="Must match the provider the collection was indexed with.",
    )
    parser.add_argument("--collection", default="demo")
    parser.add_argument(
        "--query",
        default="How does hybrid retrieval combine BM25 and dense search?",
    )
    parser.add_argument("--corpus", type=Path, default=DEMO)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Query an already-populated collection (the Docker demo self-indexes).",
    )
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120.0) as client:
        health = client.get("/v1/health")
        health.raise_for_status()
        _print_section("health", health.json())

        if not args.skip_index:
            index = client.post(
                "/v1/index",
                json={
                    "path": str(args.corpus.resolve()),
                    "provider": args.provider,
                    "recursive": True,
                    "collection": args.collection,
                },
            )
            if index.status_code == 403:
                print(
                    "\nThe server refused that path. It restricts indexing to "
                    "api.allowed_index_roots; pass --skip-index when talking to "
                    "the Docker demo, which indexes its own corpus at startup.",
                    file=sys.stderr,
                )
                return 1
            index.raise_for_status()
            _print_section("index", index.json())

        context = client.post(
            "/v1/context",
            json={
                "query": args.query,
                "mode": "hybrid",
                "provider": args.provider,
                "collection": args.collection,
                "top_k": 5,
            },
        )
        if context.status_code == 409:
            print(f"\n{context.json()['detail']}", file=sys.stderr)
            return 1
        context.raise_for_status()
        payload = context.json()

        print(f"\nretrieval_quality: {payload.get('retrieval_quality')}")
        _print_section("context", payload.get("context", ""), limit=1200)
        _print_section("citations", payload.get("citations", []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
