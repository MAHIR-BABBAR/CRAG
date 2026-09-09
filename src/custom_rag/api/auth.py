"""Optional API-key dependency for mutating/retrieval routes."""

from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, Request

from custom_rag.api.deps import get_settings
from custom_rag.core.config import CRAGSettings


def require_api_key(
    request: Request,
    cfg: CRAGSettings = Depends(get_settings),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """If ``api.api_key`` is configured, require Bearer or X-API-Key."""
    _ = request
    expected = cfg.api.api_key
    if not expected:
        return

    provided: str | None = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
