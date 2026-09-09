FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY demo_corpus ./demo_corpus

RUN pip install -e ".[ingestion,embeddings,storage,numpy]"

ENV HF_HOME=/cache/huggingface

# Bake the embedding model into the image so the demo starts without network
# access and the first query is not paying for a model download.
RUN mkdir -p /cache/huggingface \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV CRAG_STORAGE_PATH=/data/crag.sqlite3 \
    CRAG_COLLECTION=demo \
    EMBEDDING_PROVIDER=local \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    CRAG_RETRIEVAL_MODE=hybrid \
    CRAG_API_HOST=0.0.0.0 \
    CRAG_API_PORT=8000 \
    CRAG_ALLOWED_INDEX_ROOTS=/app/demo_corpus \
    CRAG_DEMO_CORPUS=/app/demo_corpus \
    HF_HUB_OFFLINE=1

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 10001 crag \
    && mkdir -p /data \
    && chown -R crag:crag /data /cache /app

USER crag

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=6 \
    CMD curl -fsS "http://127.0.0.1:${CRAG_API_PORT:-8000}/v1/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
