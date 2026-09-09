#!/bin/sh
# Index the demo corpus, then serve. Indexing is content-hash deduplicated, so
# restarting the container re-uses the existing index instead of rebuilding it.
set -eu

CORPUS="${CRAG_DEMO_CORPUS:-/app/demo_corpus}"
COLLECTION="${CRAG_COLLECTION:-demo}"

if [ -d "$CORPUS" ]; then
    echo "Indexing ${CORPUS} into collection '${COLLECTION}'..."
    crag index "$CORPUS" --provider local --collection "$COLLECTION"
else
    echo "No demo corpus at ${CORPUS}; starting with whatever is already indexed."
fi

echo "Serving on ${CRAG_API_HOST:-0.0.0.0}:${CRAG_API_PORT:-8000}"
exec crag serve --host "${CRAG_API_HOST:-0.0.0.0}" --port "${CRAG_API_PORT:-8000}"
