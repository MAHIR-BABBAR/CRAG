# Running CRAG

Everything here assumes a single-node deployment: one process, one SQLite file.
That is the shape the project is built for ([ADR 0001](adr/0001-sqlite-as-the-index.md)).

## Configuration

Settings resolve in this order, last one wins: defaults in code,
`configs/default.yaml`, environment variables, explicit CLI flags. Every setting
has an environment alias prefixed with `CRAG_`; see [`.env.example`](../.env.example).

| Variable | Default | Notes |
|---|---|---|
| `CRAG_API_KEY` | unset | When set, index/retrieve/context require it. Health stays open. |
| `CRAG_ALLOWED_INDEX_ROOTS` | unset | Comma-separated roots `/v1/index` may read from. |
| `CRAG_DENSE_SEARCH` | `python` | `numpy` is the faster exact backend. |
| `CRAG_EMBEDDING_PROVIDER` | `mock` | `local`, `openai`, or `mock`. |
| `CRAG_STORAGE_PATH` | `workspace_index/index.sqlite3` | The whole index. |
| `CRAG_MAX_INDEX_FILES` | 500 | Directory indexing cap. |

`mock` is a deterministic hash-based embedder. It exists so tests and demos run
without a model or an API key; it does not retrieve meaningfully.

## Serving

```bash
crag serve --host 127.0.0.1 --port 8000
```

The server refuses to be interesting on a public interface without auth: binding
to anything other than loopback with no `CRAG_API_KEY` set logs a warning, and
you should treat that warning as a blocker. There is no TLS, no rate limiting,
and no per-user authorization in this project. Put it behind a reverse proxy if
it needs to face anything.

### Docker

```bash
docker compose up --build
```

The image bakes `all-MiniLM-L6-v2` in at build time and sets `HF_HUB_OFFLINE=1`,
so a container start does not depend on Hugging Face being reachable. It runs as
the non-root user `crag`, binds to `127.0.0.1:8000`, requires the API key
`crag-demo-key`, and restricts indexing to `/app/demo_corpus`. Change the key
before running it anywhere shared.

The health check hits `/v1/health`, which reports status, version, and whether
the index is reachable.

## Indexing

```bash
crag index ./docs --provider local
```

Directory indexing walks recursively, takes only extensions a registered parser
claims, sorts paths for determinism, and stops at `max_index_files`, setting
`truncated: true` in the response rather than silently dropping the rest.

Re-indexing is cheap. A document whose content hash is unchanged is skipped
before parsing, and an unchanged chunk hits the embedding cache rather than the
model. Re-running an index over a mostly-unchanged folder costs seconds.

## Operational limits worth knowing

**Dense search is a full scan.** Query latency grows linearly with the number of
stored vectors. At 5k documents (about 25k chunks) the numpy backend answers in
tens of milliseconds; at a million chunks it will not. That is the tradeoff
recorded in [ADR 0003](adr/0003-exact-dense-search.md), and the fix is an ANN
index, which this project does not have.

**Writes are serialized.** SQLite is in WAL mode, so readers do not block, but
concurrent indexing from several processes will contend. Index from one process.

**Changing the embedding model invalidates the index.** Vectors from a different
model are not comparable, so a query embedded with a model that does not match
the collection is refused with a config error instead of returning plausible
nonsense. Re-index after a model change.

**The model load dominates cold start.** The API loads the sentence-transformers
model once and caches it per provider configuration. The first request after
start-up pays a few seconds; the rest do not.

## When retrieval looks wrong

Start with the quality label on the context response. `empty` means nothing was
retrieved, `low` means the top scores are weak enough that the agent should
consider saying it does not know, and `high` means both channels agreed.

Then narrow down by channel: run the same query with `--mode bm25` and
`--mode vector` separately. If BM25 finds it and dense does not, the query
phrasing is lexically close but semantically distant from the chunk, or the wrong
embedding model is loaded. If dense finds it and BM25 does not, the terms in the
query do not appear in the text, which is the case hybrid exists to cover. If
neither finds it, check that the document was actually indexed and chunked, since
an unclaimed file extension is skipped silently by directory indexing.

## Backup

The index is one file plus its WAL sidecars. Stop the writer and copy
`workspace_index/`, or use `sqlite3 index.sqlite3 ".backup out.sqlite3"` for a
consistent copy while running. There is no other state.
    