# Examples

## Agent client

`agent_client.py` is what a calling agent does: check health, index a folder,
ask for packed context, and read the citations back. It needs a running socket.

```bash
# Terminal A
crag serve --host 127.0.0.1 --port 8000

# Terminal B
python examples/agent_client.py
```

You should see health JSON, an index summary for `demo_corpus/`, then a packed
context block with a `retrieval_quality` label and a list of citations.

With authentication enabled:

```powershell
$env:CRAG_API_KEY="demokey"
crag serve
```

```bash
python examples/agent_client.py --api-key demokey
```

Useful flags: `--provider local` to use real embeddings instead of the mock
hash embedder, and `--skip-index` to query an index that already exists.

## Docker

The compose stack requires an API key and pre-indexes `demo_corpus/` at start-up.

```bash
docker compose up --build
python examples/agent_client.py --api-key crag-demo-key --skip-index
```

Or against the raw HTTP surface:

```bash
curl http://127.0.0.1:8000/v1/health

curl -X POST http://127.0.0.1:8000/v1/context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: crag-demo-key" \
  -d '{"query":"What is hybrid retrieval?","mode":"hybrid","provider":"local","collection":"demo"}'
```
