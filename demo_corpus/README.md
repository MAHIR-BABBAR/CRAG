# Demo corpus

Five short documents written for the demo so that `docker compose up` has
something to retrieve over immediately. They describe CRAG itself, which makes
the sample queries in `examples/agent_client.py` answerable and easy to check by
eye.

The mix is deliberate: Markdown with headings, a plain-text file, and a Python
module, so a single `crag index ./demo_corpus` exercises three different parsers
and both chunk roles.

This corpus is a demo fixture. It is not an evaluation set, and no number in
`docs/benchmarks.md` comes from it — those come from BEIR SciFact.
