# Architecture decision records

Short notes on the choices that shaped CRAG, including what each one costs.

| # | Decision |
|---|---|
| [0001](0001-sqlite-as-the-index.md) | One SQLite file holds documents, vectors, and the keyword index |
| [0002](0002-parent-child-chunking.md) | Search small chunks, return the sections they came from |
| [0003](0003-exact-dense-search.md) | Dense search is exact, and the setting is named for that |
| [0004](0004-rrf-for-fusion.md) | Combine channels with Reciprocal Rank Fusion |
| [0005](0005-retrieval-only.md) | The library stops at context; agents generate |
