# CRAG eval report - `fixtures`

- provider: `mock`
- top_k: `5`
- docs: `4`
- queries: `10`

| mode | Hit@k | Recall@k | MRR | nDCG@k |
|------|------:|---------:|----:|-------:|
| vector | 0.700 | 0.700 | 0.362 | 0.445 |
| bm25 | 0.900 | 0.900 | 0.900 | 0.900 |
| hybrid | 1.000 | 1.000 | 0.933 | 0.950 |

## Comparison

- hybrid - bm25 Hit@k: `+0.100`
- hybrid - vector Hit@k: `+0.300`
- hybrid - bm25 nDCG@k: `+0.050`
- hybrid - vector nDCG@k: `+0.505`
