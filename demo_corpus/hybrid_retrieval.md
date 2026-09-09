# Hybrid Retrieval Notes

BM25 ranks documents by term frequency with length normalization.
Dense retrieval embeds queries and chunks into a shared vector space and uses cosine similarity.
CRAG fuses the two ranked lists with Reciprocal Rank Fusion (RRF) using k=60 by default.

Optional cross-encoder rerank can rescore the fused candidates, but on scientific
benchmarks a general MS MARCO cross-encoder may not beat strong hybrid RRF.
