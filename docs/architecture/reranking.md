# Cross-encoder reranking

FinDocRAG treats reranking as an optional stage after lexical, dense, or hybrid
candidate retrieval. It is deliberately independent of retrieval mode so experiments
can compare the same candidate source with and without reranking.

```text
lexical / dense / RRF candidates -> CrossEncoder -> top-k evidence
```

The default model is `BAAI/bge-reranker-v2-m3`, a multilingual reranker suitable for
Chinese and mixed-language documents. The model is loaded lazily on the first rerank
request and reused for the lifetime of the service process.

```toml
[reranker]
enabled = true
model = "BAAI/bge-reranker-v2-m3"
batch_size = 16
```

When enabled, requests rerank by default. A caller can explicitly set `"rerank":
false` for an online control query. If the service has no configured reranker,
requesting `"rerank": true` returns a structured client error rather than silently
falling back.

The retrieval `candidate_k` controls how many candidates reach the reranker; `top_k`
controls how many reranked hits are returned. Each returned hit retains earlier score
metadata and adds `original_rank`, `rank_delta`, and `rerank_score`. Trace records include a separate
`rerank` stage with latency and ranking snapshots. Raw scores are model-specific and
should not be treated as calibrated probabilities.

```powershell
uv run findoc-rag search-index data/indexes/corpus `
  "贵州茅台2024年营业收入是多少" --mode hybrid --candidate-k 30 `
  --top-k 5 --rerank --output reports/reranking/example.json
```
