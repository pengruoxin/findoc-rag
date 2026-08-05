# Reranker validation attempt — 2026-08-05

## Automated validation

The offline test suite uses a deterministic fake CrossEncoder so CI does not depend
on network access or mutable remote model artifacts. It verifies candidate expansion,
stable tie-breaking, final scores, original ranks, rank deltas, stage tracing, model
identity, and explicit failure when reranking is requested but not configured.

## Real-model attempt

Command:

```powershell
uv run findoc-rag search-index data/indexes/corpus `
  "贵州茅台2024年营业收入是多少" --mode hybrid --candidate-k 30 `
  --top-k 5 --rerank --output reports/reranking/moutai_revenue_2024.json
```

Model: `BAAI/bge-reranker-v2-m3`

Corpus: 2 real Chinese annual reports, 958 chunks.

Result: the process reached the 604-second command timeout during first-time model
acquisition. The Hugging Face cache entry remained at 0 MB, so no ranking output was
produced. This is recorded as an external model-download failure, not a retrieval or
reranking quality result. No accuracy or latency claim is made from this attempt.

The command is safe to rerun. Once model acquisition succeeds, its complete output
will be stored at the path above and evaluated against versioned query expectations.
