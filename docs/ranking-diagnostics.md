
# Chinese ranking diagnostics

The diagnostic dataset evaluates whether retrieval ranks a structurally correct
filing passage ahead of plausible confounders. It does not require an answer string
and does not use an LLM to invent gold labels.

## Gold construction

Each document has an explicit profile containing its stable registry key, company
name, and reporting year. A query is eligible only when the target metric occurs in
a chunk whose section path matches the requested scope. The chunk ID, page range,
section path, and text excerpt are retained for audit.

Hybrid retrieval supplies hard-negative candidates. Rules can safely reject a
candidate when it belongs to another profiled company or its section establishes a
conflicting period or scope. Everything else remains `needs_review`; it is not used
as a negative during scoring.

## Reproducibility

```powershell
uv run findoc-rag generate-ranking-diagnostics `
  configs/ranking-diagnostic-profiles.json --candidate-k 20

$env:HF_HUB_OFFLINE = "1"
uv run findoc-rag evaluate-ranking-diagnostics `
  data/diagnostics/ranking-diagnostics-v1.json `
  --mode hybrid --top-k 5 --candidate-k 20 `
  --metadata-filters --scope-routing `
  --output reports/ranking/hybrid-v1.json
```

Each result includes `candidate_first_rank` and `candidate_recall`, separating
candidate-pool misses from downstream scope/reranker ordering failures. It also stores
`candidate_pool_size`. `analyze-ranking-failures` computes full BM25, Dense, and RRF
gold ranks and classifies budget misses, fusion displacement, metadata mismatch, and
downstream regression. The dataset
stores a content-derived ID and source index ID. Unreviewed candidates never become
implicit negatives.

## Current limitations

- The first version contains two companies from one reporting year, so it is a
  diagnostic slice rather than a representative benchmark.
- One structurally preferred chunk is labeled per query; additional valid evidence
  chunks may still need review.
- Company names are supplied by profiles and persisted as metadata, but scope labels
  still rely on a small explainable rule table.
- Hit@K and MRR measure evidence ranking, not numerical answer correctness.
