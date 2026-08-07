# Ranking diagnostics and holdout review

Two datasets predate the canonical `benchmark-v2` and remain in use for retrieval
diagnosis. Both share one principle: a candidate passage is never silently promoted
to gold, and an unreviewed candidate is never used as a negative. This prevents
scope-router rules from becoming evaluation labels.

For the canonical benchmark, metrics, and gates, see
[benchmark-and-metrics-zh.md](./benchmark-and-metrics-zh.md). Current numbers live in
[baseline-zh.md](./baseline-zh.md). Chinese explanations of the terminology used here
are in the [glossary](../glossary-zh.md).

## 1. Chinese ranking diagnostics

The diagnostic dataset evaluates whether retrieval ranks a structurally correct filing
passage ahead of plausible confounders. It does not require an answer string and does
not use an LLM to invent gold labels.

### Gold construction

Each document has an explicit profile containing its stable registry key, company
name, and reporting year. A query is eligible only when the target metric occurs in a
chunk whose section path matches the requested scope. The chunk ID, page range,
section path, and text excerpt are retained for audit.

Hybrid retrieval supplies hard-negative candidates. Rules can safely reject a
candidate when it belongs to another profiled company or its section establishes a
conflicting period or scope. Everything else remains `needs_review` and is not used as
a negative during scoring.

### Reproducibility

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

Each result includes `candidate_first_rank`, `candidate_recall`, and
`candidate_pool_size`, separating candidate-pool misses from downstream
scope/reranker ordering failures. `analyze-ranking-failures` computes full BM25,
Dense, and RRF gold ranks and classifies budget misses, fusion displacement, metadata
mismatch, and downstream regression. The dataset stores a content-derived ID and
source index ID.

### Limitations

- Two companies from one reporting year, so it is a diagnostic slice rather than a
  representative benchmark.
- One structurally preferred chunk is labeled per query; additional valid evidence
  chunks may still need review.
- Company names come from profiles and are persisted as metadata, but scope labels
  still rely on a small explainable rule table.
- Hit@K and MRR measure evidence ranking, not numerical answer correctness.

## 2. Independent holdout review workflow

The review pack is deliberately separate from the scored diagnostic dataset.
Generated questions are candidates only; their proposed evidence is a suggestion, not
frozen gold.

Generate a new pack after the active corpus is built:

```powershell
uv run findoc-rag generate-holdout-review `
  configs/ranking-diagnostic-profiles.json
```

Outputs:

- `data/diagnostics/holdout-review-v1.json` — machine-editable review records
- `reports/ranking/holdout-review-v1.md` — readable review sheet

For each item, review whether the question is unambiguous about company, year,
period, unit, and scope; whether the proposed passage directly answers it; and
whether an alternative passage should also be relevant. Mark ambiguous or unsupported
items `rejected`.

Set `status` to `approved`, `rejected`, or `edited`. For edited items fill
`reviewer_query`, `reviewer_gold_chunk_ids`, and `reviewer_notes`. The evaluator
consumes only approved/edited items and never treats pending items as gold.

The first pack (`f19011fa40507d1f1efe`) contains 16 pending candidates from the two
2024 annual reports and excludes the 10 queries used by existing routing diagnostics.
It is a review queue, not yet an evaluation benchmark. The normalized evaluation input
`data/diagnostics/holdout-eval-v2.json` is intentionally labelled
`independent_gold: false`; do not report it as an independently annotated benchmark.
