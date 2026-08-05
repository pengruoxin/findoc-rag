# Chinese ranking diagnostics

The diagnostic dataset evaluates whether retrieval ranks a structurally correct
filing passage ahead of plausible confounders. It does not require an answer string
and does not use an LLM to invent gold labels.

## Gold construction

Each document has an explicit profile containing its stable registry key, company
name, and reporting year. A query is eligible only when the target metric occurs in
a chunk whose section path matches the requested scope, such as annual indicators,
quarterly data, or segment revenue. The chunk ID, page range, section path, and text
excerpt are retained for audit.

Hybrid retrieval then supplies hard-negative candidates. Rules can safely reject a
candidate when it belongs to another profiled company or its section establishes a
conflicting period or scope. Everything else remains `needs_review`; it is not used
as a negative during scoring.

`accepted` means the query has an auditable structural positive and at least one safe
hard negative. It does not mean every retrieved candidate has been judged.

## Reproducibility

```powershell
uv run findoc-rag generate-ranking-diagnostics `
  configs/ranking-diagnostic-profiles.json --candidate-k 20

uv run findoc-rag evaluate-ranking-diagnostics `
  data/diagnostics/ranking-diagnostics-v1.json `
  --mode hybrid --top-k 5 --candidate-k 20 `
  --output reports/ranking/hybrid-v1.json
```

The dataset stores its content-derived ID and source index ID. Evaluation ignores
queries not marked `accepted` and treats only explicit `relevant` judgments as gold.
Unreviewed candidates never become implicit negatives.

## Current limitations

- The first version contains two companies from one reporting year, so it is a
  diagnostic slice rather than a representative benchmark.
- One structurally preferred chunk is labeled per query; additional valid evidence
  chunks may still need review.
- Company names are supplied by profiles but are not yet searchable chunk metadata.
  This makes company-constrained queries an intentional stress test and motivates
  metadata-aware retrieval/filtering.
- Hit@K and MRR measure evidence ranking, not numerical answer correctness.
