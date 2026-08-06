# Independent holdout review workflow

The review pack is deliberately separate from the scored diagnostic dataset. Generated
questions are candidates only; their proposed evidence is a suggestion, not frozen
gold. This prevents scope-router rules from silently becoming evaluation labels.

Generate a new pack after the active corpus is built:

```powershell
uv run findoc-rag generate-holdout-review `
  configs/ranking-diagnostic-profiles.json
```

Outputs:

- `data/diagnostics/holdout-review-v1.json`: machine-editable review records;
- `reports/ranking/holdout-review-v1.md`: readable review sheet.

For each item, review whether the question is unambiguous about company, year, period,
unit, and scope; whether the proposed passage directly answers it; and whether an
alternative passage should also be relevant. Mark ambiguous or unsupported items
`rejected`.

Set `status` to `approved`, `rejected`, or `edited`; for edited items fill
`reviewer_query`, `reviewer_gold_chunk_ids`, and `reviewer_notes`. The future evaluator
will consume only approved/edited items and never treat pending items as gold.

The first pack (`f19011fa40507d1f1efe`) contains 16 pending candidates from the two
2024 annual reports and excludes the 10 queries used by existing routing diagnostics.
It is a review queue, not yet an evaluation benchmark.
