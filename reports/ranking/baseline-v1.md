# Chinese ranking diagnostic baseline v1

Dataset ID: `ba7aa7e2fd771fb28ff0`

Source index: `a9645108d5a7ce8b73b8`

Corpus: 贵州茅台 and 伊利股份 2024 annual reports, 958 chunks.

## Dataset composition

- 10 accepted structure-anchored queries;
- 10 relevant chunk judgments;
- 52 rule-confirmed hard negatives;
- 141 candidates retained as `needs_review` and excluded from negative scoring;
- hard negatives: 43 wrong-company, 8 wrong-scope, and 1 wrong-period.

The queries cover annual financial indicators, quarterly revenue, and segment
revenue. The generator produced 10 rather than padding to a target size because only
those query/spec combinations passed the current structural rules.

## Baseline results

| Retriever | Hit@5 | MRR@5 |
|---|---:|---:|
| SQLite BM25 | 0.7000 | 0.5500 |
| E5 + BM25 RRF hybrid | 0.6000 | 0.3200 |

Hybrid retrieval underperformed lexical retrieval on this small diagnostic slice.
For example, the quarterly and basic-EPS anchors often moved downward after fusion.
This is evidence of a ranking problem worth investigating, not evidence that dense
retrieval is generally worse.

The audit also exposed a missing capability: the query contains a company name, but
chunks and the index do not yet carry the registry's company identity as a searchable
or filterable field. Semantically similar evidence from another company can therefore
compete directly. Metadata-aware retrieval should be implemented before tuning RRF
weights against this slice.

## Claim boundary

These numbers are reproducible development diagnostics, not resume-ready accuracy
claims. The dataset is currently too small and narrow. It is already useful for
regression testing and for identifying concrete infrastructure work: metadata
filters, more complete relevance judgments, broader filing coverage, and reranker
comparison once the remote model is available.
