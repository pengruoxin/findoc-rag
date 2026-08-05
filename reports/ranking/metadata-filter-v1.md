# Metadata-aware retrieval experiment v1

The corpus was migrated to index format v3 and generation
`10fb50419145d56720c9`. Reviewed profiles supplied company name, report year, and
document type for both active annual reports. Exact filters were applied inside both
lexical and dense retrieval before RRF.

The migration rebuilt 958 embeddings because v2 generations are intentionally not
opened as reuse parents by a v3 reader. This is a one-time format migration cost.

| Retriever | No-filter Hit@5 | Filtered Hit@5 | No-filter MRR@5 | Filtered MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.7000 | 0.7000 | 0.5500 | 0.7000 |
| E5 + BM25 RRF | 0.6000 | 0.5000 | 0.3200 | 0.2783 |

Company/year filtering improved BM25 ordering: five retrieved anchors moved to rank
1. It did not recover three annual-summary anchors already outside the top five.

Hybrid quality declined on this small slice. Removing cross-company candidates
changes RRF component ranks, while dense retrieval still prefers passages with the
wrong within-document scope. The next fix should be scope-aware retrieval or
reranking, not unprincipled RRF weight tuning.

Filters remain necessary for correctness and document isolation even when one
aggregate ranking metric does not improve. This experiment shows that filters and
rankers must be evaluated together.
