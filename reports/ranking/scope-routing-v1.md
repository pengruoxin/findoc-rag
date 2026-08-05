# Scope-routing experiment v1

Dataset: `ba7aa7e2fd771fb28ff0` (10 structure-anchored queries).

Primary configuration: `candidate_k=20`, `top_k=5`, exact company/year filters.

| Pipeline | Hit@5 | MRR@5 |
|---|---:|---:|
| BM25 + metadata | 0.7000 | 0.7000 |
| BM25 + metadata + scope | 0.8000 | 0.8000 |
| Hybrid + metadata | 0.5000 | 0.2783 |
| Hybrid + metadata + scope | 0.8000 | 0.8000 |

Eight recovered anchors were ranked first. The two misses were both annual total
revenue queries whose gold chunks did not enter the 20-candidate pool. This separates
candidate recall failures from scope-ordering failures.

Candidate-depth sensitivity for hybrid + metadata + scope:

| candidate_k | Hit@5 | MRR@5 |
|---:|---:|---:|
| 20 | 0.8000 | 0.8000 |
| 100 | 0.9000 | 0.9000 |
| 400 | 1.0000 | 1.0000 |

The k=400 result is a diagnostic upper bound, not a viable default: it increases
dense scoring and downstream reranking cost. More importantly, the dataset positives
and router both rely on section cues, so this experiment is partly circular. It proves
pipeline behavior and exposes candidate-recall limits, but does not constitute an
independent quality estimate. A human-reviewed holdout set is required before using
these figures as resume accuracy claims.
