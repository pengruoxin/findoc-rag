# FinanceBench bootstrap retrieval baseline

## Scope

- 150 public FinanceBench questions
- 168 unique gold-evidence pages used as the retrieval corpus
- BM25 lexical retrieval versus `intfloat/e5-small-v2` dense retrieval
- Top-k is computed globally across the 168 pages

This is a pipeline-validation experiment. The public QA sample does not contain
every page of every source 10-K, so these numbers must not be presented as
full-document FinanceBench performance.

## Results

| Metric | BM25 | Dense | Absolute change |
|---|---:|---:|---:|
| Hit@1 | 0.2133 | 0.5267 | +0.3134 |
| Hit@5 | 0.3733 | 0.8800 | +0.5067 |
| Hit@10 | 0.4333 | 0.9133 | +0.4800 |
| Recall@1 | 0.2000 | 0.4433 | +0.2433 |
| Recall@5 | 0.3478 | 0.8444 | +0.4966 |
| Recall@10 | 0.4078 | 0.8978 | +0.4900 |
| MRR | 0.2736 | 0.6694 | +0.3958 |

First-gold-rank comparison:

- Dense improved 100 of 150 questions.
- BM25 improved 17 of 150 questions.
- 33 questions tied.

## Initial failure analysis

Dense retrieval fixed several lexical mismatches, including questions phrased
with `capital expenditure` or `net PP&E` whose evidence uses longer accounting
line-item descriptions such as `purchases of property, plant and equipment`.

BM25 still ranked some qualitative questions better, including questions about
specific acquisitions, geographic operations, restructuring liabilities, and
production-rate changes. This supports testing hybrid retrieval rather than
replacing lexical retrieval outright.

## Next experiment

Implement Reciprocal Rank Fusion (RRF) over BM25 and dense rankings, then test
whether hybrid retrieval recovers BM25's exact-match strengths without losing
the semantic gains from dense retrieval.

