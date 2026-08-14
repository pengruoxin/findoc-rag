# RRF fusion-weight sweep v1

- dataset: `benchmark-v2-retrieval-view` | split: `dev` | index: `9898c95e13d01c51c156`
- top_k=5 | candidate_k=20 | rrf_k=60 | positive instances: 57
- 指标为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。
- per-regime best 是在评测集上选择的最优权重，属于 development-only 上限，不代表独立泛化结论。

## Query-level Hit@5（query_parser 过滤）

| weight | canonical | ticker | semantic/相对时间 |
|---|---:|---:|---:|
| 1.0:1.0 | 0.5263 | 0.7368 | 0.5263 |
| 2.0:1.0 | 0.5789 | 0.7368 | 0.5789 |
| 3.0:1.0 | 0.5789 | 0.7368 | 0.5789 |
| 4.0:1.0 | 0.7368 | 0.7368 | 0.6842 |
| 1.0:0.0 | 0.8947 | 0.9474 | 0.8947 |
| 0.0:1.0 | 0.2632 | 0.6842 | 0.2105 |

## Query-level MRR@5（query_parser 过滤）

| weight | canonical | ticker | semantic/相对时间 |
|---|---:|---:|---:|
| 1.0:1.0 | 0.3991 | 0.6754 | 0.3465 |
| 2.0:1.0 | 0.3877 | 0.6842 | 0.3439 |
| 3.0:1.0 | 0.4228 | 0.6842 | 0.3789 |
| 4.0:1.0 | 0.4570 | 0.6842 | 0.4114 |
| 1.0:0.0 | 0.7632 | 0.8509 | 0.7368 |
| 0.0:1.0 | 0.1947 | 0.5088 | 0.1842 |

## Per-regime best（query_parser 过滤，development-only）

| regime | best weight | Hit@5 | MRR@5 | NDCG@5 |
|---|---|---:|---:|---:|
| canonical | 1.0:0.0 | 0.8947 | 0.7632 | 0.7890 |
| ticker_or_finance_shorthand | 1.0:0.0 | 0.9474 | 0.8509 | 0.8726 |
| semantic_or_relative_time | 1.0:0.0 | 0.8947 | 0.7368 | 0.7724 |
