# RRF fusion-weight sweep v1

- dataset: `benchmark-v2-retrieval-view` | index: `10fb50419145d56720c9`
- top_k=5 | candidate_k=20 | rrf_k=60 | positive instances: 111
- 指标为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。
- per-regime best 是在评测集上选择的最优权重，属于 development-only 上限，不代表独立泛化结论。

## Query-level Hit@5（query_parser 过滤）

| weight | canonical | ticker | semantic/相对时间 |
|---|---:|---:|---:|
| 1.0:1.0 | 0.5946 | 0.7297 | 0.4324 |
| 2.0:1.0 | 0.6486 | 0.7297 | 0.4595 |
| 3.0:1.0 | 0.6486 | 0.7297 | 0.4595 |
| 4.0:1.0 | 0.7027 | 0.7297 | 0.5135 |
| 1.0:0.0 | 0.8378 | 0.8108 | 0.7297 |
| 0.0:1.0 | 0.2162 | 0.6757 | 0.1622 |

## Query-level MRR@5（query_parser 过滤）

| weight | canonical | ticker | semantic/相对时间 |
|---|---:|---:|---:|
| 1.0:1.0 | 0.3568 | 0.6396 | 0.3063 |
| 2.0:1.0 | 0.3653 | 0.6712 | 0.3207 |
| 3.0:1.0 | 0.3968 | 0.6757 | 0.3387 |
| 4.0:1.0 | 0.4171 | 0.6892 | 0.3495 |
| 1.0:0.0 | 0.6937 | 0.7239 | 0.6329 |
| 0.0:1.0 | 0.1459 | 0.4694 | 0.1126 |

## Per-regime best（query_parser 过滤，development-only）

| regime | best weight | Hit@5 | MRR@5 | NDCG@5 |
|---|---|---:|---:|---:|
| canonical | 1.0:0.0 | 0.8378 | 0.6937 | 0.7089 |
| ticker_or_finance_shorthand | 1.0:0.0 | 0.8108 | 0.7239 | 0.7188 |
| semantic_or_relative_time | 1.0:0.0 | 0.7297 | 0.6329 | 0.6083 |
