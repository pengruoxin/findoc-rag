# Variant-regime retrieval evaluation v1

- dataset: `benchmark-v2-retrieval-view` | index: `9898c95e13d01c51c156`
- top_k=5 | candidate_k=20 | rrf_k=60 | weights lexical=2.0:dense=1.0
- positive instances: 111 (canonical groups: 37)

指标均为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。

## Query-level 平均（每个 query instance 一票）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 | cand_recall | neg_in_top5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | none | lexical | 0.8108 | 0.5991 | 0.7703 | 0.2054 | 0.6237 | 0.9730 | 0.3514 |
| canonical | none | dense | 0.2432 | 0.1766 | 0.2297 | 0.0595 | 0.1834 | 0.4324 | 0.0270 |
| canonical | none | hybrid | 0.7568 | 0.4212 | 0.7027 | 0.1892 | 0.4848 | 0.9730 | 0.2703 |
| canonical | query_parser | lexical | 0.8919 | 0.7387 | 0.8649 | 0.2378 | 0.7543 | 0.9459 | 0.1351 |
| canonical | query_parser | dense | 0.2162 | 0.1631 | 0.2027 | 0.0541 | 0.1663 | 0.4054 | 0.0000 |
| canonical | query_parser | hybrid | 0.6757 | 0.3797 | 0.6216 | 0.1730 | 0.4342 | 0.9459 | 0.1081 |
| ticker_or_finance_shorthand | none | lexical | 0.8649 | 0.6189 | 0.7973 | 0.2108 | 0.6452 | 0.9459 | 0.3514 |
| ticker_or_finance_shorthand | none | dense | 0.6486 | 0.4153 | 0.6081 | 0.1514 | 0.4614 | 0.8378 | 0.2162 |
| ticker_or_finance_shorthand | none | hybrid | 0.8108 | 0.6090 | 0.7365 | 0.1946 | 0.6217 | 0.9459 | 0.2973 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8919 | 0.7599 | 0.8514 | 0.2324 | 0.7665 | 0.9189 | 0.1351 |
| ticker_or_finance_shorthand | query_parser | dense | 0.7027 | 0.5000 | 0.6486 | 0.1622 | 0.5292 | 0.8108 | 0.0270 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7838 | 0.6937 | 0.7230 | 0.1946 | 0.6774 | 0.9189 | 0.0811 |
| semantic_or_relative_time | none | lexical | 0.8919 | 0.6518 | 0.8243 | 0.2216 | 0.6687 | 0.9730 | 0.3784 |
| semantic_or_relative_time | none | dense | 0.2162 | 0.1464 | 0.2027 | 0.0486 | 0.1593 | 0.4595 | 0.0541 |
| semantic_or_relative_time | none | hybrid | 0.7297 | 0.4194 | 0.6689 | 0.1784 | 0.4644 | 0.9730 | 0.2432 |
| semantic_or_relative_time | query_parser | lexical | 0.9189 | 0.7613 | 0.8514 | 0.2324 | 0.7541 | 0.9459 | 0.1622 |
| semantic_or_relative_time | query_parser | dense | 0.1892 | 0.1261 | 0.1757 | 0.0432 | 0.1370 | 0.4324 | 0.0270 |
| semantic_or_relative_time | query_parser | hybrid | 0.5676 | 0.3667 | 0.5338 | 0.1459 | 0.3939 | 0.9459 | 0.1081 |

## Canonical-group 平均（先组内聚合再平均，防伪重复）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
|---|---|---|---:|---:|---:|---:|
| canonical | none | lexical | 0.8108 | 0.5991 | 0.7703 | 0.6237 |
| canonical | none | dense | 0.2432 | 0.1766 | 0.2297 | 0.1834 |
| canonical | none | hybrid | 0.7568 | 0.4212 | 0.7027 | 0.4848 |
| canonical | query_parser | lexical | 0.8919 | 0.7387 | 0.8649 | 0.7543 |
| canonical | query_parser | dense | 0.2162 | 0.1631 | 0.2027 | 0.1663 |
| canonical | query_parser | hybrid | 0.6757 | 0.3797 | 0.6216 | 0.4342 |
| ticker_or_finance_shorthand | none | lexical | 0.8649 | 0.6189 | 0.7973 | 0.6452 |
| ticker_or_finance_shorthand | none | dense | 0.6486 | 0.4153 | 0.6081 | 0.4614 |
| ticker_or_finance_shorthand | none | hybrid | 0.8108 | 0.6090 | 0.7365 | 0.6217 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8919 | 0.7599 | 0.8514 | 0.7665 |
| ticker_or_finance_shorthand | query_parser | dense | 0.7027 | 0.5000 | 0.6486 | 0.5292 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7838 | 0.6937 | 0.7230 | 0.6774 |
| semantic_or_relative_time | none | lexical | 0.8919 | 0.6518 | 0.8243 | 0.6687 |
| semantic_or_relative_time | none | dense | 0.2162 | 0.1464 | 0.2027 | 0.1593 |
| semantic_or_relative_time | none | hybrid | 0.7297 | 0.4194 | 0.6689 | 0.4644 |
| semantic_or_relative_time | query_parser | lexical | 0.9189 | 0.7613 | 0.8514 | 0.7541 |
| semantic_or_relative_time | query_parser | dense | 0.1892 | 0.1261 | 0.1757 | 0.1370 |
| semantic_or_relative_time | query_parser | hybrid | 0.5676 | 0.3667 | 0.5338 | 0.3939 |
