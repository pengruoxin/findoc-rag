# Variant-regime retrieval evaluation v1

- dataset: `benchmark-v2-retrieval-view` | index: `6a951f4e8b7bd913d918`
- top_k=5 | candidate_k=20 | rrf_k=60 | weights lexical=2.0:dense=1.0
- positive instances: 111 (canonical groups: 37)

指标均为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。

## Query-level 平均（每个 query instance 一票）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 | cand_recall | neg_in_top5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | none | lexical | 0.7568 | 0.5586 | 0.7162 | 0.1892 | 0.5817 | 0.9189 | 0.2973 |
| canonical | none | dense | 0.1622 | 0.1306 | 0.1284 | 0.0378 | 0.1178 | 0.4595 | 0.0270 |
| canonical | none | hybrid | 0.6757 | 0.3847 | 0.6284 | 0.1676 | 0.4377 | 0.9189 | 0.2703 |
| canonical | query_parser | lexical | 0.8378 | 0.6937 | 0.8108 | 0.2216 | 0.7089 | 0.8919 | 0.1081 |
| canonical | query_parser | dense | 0.1622 | 0.1306 | 0.1284 | 0.0378 | 0.1200 | 0.4324 | 0.0270 |
| canonical | query_parser | hybrid | 0.6757 | 0.3617 | 0.6284 | 0.1676 | 0.4218 | 0.8919 | 0.1081 |
| ticker_or_finance_shorthand | none | lexical | 0.7838 | 0.5707 | 0.7297 | 0.1892 | 0.5948 | 0.8649 | 0.2973 |
| ticker_or_finance_shorthand | none | dense | 0.4595 | 0.2698 | 0.3919 | 0.1081 | 0.2900 | 0.7027 | 0.1351 |
| ticker_or_finance_shorthand | none | hybrid | 0.7568 | 0.4982 | 0.7095 | 0.1838 | 0.5325 | 0.8649 | 0.1892 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8108 | 0.7239 | 0.7703 | 0.2054 | 0.7188 | 0.8378 | 0.1081 |
| ticker_or_finance_shorthand | query_parser | dense | 0.5676 | 0.3495 | 0.5135 | 0.1351 | 0.3812 | 0.7297 | 0.0270 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7568 | 0.5838 | 0.7095 | 0.1892 | 0.5907 | 0.8378 | 0.0270 |
| semantic_or_relative_time | none | lexical | 0.7297 | 0.5887 | 0.6622 | 0.1784 | 0.5806 | 0.8108 | 0.2432 |
| semantic_or_relative_time | none | dense | 0.1622 | 0.1306 | 0.1284 | 0.0378 | 0.1167 | 0.2973 | 0.0270 |
| semantic_or_relative_time | none | hybrid | 0.5135 | 0.2811 | 0.4662 | 0.1243 | 0.3187 | 0.8108 | 0.1622 |
| semantic_or_relative_time | query_parser | lexical | 0.7297 | 0.6329 | 0.6622 | 0.1784 | 0.6083 | 0.7838 | 0.1351 |
| semantic_or_relative_time | query_parser | dense | 0.1622 | 0.1306 | 0.1284 | 0.0378 | 0.1167 | 0.2973 | 0.0270 |
| semantic_or_relative_time | query_parser | hybrid | 0.4865 | 0.2743 | 0.4122 | 0.1135 | 0.2925 | 0.7838 | 0.1081 |

## Canonical-group 平均（先组内聚合再平均，防伪重复）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
|---|---|---|---:|---:|---:|---:|
| canonical | none | lexical | 0.7568 | 0.5586 | 0.7162 | 0.5817 |
| canonical | none | dense | 0.1622 | 0.1306 | 0.1284 | 0.1178 |
| canonical | none | hybrid | 0.6757 | 0.3847 | 0.6284 | 0.4377 |
| canonical | query_parser | lexical | 0.8378 | 0.6937 | 0.8108 | 0.7089 |
| canonical | query_parser | dense | 0.1622 | 0.1306 | 0.1284 | 0.1200 |
| canonical | query_parser | hybrid | 0.6757 | 0.3617 | 0.6284 | 0.4218 |
| ticker_or_finance_shorthand | none | lexical | 0.7838 | 0.5707 | 0.7297 | 0.5948 |
| ticker_or_finance_shorthand | none | dense | 0.4595 | 0.2698 | 0.3919 | 0.2900 |
| ticker_or_finance_shorthand | none | hybrid | 0.7568 | 0.4982 | 0.7095 | 0.5325 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8108 | 0.7239 | 0.7703 | 0.7188 |
| ticker_or_finance_shorthand | query_parser | dense | 0.5676 | 0.3495 | 0.5135 | 0.3812 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7568 | 0.5838 | 0.7095 | 0.5907 |
| semantic_or_relative_time | none | lexical | 0.7297 | 0.5887 | 0.6622 | 0.5806 |
| semantic_or_relative_time | none | dense | 0.1622 | 0.1306 | 0.1284 | 0.1167 |
| semantic_or_relative_time | none | hybrid | 0.5135 | 0.2811 | 0.4662 | 0.3187 |
| semantic_or_relative_time | query_parser | lexical | 0.7297 | 0.6329 | 0.6622 | 0.6083 |
| semantic_or_relative_time | query_parser | dense | 0.1622 | 0.1306 | 0.1284 | 0.1167 |
| semantic_or_relative_time | query_parser | hybrid | 0.4865 | 0.2743 | 0.4122 | 0.2925 |
