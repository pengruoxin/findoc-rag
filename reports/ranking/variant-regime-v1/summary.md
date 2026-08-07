# Variant-regime retrieval evaluation v1

- dataset: `benchmark-v2-retrieval-view` | index: `10fb50419145d56720c9`
- top_k=5 | candidate_k=20 | rrf_k=60 | weights lexical=2.0:dense=1.0
- positive instances: 111 (canonical groups: 37)

指标均为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。

## Query-level 平均（每个 query instance 一票）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 | cand_recall | neg_in_top5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | none | lexical | 0.7568 | 0.5586 | 0.7162 | 0.1892 | 0.5817 | 0.9189 | 0.2973 |
| canonical | none | dense | 0.2162 | 0.1676 | 0.2027 | 0.0541 | 0.1699 | 0.4324 | 0.0270 |
| canonical | none | hybrid | 0.6757 | 0.3932 | 0.6351 | 0.1730 | 0.4480 | 0.9189 | 0.2432 |
| canonical | query_parser | lexical | 0.8378 | 0.6937 | 0.8108 | 0.2216 | 0.7089 | 0.8919 | 0.1081 |
| canonical | query_parser | dense | 0.2162 | 0.1459 | 0.2027 | 0.0541 | 0.1533 | 0.4054 | 0.0000 |
| canonical | query_parser | hybrid | 0.6486 | 0.3653 | 0.5946 | 0.1676 | 0.4169 | 0.8919 | 0.0811 |
| ticker_or_finance_shorthand | none | lexical | 0.7838 | 0.5707 | 0.7297 | 0.1892 | 0.5948 | 0.8649 | 0.2973 |
| ticker_or_finance_shorthand | none | dense | 0.5946 | 0.3802 | 0.5541 | 0.1351 | 0.4192 | 0.8378 | 0.1892 |
| ticker_or_finance_shorthand | none | hybrid | 0.7838 | 0.5910 | 0.7095 | 0.1838 | 0.6011 | 0.8649 | 0.2432 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8108 | 0.7239 | 0.7703 | 0.2054 | 0.7188 | 0.8378 | 0.1081 |
| ticker_or_finance_shorthand | query_parser | dense | 0.6757 | 0.4694 | 0.6216 | 0.1514 | 0.4972 | 0.8108 | 0.0541 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7297 | 0.6712 | 0.6689 | 0.1784 | 0.6468 | 0.8378 | 0.0541 |
| semantic_or_relative_time | none | lexical | 0.7297 | 0.5887 | 0.6622 | 0.1784 | 0.5806 | 0.8108 | 0.2432 |
| semantic_or_relative_time | none | dense | 0.1892 | 0.1374 | 0.1757 | 0.0432 | 0.1458 | 0.3784 | 0.0270 |
| semantic_or_relative_time | none | hybrid | 0.5946 | 0.3563 | 0.5338 | 0.1459 | 0.3810 | 0.8108 | 0.1892 |
| semantic_or_relative_time | query_parser | lexical | 0.7297 | 0.6329 | 0.6622 | 0.1784 | 0.6083 | 0.7838 | 0.1351 |
| semantic_or_relative_time | query_parser | dense | 0.1622 | 0.1126 | 0.1486 | 0.0378 | 0.1199 | 0.3514 | 0.0270 |
| semantic_or_relative_time | query_parser | hybrid | 0.4595 | 0.3207 | 0.4257 | 0.1189 | 0.3306 | 0.7838 | 0.0811 |

## Canonical-group 平均（先组内聚合再平均，防伪重复）

| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
|---|---|---|---:|---:|---:|---:|
| canonical | none | lexical | 0.7568 | 0.5586 | 0.7162 | 0.5817 |
| canonical | none | dense | 0.2162 | 0.1676 | 0.2027 | 0.1699 |
| canonical | none | hybrid | 0.6757 | 0.3932 | 0.6351 | 0.4480 |
| canonical | query_parser | lexical | 0.8378 | 0.6937 | 0.8108 | 0.7089 |
| canonical | query_parser | dense | 0.2162 | 0.1459 | 0.2027 | 0.1533 |
| canonical | query_parser | hybrid | 0.6486 | 0.3653 | 0.5946 | 0.4169 |
| ticker_or_finance_shorthand | none | lexical | 0.7838 | 0.5707 | 0.7297 | 0.5948 |
| ticker_or_finance_shorthand | none | dense | 0.5946 | 0.3802 | 0.5541 | 0.4192 |
| ticker_or_finance_shorthand | none | hybrid | 0.7838 | 0.5910 | 0.7095 | 0.6011 |
| ticker_or_finance_shorthand | query_parser | lexical | 0.8108 | 0.7239 | 0.7703 | 0.7188 |
| ticker_or_finance_shorthand | query_parser | dense | 0.6757 | 0.4694 | 0.6216 | 0.4972 |
| ticker_or_finance_shorthand | query_parser | hybrid | 0.7297 | 0.6712 | 0.6689 | 0.6468 |
| semantic_or_relative_time | none | lexical | 0.7297 | 0.5887 | 0.6622 | 0.5806 |
| semantic_or_relative_time | none | dense | 0.1892 | 0.1374 | 0.1757 | 0.1458 |
| semantic_or_relative_time | none | hybrid | 0.5946 | 0.3563 | 0.5338 | 0.3810 |
| semantic_or_relative_time | query_parser | lexical | 0.7297 | 0.6329 | 0.6622 | 0.6083 |
| semantic_or_relative_time | query_parser | dense | 0.1622 | 0.1126 | 0.1486 | 0.1199 |
| semantic_or_relative_time | query_parser | hybrid | 0.4595 | 0.3207 | 0.4257 | 0.3306 |
