# Paired retrieval comparison

- old: `historical-10fb-expanded-v2`
- new: `migration-9898-routed-v1`
- configuration: filter=`query_parser`, mode=`lexical`
- positive pairs: 111 | exact metric matches: 109
- Hit@5 fixed/regressed: 0/0 | MRR improved/regressed: 0/0

| regime | old Hit@5 | new Hit@5 | Δ | old MRR@5 | new MRR@5 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.8919 | 0.8919 | +0.0000 | 0.7387 | 0.7387 | +0.0000 |
| ticker_or_finance_shorthand | 0.8919 | 0.8919 | +0.0000 | 0.7599 | 0.7599 | +0.0000 |
| semantic_or_relative_time | 0.9189 | 0.9189 | +0.0000 | 0.7613 | 0.7613 | +0.0000 |

## Hit@5 fixed

None.

## Hit@5 regressed

None.

## MRR improved

None.

## MRR regressed

None.
