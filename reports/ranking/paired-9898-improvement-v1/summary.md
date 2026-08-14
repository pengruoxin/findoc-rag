# Paired retrieval comparison

- old: `migration-9898-raw-v1`
- new: `migration-9898-rewrite-routed-v1`
- configuration: filter=`query_parser`, mode=`lexical`
- positive pairs: 111 | exact metric matches: 91
- Hit@5 fixed/regressed: 12/0 | MRR improved/regressed: 14/3

| regime | old Hit@5 | new Hit@5 | Δ | old MRR@5 | new MRR@5 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.8378 | 0.8919 | +0.0541 | 0.6937 | 0.7387 | +0.0450 |
| ticker_or_finance_shorthand | 0.8108 | 0.8919 | +0.0811 | 0.7239 | 0.7599 | +0.0360 |
| semantic_or_relative_time | 0.7297 | 0.9189 | +0.1892 | 0.6329 | 0.7613 | +0.1284 |

## Hit@5 fixed

`customer_concentration_comparison::customer_concentration_comparison:v2`, `moutai_disclosed_risks::moutai_disclosed_risks:v2`, `moutai_revenue_yoy`, `moutai_revenue_yoy::moutai_revenue_yoy:v1`, `moutai_revenue_yoy::moutai_revenue_yoy:v2`, `moutai_roe::moutai_roe:v2`, `yili_consolidated_parent_revenue::yili_consolidated_parent_revenue:v1`, `yili_consolidated_parent_revenue::yili_consolidated_parent_revenue:v2`, `yili_disclosed_risks`, `yili_disclosed_risks::yili_disclosed_risks:v1`, `yili_disclosed_risks::yili_disclosed_risks:v2`, `yili_product_margin::yili_product_margin:v2`

## Hit@5 regressed

None.

## MRR improved

`customer_concentration_comparison::customer_concentration_comparison:v1`, `customer_concentration_comparison::customer_concentration_comparison:v2`, `moutai_disclosed_risks::moutai_disclosed_risks:v2`, `moutai_revenue_yoy`, `moutai_revenue_yoy::moutai_revenue_yoy:v1`, `moutai_revenue_yoy::moutai_revenue_yoy:v2`, `moutai_roe::moutai_roe:v2`, `yili_cashflow_change`, `yili_consolidated_parent_revenue::yili_consolidated_parent_revenue:v1`, `yili_consolidated_parent_revenue::yili_consolidated_parent_revenue:v2`, `yili_disclosed_risks`, `yili_disclosed_risks::yili_disclosed_risks:v1`, `yili_disclosed_risks::yili_disclosed_risks:v2`, `yili_product_margin::yili_product_margin:v2`

## MRR regressed

`moutai_cashflow_change::moutai_cashflow_change:v1`, `yili_cashflow_change::yili_cashflow_change:v1`, `yili_cashflow_change::yili_cashflow_change:v2`
