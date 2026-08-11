# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-table-v2`
- Candidate：`retrieved_context-benchmark-v2-deepseek-chat-abstain-v1`
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.5714 | 0.8000 | +0.2286 |
| Expected behavior accuracy | 0.8333 | 0.8750 | +0.0417 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（8）

- `u_compare_audit_quality`
- `u_investment_advice`
- `u_moutai_liquid_milk`
- `u_moutai_stock_cause`
- `u_yili_moutai_wine`
- `u_yili_profit_causality`
- `u_yili_q4_loss_cause`
- `u_yili_top_customer_names`

## 行为回归（6）

- `moutai_cost_reconciliation`
- `moutai_product_margin`
- `moutai_quarterly_cashflow`
- `yili_annual_deducted_profit`
- `yili_consolidated_parent_revenue`
- `yili_quarterly_cashflow_reconcile`

## Strict 修复（8）

- `u_compare_audit_quality`
- `u_investment_advice`
- `u_moutai_liquid_milk`
- `u_moutai_stock_cause`
- `u_yili_moutai_wine`
- `u_yili_profit_causality`
- `u_yili_q4_loss_cause`
- `u_yili_top_customer_names`

## Strict 回归（0）

- 无
