# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-table-v2`
- Candidate：`retrieved_context-benchmark-v2-deepseek-chat-abstain-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：测量变更：远程拒答检测（grounded/abstain 判定 + provider=remote-abstention）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.5714 | 0.8000 | +0.2286 |
| Expected behavior accuracy | 0.8333 | 0.8958 | +0.0625 |
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

## 行为回归（5）

- `moutai_product_margin`
- `moutai_quarterly_cashflow`
- `yili_2025_plan_bounded`
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
