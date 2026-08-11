# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-deepseek-chat-table-v2`
- Candidate：`robustness-benchmark-v2-deepseek-chat-abstain-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：测量变更：远程拒答检测（grounded/abstain 判定 + provider=remote-abstention）
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.6364 | 0.8636 | +0.2273 |
| Expected behavior accuracy | 0.7931 | 0.8276 | +0.0345 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（6）

- `u_compare_audit_quality`
- `u_investment_advice`
- `u_moutai_stock_cause`
- `u_yili_profit_causality`
- `u_yili_q4_loss_cause`
- `u_yili_top_customer_names`

## 行为回归（5）

- `audit_opinion_comparison`
- `moutai_concentration`
- `yili_2025_plan_bounded`
- `yili_concentration`
- `yili_consolidated_parent_revenue`

## Strict 修复（6）

- `u_compare_audit_quality`
- `u_investment_advice`
- `u_moutai_stock_cause`
- `u_yili_profit_causality`
- `u_yili_q4_loss_cause`
- `u_yili_top_customer_names`

## Strict 回归（1）

- `yili_concentration`
