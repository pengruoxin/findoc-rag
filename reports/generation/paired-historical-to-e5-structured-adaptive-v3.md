# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-table-remote-v1`
- Candidate：`retrieved_context-benchmark-v2-deepseek-structured-adaptive-v3`
- 代码版本：baseline `640ab99b9cbd609d8be7b24d620caccee44ebc01` / candidate `30d108d604f22e4db9588613614848b996917618`（不同）
- 代码状态：不同或不可证明；fingerprint 不一致/缺失
- 受控变量：E5 migration + production query routing + structured-table evidence routing + adaptive candidate budget + scorer/guard fixes; DeepSeek output is stochastic
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8286 | 0.9429 | +0.1143 |
| Expected behavior accuracy | 0.9583 | 1.0000 | +0.0417 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（2）

- `moutai_product_margin`
- `yili_quarterly_cashflow_reconcile`

## 行为回归（0）

- 无

## Strict 修复（5）

- `moutai_product_margin`
- `revenue_cross_company`
- `yili_annual_deducted_profit`
- `yili_consolidated_parent_revenue`
- `yili_quarterly_cashflow_reconcile`

## Strict 回归（1）

- `moutai_annual_deducted_profit`
