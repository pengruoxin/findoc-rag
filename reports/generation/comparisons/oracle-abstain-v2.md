# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-table-v2`
- Candidate：`oracle_context-benchmark-v2-deepseek-chat-abstain-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：测量变更：远程拒答检测（grounded/abstain 判定 + provider=remote-abstention）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.9714 | 0.9429 | -0.0286 |
| Expected behavior accuracy | 1.0000 | 0.9792 | -0.0208 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（1）

- `yili_consolidated_parent_revenue`

## Strict 修复（1）

- `yili_quarterly_profit_reconcile`

## Strict 回归（2）

- `moutai_annual_deducted_profit`
- `yili_consolidated_parent_revenue`
