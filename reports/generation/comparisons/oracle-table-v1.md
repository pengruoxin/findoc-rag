# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-no-llm-clarify-v1`
- Candidate：`oracle_context-benchmark-v2-no-llm-table-v1`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：no-LLM A/B：确定性表格路径开关（FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.3143 | 0.6571 | +0.3429 |
| Expected behavior accuracy | 1.0000 | 1.0000 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（12）

- `moutai_channel_margin`
- `moutai_cost_reconciliation`
- `moutai_product_margin`
- `moutai_quarterly_cashflow`
- `moutai_revenue_yoy`
- `revenue_cross_company`
- `yili_consolidated_parent_revenue`
- `yili_cost_reconciliation`
- `yili_note_cost_scope`
- `yili_quarterly_cashflow_reconcile`
- `yili_quarterly_net_profit`
- `yili_quarterly_profit_reconcile`

## Strict 回归（0）

- 无
