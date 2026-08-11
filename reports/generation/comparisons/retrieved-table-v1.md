# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-no-llm-clarify-v1`
- Candidate：`retrieved_context-benchmark-v2-no-llm-table-v1`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：no-LLM A/B：确定性表格路径开关
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.0857 | 0.2571 | +0.1714 |
| Expected behavior accuracy | 0.8333 | 0.8333 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（6）

- `moutai_channel_margin`
- `moutai_quarterly_cashflow`
- `moutai_revenue_yoy`
- `yili_cost_reconciliation`
- `yili_note_cost_scope`
- `yili_quarterly_net_profit`

## Strict 回归（0）

- 无
