# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-no-llm-clarify-v1`
- Candidate：`robustness-benchmark-v2-no-llm-table-v1`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：no-LLM A/B：确定性表格路径开关
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.2273 | 0.3636 | +0.1364 |
| Expected behavior accuracy | 0.7931 | 0.7931 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（3）

- `moutai_revenue_yoy`
- `yili_note_cost_scope`
- `yili_quarterly_profit_reconcile`

## Strict 回归（0）

- 无
