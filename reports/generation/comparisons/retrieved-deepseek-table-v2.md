# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-v2`
- Candidate：`retrieved_context-benchmark-v2-deepseek-chat-table-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：非受控参考：8/7 基线早于同义词改写 runner 与 api_model 元数据修正，仅趋势观察
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.5429 | 0.5714 | +0.0286 |
| Expected behavior accuracy | 0.8333 | 0.8333 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（2）

- `moutai_annual_deducted_profit`
- `moutai_revenue_yoy`

## Strict 回归（1）

- `moutai_quarterly_cashflow`
