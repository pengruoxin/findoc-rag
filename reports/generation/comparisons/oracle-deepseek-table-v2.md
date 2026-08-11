# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-v2`
- Candidate：`oracle_context-benchmark-v2-deepseek-chat-table-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：非受控参考：8/7 基线早于同义词改写 runner 与 api_model 元数据修正，仅趋势观察
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.9714 | 0.9714 | +0.0000 |
| Expected behavior accuracy | 1.0000 | 1.0000 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（1）

- `moutai_annual_deducted_profit`

## Strict 回归（1）

- `yili_quarterly_profit_reconcile`
