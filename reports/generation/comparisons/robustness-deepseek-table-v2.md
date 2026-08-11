# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-deepseek-chat`
- Candidate：`robustness-benchmark-v2-deepseek-chat-table-v2`
- 代码版本：baseline `None` / candidate `None`（不同）
- 受控变量：非受控参考：8/7 基线早于同义词改写 runner 与 api_model 元数据修正，仅趋势观察
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.5455 | 0.6364 | +0.0909 |
| Expected behavior accuracy | 0.7931 | 0.7931 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（2）

- `yili_concentration`
- `yili_quarterly_profit_reconcile`

## Strict 回归（0）

- 无
