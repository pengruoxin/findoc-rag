# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-table-remote-v1`
- Candidate：`retrieved_context-benchmark-v2-deepseek-migration-v2`
- 代码版本：baseline `640ab99b9cbd609d8be7b24d620caccee44ebc01` / candidate `30d108d604f22e4db9588613614848b996917618`（不同）
- 受控变量：新 E5 迁移索引 + query-derived production routing + deterministic rewrite；DeepSeek 输出存在非确定性，非严格单变量
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8286 | 0.7714 | -0.0571 |
| Expected behavior accuracy | 0.9583 | 0.9583 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（0）

- 无

## Strict 回归（2）

- `moutai_annual_deducted_profit`
- `yili_quarterly_profit_reconcile`
