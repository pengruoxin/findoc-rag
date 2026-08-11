# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-concentration-v2`
- Candidate：`oracle_context-benchmark-v2-deepseek-chat-rewrite-llm-v1`
- 代码版本：baseline `f8a1be898b9a67d9684a409b73bf8e9889d325de` / candidate `271588fb09da43ba48984190361b846408a9d26a`（不同）
- 受控变量：retrieved lane 查询改写模式：deterministic -> llm（含持久化缓存，单变量）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 1.0000 | 0.9714 | -0.0286 |
| Expected behavior accuracy | 1.0000 | 1.0000 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（0）

- 无

## Strict 回归（1）

- `moutai_annual_deducted_profit`
