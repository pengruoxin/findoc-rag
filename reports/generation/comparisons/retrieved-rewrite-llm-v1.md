# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-concentration-v2`
- Candidate：`retrieved_context-benchmark-v2-deepseek-chat-rewrite-llm-v1`
- 代码版本：baseline `f8a1be898b9a67d9684a409b73bf8e9889d325de` / candidate `271588fb09da43ba48984190361b846408a9d26a`（不同）
- 受控变量：retrieved lane 查询改写模式：deterministic -> llm（含持久化缓存，单变量）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8286 | 0.8286 | +0.0000 |
| Expected behavior accuracy | 0.9375 | 0.8958 | -0.0417 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（2）

- `moutai_disclosed_risks`
- `yili_disclosed_risks`

## Strict 修复（1）

- `yili_annual_deducted_profit`

## Strict 回归（1）

- `moutai_revenue_yoy`
