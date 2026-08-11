# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-concentration-v1`
- Candidate：`oracle_context-benchmark-v2-deepseek-chat-concentration-v2`
- 代码版本：baseline `c828639d856ea5b6bc24df0893be895cccc3af8f` / candidate `f8a1be898b9a67d9684a409b73bf8e9889d325de`（不同）
- 受控变量：修复 concentration 单公司选题 bug：按查询中的公司选取（Robustness 负例前置场景）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 1.0000 | 1.0000 | +0.0000 |
| Expected behavior accuracy | 1.0000 | 1.0000 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（0）

- 无

## Strict 回归（0）

- 无
