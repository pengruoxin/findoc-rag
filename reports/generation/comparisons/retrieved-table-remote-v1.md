# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-deepseek-chat-abstain-v2`
- Candidate：`retrieved_context-benchmark-v2-deepseek-chat-table-remote-v1`
- 代码版本：baseline `None` / candidate `640ab99b9cbd609d8be7b24d620caccee44ebc01`（不同）
- 受控变量：远程模式启用确定性表格优先（单变量：FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8000 | 0.8286 | +0.0286 |
| Expected behavior accuracy | 0.8958 | 0.9583 | +0.0625 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（3）

- `moutai_quarterly_cashflow`
- `yili_2025_plan_bounded`
- `yili_consolidated_parent_revenue`

## 行为回归（0）

- 无

## Strict 修复（1）

- `moutai_quarterly_cashflow`

## Strict 回归（0）

- 无
