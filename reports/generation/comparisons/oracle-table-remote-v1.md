# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-abstain-v2`
- Candidate：`oracle_context-benchmark-v2-deepseek-chat-table-remote-v1`
- 代码版本：baseline `None` / candidate `640ab99b9cbd609d8be7b24d620caccee44ebc01`（不同）
- 受控变量：远程模式启用确定性表格优先（单变量：FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES）
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.9429 | 1.0000 | +0.0571 |
| Expected behavior accuracy | 0.9792 | 1.0000 | +0.0208 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（1）

- `yili_consolidated_parent_revenue`

## 行为回归（0）

- 无

## Strict 修复（2）

- `moutai_annual_deducted_profit`
- `yili_consolidated_parent_revenue`

## Strict 回归（0）

- 无
