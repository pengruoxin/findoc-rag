# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-deepseek-chat-table-remote-v1`
- Candidate：`robustness-benchmark-v2-deepseek-chat-concentration-v1`
- 代码版本：baseline `640ab99b9cbd609d8be7b24d620caccee44ebc01` / candidate `c828639d856ea5b6bc24df0893be895cccc3af8f`（不同）
- 受控变量：新增 concentration 表型抽取与生成路径（单一能力变更，其余固定）
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8636 | 0.8636 | +0.0000 |
| Expected behavior accuracy | 0.8621 | 0.9655 | +0.1034 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（3）

- `audit_opinion_comparison`
- `moutai_concentration`
- `yili_concentration`

## 行为回归（0）

- 无

## Strict 修复（0）

- 无

## Strict 回归（0）

- 无
