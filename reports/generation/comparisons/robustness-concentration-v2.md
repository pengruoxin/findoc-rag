# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-deepseek-chat-concentration-v1`
- Candidate：`robustness-benchmark-v2-deepseek-chat-concentration-v2`
- 代码版本：baseline `c828639d856ea5b6bc24df0893be895cccc3af8f` / candidate `f8a1be898b9a67d9684a409b73bf8e9889d325de`（不同）
- 受控变量：修复 concentration 单公司选题 bug：按查询中的公司选取（Robustness 负例前置场景）
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.8636 | 0.9545 | +0.0909 |
| Expected behavior accuracy | 0.9655 | 0.9655 | +0.0000 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（0）

- 无

## 行为回归（0）

- 无

## Strict 修复（2）

- `moutai_concentration`
- `yili_concentration`

## Strict 回归（0）

- 无
