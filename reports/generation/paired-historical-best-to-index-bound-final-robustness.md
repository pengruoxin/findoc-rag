# 生成评测配对对比：robustness

- Dataset：`benchmark-v2`
- Baseline：`robustness-benchmark-v2-deepseek-chat-concentration-v2`
- Candidate：`robustness-benchmark-v2-deepseek-index-bound-final`
- 代码版本：baseline `f8a1be898b9a67d9684a409b73bf8e9889d325de` / candidate `30d108d604f22e4db9588613614848b996917618`（不同）
- 代码状态：不同或不可证明；fingerprint 不一致/缺失
- 受控变量：index-bound structured artifacts + metadata/forecast routing + adaptive candidate budget + deterministic financial reconciliation + scorer/runner audit fixes; DeepSeek output is stochastic
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.9545 | 1.0000 | +0.0455 |
| Expected behavior accuracy | 0.9655 | 1.0000 | +0.0345 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（1）

- `yili_2025_plan_bounded`

## 行为回归（0）

- 无

## Strict 修复（1）

- `yili_consolidated_parent_revenue`

## Strict 回归（0）

- 无
