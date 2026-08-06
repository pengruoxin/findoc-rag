# 生成评测配对对比：robustness

- Dataset：`generation-eval-v1-b7f4d6113c96`
- Baseline：`robustness-generation-eval-v1-b7f4d6113c96-no-llm-baseline`
- Candidate：`robustness-generation-eval-v1-b7f4d6113c96-no-llm-clarify-v1`
- 配对样本：29

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.1364 | 0.2273 | +0.0909 |
| Expected behavior accuracy | 0.7241 | 0.7931 | +0.0690 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（2）

- `u_moutai_profit_ambiguous`
- `u_yili_cost_scope_ambiguous`

## 行为回归（0）

- 无

## Strict 修复（2）

- `u_moutai_profit_ambiguous`
- `u_yili_cost_scope_ambiguous`

## Strict 回归（0）

- 无
