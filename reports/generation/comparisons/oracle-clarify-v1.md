# 生成评测配对对比：oracle_context

- Dataset：`generation-eval-v1-b7f4d6113c96`
- Baseline：`oracle_context-generation-eval-v1-b7f4d6113c96-no-llm-baseline`
- Candidate：`oracle_context-generation-eval-v1-b7f4d6113c96-no-llm-clarify-v1`
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.2571 | 0.3143 | +0.0571 |
| Expected behavior accuracy | 0.9583 | 1.0000 | +0.0417 |
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
