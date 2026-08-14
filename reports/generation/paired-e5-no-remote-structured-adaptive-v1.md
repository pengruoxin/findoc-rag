# 生成评测配对对比：retrieved_context

- Dataset：`benchmark-v2`
- Baseline：`retrieved_context-benchmark-v2-no-remote-e5-migration-v1`
- Candidate：`retrieved_context-benchmark-v2-no-remote-structured-adaptive-v1`
- 代码版本：baseline `30d108d604f22e4db9588613614848b996917618` / candidate `30d108d604f22e4db9588613614848b996917618`（一致）
- 代码状态：不同或不可证明；fingerprint 不一致/缺失
- 受控变量：structured-table evidence routing + scope-adaptive candidate budget + forecast-year/scorer fixes
- 配对样本：48

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| Strict success | 0.6000 | 0.7429 | +0.1429 |
| Expected behavior accuracy | 0.5000 | 0.5833 | +0.0833 |
| Run error rate | 0.0000 | 0.0000 | +0.0000 |

## 行为修复（4）

- `moutai_product_margin`
- `revenue_cross_company`
- `yili_quarterly_cashflow_reconcile`
- `yili_quarterly_profit_reconcile`

## 行为回归（0）

- 无

## Strict 修复（5）

- `moutai_product_margin`
- `revenue_cross_company`
- `yili_consolidated_parent_revenue`
- `yili_quarterly_cashflow_reconcile`
- `yili_quarterly_profit_reconcile`

## Strict 回归（0）

- 无
