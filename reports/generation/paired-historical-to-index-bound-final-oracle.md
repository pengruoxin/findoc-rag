# 生成评测配对对比：oracle_context

- Dataset：`benchmark-v2`
- Baseline：`oracle_context-benchmark-v2-deepseek-chat-table-remote-v1`
- Candidate：`oracle_context-benchmark-v2-deepseek-index-bound-final`
- 代码版本：baseline `640ab99b9cbd609d8be7b24d620caccee44ebc01` / candidate `30d108d604f22e4db9588613614848b996917618`（不同）
- 代码状态：不同或不可证明；fingerprint 不一致/缺失
- 受控变量：index-bound structured artifacts + deterministic financial reconciliation + scorer/runner audit fixes; DeepSeek output is stochastic
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
