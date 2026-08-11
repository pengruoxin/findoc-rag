# 变体检索评测：bge-small-zh-v1.5 dense 对照

- 数据集：`benchmark-v2-retrieval-view`（111 positive instances / 37 canonical groups）
- 索引：`6a951f4e8b7bd913d918`（bge-small-zh-v1.5，512 维，958 chunks）
- 配置：top_k=5 | candidate_k=20 | rrf_k=60 | 无查询改写 | query_parser 过滤

## 与 E5 的对比（dense 路）

| regime | E5 dense Hit@5 / MRR@5 | bge-zh dense Hit@5 / MRR@5 |
|---|---:|---:|
| canonical | 0.2162 / 0.1631 | 0.1622 / 0.1306 |
| ticker_or_finance_shorthand | 0.7027 / 0.5000 | 0.5676 / 0.3495 |
| semantic_or_relative_time | 0.1892 / 0.1261 | 0.1622 / 0.1306 |

## 融合路

| regime | E5 hybrid Hit@5 / MRR@5 | bge-zh hybrid Hit@5 / MRR@5 |
|---|---:|---:|
| canonical | 0.6757 / 0.3617 | 0.6757 / 0.3617 |
| ticker_or_finance_shorthand | 0.7838 / 0.6937 | 0.7568 / 0.5838 |
| semantic_or_relative_time | 0.5676 / 0.3667 | 0.4865 / 0.2743 |

## 结论

1. **bge-small-zh-v1.5 在三种问法上都弱于 E5**，唯一对 dense 有利的 ticker 问法也从 0.70 掉到 0.57。中文专用小模型不是当前 dense 失败的原因。
2. hybrid 在 canonical 上持平（lexical 主导），在 ticker / semantic 上继续被弱 dense 拖累；lexical-only 依旧是最优默认。
3. 与 `oov-eval-bge-zh-v1`（OOV dense 0.083 < E5 0.139）一致：在问句形态与表格线性化问题解决前，换 dense 模型没有收益。

## 边界

本组数字用于回答"要不要换更强 dense 模型"，不是模型间全面 benchmark；bge-m3 等更大模型未测，待表格结构化（B 阶段）后与查询改写一起重验。
