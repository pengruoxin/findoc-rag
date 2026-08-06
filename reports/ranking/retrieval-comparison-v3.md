# BM25、Dense 与 RRF Hybrid 公平对比

评测条件：同一 index、同一 holdout v2（16 条）、top-k=5、candidate-k=20，并关闭 metadata filters、scope routing、adaptive candidate budget 和 reranker。

| Pipeline | Hit@5 / Recall@5 | MRR@5 | 相对 BM25 |
|---|---:|---:|---:|
| BM25 | 0.6875 | 0.4531 | baseline |
| multilingual-e5 Dense | 0.2500 | 0.0802 | Hit@5 -0.4375；MRR -0.3729 |
| BM25 + Dense + RRF | 0.6250 | 0.3917 | Hit@5 -0.0625；MRR -0.0614 |
| BM25 + Dense + 加权 RRF（2:1） | 0.6875 | 0.3948 | Hit@5 +0.0000；MRR -0.0583 |

## 结论

在当前 16 条中文年报问题上，单独 BM25 最好。RRF Hybrid 比 Dense 明显好，但仍比 BM25 低 6.25 个百分点的 Hit@5，MRR 低 0.0614。说明当前 dense 分支质量不足，RRF 将较弱的 dense 排名无差别融合后，反而稀释了 BM25 的强精确匹配信号。

这并不说明混合检索没有价值，而是说明不能默认两路等权融合。下一步应比较：调整 RRF 权重、扩大候选池、只在语义改写型问题启用 Dense，或先提升 dense embedding/reranker 后再融合。

## 加权 RRF 实验

将 BM25 与 Dense 权重调整为 2:1 后，Hit@5 从等权 RRF 的 0.6250 恢复到 0.6875（+0.0625），与单独 BM25 持平；MRR 从 0.3917 提升到 0.3948（+0.0031），但仍低于 BM25 的 0.4531。说明加权融合修复了等权融合的召回退化，但首个相关结果排序仍需优化。

原始结果：`retrieval-comparison-v3-lexical.json`、`retrieval-comparison-v3-dense.json`、`retrieval-comparison-v3-hybrid.json`。
