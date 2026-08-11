# OOV 词表外改写评测：LLM 查询改写 + 中文 dense 对照

- 数据集：`oov-variants-v1`（36 个词表外改写实例 / 12 题）
- 索引：E5 `10fb50419145d56720c9` vs bge-small-zh-v1.5 `6a951f4e8b7bd913d918`
- 配置：top_k=5 | candidate_k=20 | rrf_k=60 | weights 2:1
- 指标为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）

## 1. 三种改写方式的收益（E5 索引，query_parser 过滤）

| rewrite | lexical Hit@5 / MRR@5 | dense Hit@5 | hybrid Hit@5 |
|---|---:|---:|---:|
| none（基线） | 0.194 / 0.148 | 0.139 | 0.222 |
| deterministic（7 组词表） | 0.194 / 0.148 | 0.139 | 0.222 |
| **LLM** | **0.694 / 0.498** | 0.167 | 0.472 |

结论：

- 词表外改写对确定性词表完全免疫（0.194 → 0.194），OOV 集达到了设计目的。
- **LLM 查询改写是决定性杠杆**：lexical Hit@5 +0.50（0.194 → 0.694），MRR +0.35；候选召回 0.333 → 0.861。
- LLM 改写后融合仍然是负优化：hybrid 0.472 < lexical 0.694，dense 只有 0.167。dense 分支继续拖累排序，与 fusion-sweep-v1 结论一致。

## 2. 中文 dense 对照：bge-small-zh-v1.5 是否解决了同义词

同一 OOV 集、同一查询、不同 dense 模型：

| 索引 | dense Hit@5（none） | dense Hit@5（LLM 改写后） | hybrid Hit@5（LLM 改写后） |
|---|---:|---:|---:|
| E5-small（multilingual） | 0.139 | 0.167 | 0.472 |
| **bge-small-zh-v1.5** | **0.083** | **0.083** | **0.444** |

变体矩阵（111 问句，query_parser 过滤，无改写）：

| regime | E5 dense Hit@5 / MRR@5 | bge-zh dense Hit@5 / MRR@5 |
|---|---:|---:|
| canonical（原题） | 0.216 / 0.163 | 0.162 / 0.131 |
| ticker / 简称 | 0.703 / 0.500 | 0.568 / 0.350 |
| semantic / 相对时间 | 0.189 / 0.126 | 0.162 / 0.131 |

结论：**换中文专用小模型没有帮助，反而全面退化**。bge 在三种问法上都低于 E5；"股票代码/简称"这个唯一对 dense 有利的形态也从 0.703 掉到 0.568。这证明当前 dense 失败的主因不是"中文/多语言模型能力"，而是问句形态和上游表格线性化——即使 LLM 把问句归一为年报措辞，dense 仍只有 0.08–0.17。

## 3. 剩余 11 个 miss 的失败归因

LLM 改写后 lexical 25/36 命中，11 个 miss 分三类：

| 类型 | 实例 | 根因 |
|---|---|---|
| 财务简称未归一（4） | `moutai_annual_deducted_profit` oov1–3（扣非净利润→扣除非经常性损益后的净利润）、`moutai_revenue_yoy` oov2–3（同比变化→比上年同期增减） | LLM 改写不包含文档措辞知识；改写结果没有叠加确定性词表兜底 |
| 行内换行断裂（2） | `moutai_product_margin` oov1、oov3（其他系列 酒 中间被 PDF 换行拆开） | 上游表格线性化把"其他系列酒"切成两个 text 块，bigram 断裂——B 阶段表格重建才能根治 |
| 文档措辞未知（3） | `moutai_disclosed_risks` oov1–3（风险因素/未来风险与挑战→可能面对的风险） | 查询侧 LLM 不知道新文档用什么章节标题；需要文档侧 ingest 归一化（term-normalization-design 三层架构第 2 层） |
| 语义改写偏差（2） | `moutai_product_margin` oov1 改写为"茅台酒和其他系列酒"、`yili_product_margin` oov3 改写为"营业收入及净利润" | LLM 改写引入了原问题没有的指标，检索方向被带偏 |

## 4. 可复现性风险：LLM 改写跨 run 不稳定

同一次实验（temperature=0）连续跑两遍，36 条改写中 8 条输出不同，其中 1 条改变了命中结果（Hit@5 0.694 vs 0.667）。当前 `LLMQueryRewriter` 只在进程内缓存，评测报告的 `resolved_query` 已落盘可审计，但**重跑无法复现同一组改写**。

下一步：把改写缓存持久化到 run 目录（`rewrites.jsonl`），重跑时优先复用；LLM 改写后再叠加确定性词表兜底（如"扣非→扣除非经常性损益"），并给改写结果加"改写前/后"对比评测。

## 5. 结论与边界

- 默认策略维持 lexical-only 不变；语义融合在 OOV 上同样无收益。
- LLM 查询改写应进入生产链路（`/v1/query`），但需要：持久化改写缓存、确定性兜底、改写质量门禁。
- 本次对比只覆盖两家公司 2024 年报的 12 个 OOV 考点（36 实例），且改写由 DeepSeek 生成、未经人工审核；数字用于工程迭代，不对外主张泛化能力。
