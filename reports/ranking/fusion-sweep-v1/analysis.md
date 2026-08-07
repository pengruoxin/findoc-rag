# RRF 融合权重 sweep v1 分析

> 配套数据：`summary.json`、`summary.md`、`config.json`。
> 重跑：`python scripts/run_retrieval_fusion_sweep.py --output-dir reports/ranking/fusion-sweep-v2`。

## 1. 实验设置

同一批组件排名（lexical / dense 各算一次）离线融合 6 组权重：1:1、2:1（v1 基线）、3:1、4:1、1:0（纯 lexical）、0:1（纯 dense）。111 个 positive instances，top-5，candidate-20，rrf_k=60。

## 2. 结果（query_parser 过滤，Hit@5 / MRR@5）

| weight | canonical | ticker | semantic/相对时间 |
|---|---:|---:|---:|
| 1:1 | 0.595 / 0.357 | 0.730 / 0.640 | 0.432 / 0.306 |
| 2:1（v1） | 0.649 / 0.365 | 0.730 / 0.671 | 0.459 / 0.321 |
| 3:1 | 0.649 / 0.397 | 0.730 / 0.676 | 0.459 / 0.339 |
| 4:1 | 0.703 / 0.417 | 0.730 / 0.689 | 0.514 / 0.350 |
| **1:0（纯 lexical）** | **0.838 / 0.694** | **0.811 / 0.724** | **0.730 / 0.633** |
| 0:1（纯 dense） | 0.216 / 0.146 | 0.676 / 0.469 | 0.162 / 0.113 |

## 3. 结论

1. **当前 E5-small dense 分支在全部三个 regime 都是负资产**：任何正的 dense 权重都让 Hit@5 / MRR@5 低于纯 lexical。权重越向 lexical 倾斜越好，外推极限就是 1:0。
2. **即使在 dense 最强的 ticker regime，lexical 仍然更高**（0.811 vs dense 0.676；最好的融合 4:1 也只有 0.730）。所谓"dense 在 ticker 有效"只是相对它自己，不是相对 lexical。
3. **per-regime best 三个 regime 全部选择 1:0**——按 regime 动态选路的"上限"在当前模型下就是全用 lexical，动态选路没有增益。
4. 无过滤态结论一致：lexical-only 在 canonical（0.757）和 semantic（0.730）最高，ticker 与 2:1–4:1 持平（0.784）。

## 4. 影响与下一步

- **默认检索策略应改为 lexical-only**（当前 CLI / 服务默认 hybrid 2:1）。这是产品代码改动，建议单独验证后落地。
- **dense 升级的重新验证协议**：换模型（bge-m3 / acge 等）后用同一 sweep 重跑；只有存在某个权重组合在某个 regime 显著超过 lexical-only 时，才值得重新启用融合。
- **同义词增强仍是最重要的下一项**：semantic regime 的 lexical-only 0.730 是当前系统上限，11 条 ldh 题的根因在查询-证据词汇不重叠，融合解决不了。

> 边界：per-regime best 在评测集上选择，属于 development-only；默认策略改动需在 holdout / 生成三轨上验证后再定论。

## 5. dense 弱的归因与决策（2026-08-07 补充）

补充检查（dense top-5 段落集中度，111 个 positive instances，query_parser 过滤）：

- **dense 最常召回的段落是"关联交易–存款业务"表格**（26 次进 top-5），与大多数问题无关；BM25 最常召回的是"主营业务分行业/分产品"表格（正确答案所在）。
- 判断：E5-small 被"数字密集 + 财务表格外观"的表面相似度误导，无法区分"这个表格在回答哪个指标"。
- **上游原因**：PDF 表格线性化（数字与标签分离、行列关系丢失）让 dense 的输入以无结构、数字密集的文本为主，放大了表面误导——信息提取层尚未结构化，是 dense 弱的独立上游因素。

**决策**：当前暂用 BM25（已落地为默认）；B 阶段（表格结构化）完成后，与更强 dense 模型（bge-m3 等）一起用同一 sweep 重新验证融合。两个因素（模型能力 / 信息提取质量）届时可以分离归因。
