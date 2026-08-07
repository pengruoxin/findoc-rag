# Variant-regime v1 分析与失败归因

> 配套数据：`summary.json`（指标）、`per_query.jsonl`（逐题）、`config.json`（复现配置）。
> 重跑：`python scripts/run_retrieval_variant_eval.py --output-dir reports/ranking/variant-regime-v1`（输出目录已存在，需换新目录或先归档）。

## 1. 关键发现

### 1.1 query_parser 过滤是 lexical 的最大杠杆

canonical regime（37 题）：Hit@5 0.757 → 0.838，MRR@5 0.559 → 0.694。公司 + 年份过滤直接消除跨公司、跨年份噪声，与 holdout v2 观察一致。

### 1.2 dense 只在 ticker regime 有价值

query_parser 过滤下 dense Hit@5：canonical 0.216、semantic 0.162、**ticker 0.676**。ticker 查询更短、指代唯一（600519 / 600887），dense 编码明显更稳定；长查询 + 语义改写时 dense 快速退化。

### 1.3 semantic / 相对时间 regime 是全员弱区

query_parser 过滤下 Hit@5：lexical 0.730、dense 0.162、hybrid 0.459。37 条中有 11 条三路全 miss（`ldh`），主因是**专业同义词改写**：`毛利水平` vs 证据中的 `毛利率`、`净资产回报率` vs `加权平均净资产收益率`、`营收` vs `营业收入`——BM25 的二元组与证据不重叠，dense 也没能跨过。

### 1.4 hybrid 被弱 dense 分支拖累（量化证据）

4 条 semantic 题在 query_parser 过滤下 hybrid 从 hit 变 miss，逐题根因一致：

| 题 | lexical gold rank（过滤） | hybrid gold rank（过滤） |
|---|---:|---:|
| yili_quarterly_net_profit | 1 | 7（池内，top5 外） |
| moutai_annual_deducted_profit | 2 | 6（池内，top5 外） |
| yili_quarterly_profit_reconcile | 1 | 6（池内，top5 外） |
| moutai_cost_components | 1 | 6（池内，top5 外） |

这 4 条 dense 全部 `candidate_recall=False`（gold 不在 dense top-20），dense 只贡献噪声，RRF 把 lexical rank 1–2 的 gold 推到 6–7。结论：当前 E5-small 分支对语义 regime 是负资产，不应无差别融合。

### 1.5 时间对齐 bug 实锤（P2-15 的评测证据）

`yili_2025_plan_bounded` 的相对时间变体 `as_of=2026-04-30`，"去年"解析为 2025（解析正确），但 query_parser 用 `report_years=[2025]` 过滤，把 gold（2024 年报中的 2025 经营计划，`report_year=2024`）排除。**解析年份 ≠ 语料报告年份时不能直接用于 metadata 过滤**，需要时间对齐分支（语料年份 ∩ 解析年份；缺失时降级 + 时间边界标记）。

### 1.6 相对时间解析 43 条全部正确

43 条 relative-time 变体全部解析出 2024（唯一例外 `yili_2025_plan_bounded` 是 as_of=2026 的设计行为，指向 2025，属预期）。

### 1.7 行为组（拒答 / 澄清）检索污染

33 条行为实例中，lexical / hybrid 有 11 条 top5 出现 hard negative，dense 3 条。检索出干扰项是行为评测的预期输入，后续行为指标（拒答正确率、澄清触发率）以此为上下文进行评测。

## 2. 建议的下一步（按优先级）

1. **RRF 权重 / 按 regime 动态选路**：sweep lexical:dense（3:1 / 4:1 / dense 仅 ticker 启用），目标消除 §1.4 的稀释；用同一脚本重跑得到配对对比
2. **专业同义词增强**：对 §1.3 的 11 条 ldh 题建立同义词映射（毛利水平→毛利率、净资产回报率→净资产收益率、营收→营业收入），评估 query 改写或 tokenizer 增强的收益
3. **P2-15 时间对齐落地**：query_parser 输出 (公司, 年份, 语料覆盖检查)，解析年份不在语料时降级过滤 + 时间边界标记
4. **dense 模型对照**（低优先级）：E5-small 明显弱，可对照 bge-m3；但 lexical 已强，dense 升级前先证明它在 semantic regime 能超过 lexical

## 3. 结论

这份评测第一次把"专业表达鲁棒性"变成了可量化指标：系统对股票代码问法（dense 强）、照原文问法（lexical 强）、口语/相对时间问法（全员弱）的能力差异清晰可见。优化优先级应当从"提升 dense 模型"转向"同义词鲁棒性 + 融合策略 + 时间对齐"。
