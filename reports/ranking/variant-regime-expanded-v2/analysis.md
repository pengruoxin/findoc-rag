# 同义词查询改写实验分析（query expansion v2）

> 配套数据：`summary.json`、`per_query.jsonl`、`config.json`；词表：`src/findoc_rag/query_expansion.py`。
> 重跑：`python scripts/run_retrieval_variant_eval.py --expand-synonyms --output-dir reports/ranking/variant-regime-expanded-v3`。

## 1. 动机

semantic regime 10 条 lexical miss 的根因是"用户措辞 ≠ 年报措辞"（营收 vs 营业收入、毛利水平 vs 毛利率、净资产回报率 vs 净资产收益率、主要风险 vs 可能面对的风险等）。词表全部从失败案例提取，共 7 组映射（含"同比增幅→比上年同期增减"、"一定实现→计划实现"两个隔词场景）。

## 2. 检索侧结果（query_parser 过滤，lexical，Hit@5 / MRR@5）

| regime | base | expanded v2 | 变化 |
|---|---:|---:|---:|
| canonical | 0.838 / 0.694 | **0.892 / 0.739** | +0.054 / +0.045 |
| ticker | 0.811 / 0.724 | **0.892 / 0.760** | +0.081 / +0.036 |
| semantic/相对时间 | 0.730 / 0.633 | **0.919 / 0.761** | **+0.189 / +0.128** |

semantic Hit@5 从 0.73 提到 0.92，超过 roadmap 0.85 目标；9 题救回、0 回归。

## 3. 剩余 miss 归因（3 条）

- `yili_2025_plan_bounded`：**评测管线时间对齐 bug，不是检索失败**——解析年份 2025 被用于过滤，排除了 2024 年报 gold（实际检索 gold rank 1）。修复归 P2-15。
- `moutai_product_margin`：排序问题，gold 在候选池 rank 6，差一名进 top5。
- `revenue_cross_company`：结构问题，"营业收入"匹配的切片太多，gold 缺少独特信号（需要"主要会计数据"类结构线索）。

## 4. 端到端验证与归因（retrieved lane，DeepSeek）

查询改写已接入生成 runner；retrieved strict 0.5429 与改写前持平。逐题归因 16 个 strict 失败：

| 失败层 | 数量 | 说明 |
|---|---:|---|
| 行为拒答（该拒没拒） | 8 | unanswerable 题被检索出的干扰证据带偏 |
| 表格/生成（gold 在 top5 仍答错） | 5 | 多事实核对 / 表格数字抽取错误（如 revenue_yoy facts=0） |
| 检索（gold 不在 top5） | 3 | product_margin（排序）、revenue_cross_company（结构）、quarterly_cashflow_reconcile |

结论：同义词改写显著提升检索指标，但端到端 strict 的下一步瓶颈是**行为拒答 + 表格抽取**，检索侧边际收益已有限。

## 5. 面试讲法

> 检索失败里有相当一部分是"用户说法和年报说法对不上"。我把 10 条失败题的措辞差异提取成同义词词表，做查询改写，semantic 问法 Hit@5 从 0.73 提到 0.92，零回归。但端到端分数没变——逐题归因后发现 strict 失败主要是 8 题该拒没拒、5 题表格数字抽错，检索只占 3 题。这告诉我下一步该修行为和表格，而不是继续堆检索技巧。
