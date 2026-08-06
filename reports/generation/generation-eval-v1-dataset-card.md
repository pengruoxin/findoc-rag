# FinDocRAG 生成评测集 v1 数据卡

## 定位与结论边界

这是面向贵州茅台、伊利股份 2024 年报的中文复杂财务文档 RAG 冻结回归集，当前 Dataset ID 为 `generation-eval-v1-b7f4d6113c96`。

它已经具备非 toy 回归集需要的原子事实、精确证据、行为标签、受控干扰和三轨 runner，但仍是 `assistant_curated_provisional`，不是独立人工 gold，也不是跨公司泛化 benchmark。旧 32 条数据集的 DeepSeek 分数不得用于描述当前版本。

## 规模与分层

| 项目 | 数量 |
|---|---:|
| 总问题 | 48 |
| 可回答 | 37 |
| 应拒答 | 9 |
| 应澄清 | 2 |
| 问题 family | 40 |
| 原子事实 | 120 |
| Gold evidence span | 67 |
| 唯一 gold chunk | 35 |
| 多 evidence-span 问题 | 25 |
| 多独立 gold chunk 问题 | 13 |
| 派生/计算事实 | 13 |
| 带受控干扰的问题 | 29 |
| Hard negatives | 53 |
| PDF 视觉复核 | 67/67，warning=0 |

Split 固定为 calibration 12、dev 24、frozen_test 12。同一 `family_id` 和同一 gold chunk 不跨 split；冻结集 12/12 均带受控干扰上下文。这里的“多 evidence-span”可能来自同一 chunk，真正跨独立上下文的题目是 13 道，不能混用两个口径。

题型分布：single fact 2、multi-fact/table 11、narrative 8、comparison 8、calculation 7、accounting policy 1、拒答/澄清 11。难度分布为 easy 6、medium 14、hard 28。

公司分布：贵州茅台单公司 18、伊利股份单公司 22、跨公司 8。

## 深度设计

评测项不只测试“查到一个数字”，还覆盖：

- 归母净利润、扣非归母净利润与非经常性损益的三方勾稽。
- 四个季度加总与年度披露核对，专门保留第四季度负号。
- 主营业务、其他业务、营业成本合计，以及合并/母公司报表口径差异。
- 分红“报告期内已实施方案”与“下一年度拟派发预案”的状态和时点区别。
- 跨公司比较、多文档计算、关键审计事项和标准审计意见边界。
- 错公司、错期间、错口径、部分证据和不支持因果关系下的拒答/澄清。

Hard negative 分布：wrong scope 14、wrong company 15、wrong period 5、partial evidence 15、unsupported causality 4。它们全部来自当前两份真实年报 chunk，不是人工编造的错误段落。

## 每条样本的数据契约

- 稳定公司 ID、别名、年份、题型、难度、family 和 split。
- `answerable / unanswerable / needs_clarification` 行为标签及 answer contract。
- 参考答案与所需引用上下文数量。
- 原子事实的主体、谓词、Decimal 字符串值、单位、币种、期间、披露口径、容差和派生公式。
- Gold chunk、document version、完整页码范围、section path、原文 quote 与 fact-evidence 绑定。
- 每个干扰 chunk 的类型和为什么不能支持目标答案。
- 标注来源、复核状态、PDF SHA-256 与视觉复核状态。

## 三条评测轨道

1. `oracle_context`：直接提供 gold evidence，隔离测试生成和回答策略上限。
2. `retrieved_context`：经过真实 BM25 + Dense + weighted RRF、metadata routing 与 guardrail，测试端到端 RAG。
3. `robustness`：在 gold 之间插入受控高相似干扰；拒答题也会获得诱导性上下文，不再用“空上下文即拒答”刷分。

生成器上下文预算已由 3 扩为 5，能够完整容纳四组关键审计事项证据；运行记录保存每个上下文的 `gold / retrieved / hard_negative:<type>` 标签。

## 自动质量门禁

Validator 当前强制检查：

- item/fact 唯一性、gold chunk 与 hard-negative chunk 存在性。
- hard negative 不得与同题 gold 重复。
- quote 必须是对应 chunk 的精确子串，页码范围必须一致。
- 每个直接数值事实必须实际出现在绑定 evidence span 中。
- 参考答案必须覆盖全部 required 数值/布尔事实。
- 每个 required fact 必须绑定证据，证据不得引用不存在的 fact。
- `required_citation_count` 必须等于唯一 gold context 数，而不是 quote span 数。
- 同一 family 和 gold chunk 不得跨 split。
- 跨页 evidence 的所有 PDF 页都必须进入视觉复核清单。

这组门禁修复了旧版中“schema valid 但 quote 实际不包含数值”“同一 chunk 两个 quote 却要求两个引用”“参考答案只覆盖 2/5 required facts”等问题。

## 当前可复现基线

下面是无 LLM 确定性基线，用来验证 runner、行为门禁和配对比较，不代表 DeepSeek 的生成质量：

| Lane | 版本 | Strict success | 行为准确率 | Error rate |
|---|---|---:|---:|---:|
| Oracle | 修改前 | 0.2571 | 0.9583 | 0 |
| Oracle | clarify-v1 | 0.3143 | 1.0000 | 0 |
| Retrieved | 修改前 | 0.0286 | 0.7917 | 0 |
| Retrieved | clarify-v1 | 0.0857 | 0.8333 | 0 |
| Robustness（29 条） | 修改前 | 0.1364 | 0.7241 | 0 |
| Robustness（29 条） | clarify-v1 | 0.2273 | 0.7931 | 0 |

`clarify-v1` 修复了两条模糊口径问题，三个 lane 均为 2 条行为修复、0 条行为回归。配对报告位于 `reports/generation/comparisons/`。

当前目录中的 DeepSeek Oracle 结果属于旧 Dataset ID `generation-eval-v1-82c3e242b307`（32 条），只能作为历史开发记录。需要在最终 Dataset ID 上重新运行 Oracle、Retrieved、Robustness 和 RAGAS 后，才能给出新版模型结论。

## 已知限制与下一版计划

- 只有两家公司、一个年份、两份原生 PDF，document-level blind test 为 0；不能宣称跨公司、跨年度或 OCR 泛化。
- frozen_test 只有 12 条，虽然全部带干扰，但任务分层仍偏向比较和审计，置信区间较宽。
- 目前全部为 assistant-curated；至少应对 frozen_test、表格、计算和拒答题做第二独立复核后再升级为 `human_frozen`。
- 当前 tolerance/单位换算策略仍需覆盖万元、亿元和合理舍入答案。
- DeepSeek 同时作为 answer model 和 judge 时必须标记 `independent_judge=false`，RAGAS 不能替代确定性数值、单位和引用门禁。
- 后续应新增其他公司、其他年度、扫描/OCR、跨页表格与整份文档 holdout，并把 frozen_test 扩到至少 24 条。
