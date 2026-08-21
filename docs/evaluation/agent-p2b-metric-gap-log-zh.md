# Agent hard-v3 P2-B：多指标缺口驱动检索

## 结论

P2-B 完成。8 道多年度比较题在最终代码上校准集 4/4、开发集 4/4 严格通过；组合后的 48 题严格通过率由 36/48（75.00%）提升到 41/48（85.42%），增加 5 题、10.42 个百分点。

本阶段未修改 Gold、评分器、索引和 frozen test。只替换 8 条 `compare + multi_year` 轨迹，其余 40 条 P2-A 轨迹原样复用。

## 根因

旧控制器只按公司和年度判断证据充分性。问题即使要求四个指标，只要每个年度拿到一个 chunk，就可能被判为 `sufficient`。同时检索提示通常只包含问题里的第一个指标，导致：

- 工商银行跨年核心指标比较：0/9；
- 工商银行资产质量比较：0/8；
- 中国平安归母营运利润、归母净利润和新业务价值比较：3/6 后拒答。

这不是 OCR 缺失，而是 Agent 没有把“问题子项”表示成可检查的证据缺口。

## 改动

1. 从问题中确定性抽取非重叠指标清单，处理“归母净利润/净利润”“研发投入金额/研发投入强度”等重叠词。
2. 每个年度的检索提示包含全部指标，并按财务摘要、资产质量、寿险经营指标、客户指标和分产品表增加通用章节词。
3. `judge_sufficiency` 除了统计 target 是否有 chunk，还逐 target 检查每个指标是否进入 Evidence Memory。
4. 工具结果显式返回 `remaining_required_metrics`；非空时 DeepSeek 只针对缺失指标再次检索。
5. Evidence Memory 摘要由 800 字扩到 1,600 字，避免长表后半部分的指标名称被截断。
6. 多指标比较最后输出独立的简明方向总结，避免“完整指标名”和方向词被长数值隔开。
7. 如果模型把明确写着 2023 年且只引用 2023 证据的 claim 错标为 2024 target，本地只在候选唯一时修正；未写年度或跨公司情况不猜测。

## 缺口驱动实例

中国平安任务首轮检索命中新业务价值页面，但本地充分性检查仍返回：

- `归母营运利润`
- `归母净利润`

第二轮因此只检索这两个指标，工具调用由普通比较的 2 次增加到 4 次，最终从 3/6 提升到 6/6。没有对已经覆盖的新业务价值重复搜索。

## 评测结果

| 数据集 | P2-A 严格通过 | P2-B 严格通过 | 事实准确率 | 行为准确率 |
|---|---:|---:|---:|---:|
| calibration | 20/24（83.33%） | 22/24（91.67%） | 75.96% → 92.31% | 91.67% → 100% |
| dev | 16/24（66.67%） | 19/24（79.17%） | 87.74% → 92.45% | 95.83% → 100% |
| 合计 | 36/48（75.00%） | 41/48（85.42%） | — | — |

最终 8 题运行成本：

- DeepSeek 请求：21 次
- 输入 token：71,479
- 输出 token：13,346
- 合计 token：84,825
- 检索工具调用：18 次

全部校准、负实验和开发确认共消耗 159,231 token。两次校准负实验都保留在汇总报告中。

## 负实验

- calibration v1：检索已明显改善，事实准确率 93.33%，但缺少独立方向总结，2/4 严格通过。
- calibration v2：只加强提示后出现模型 target 归属波动，事实准确率降至 20%，1/4 通过；没有采用这次结果。
- calibration v3：增加有边界的明确年度 target 修复后，4/4 严格通过。
- 未继续调整代码，直接运行 dev，4/4 严格通过。

## 剩余问题

当前还剩 7/48 严格失败，已不再有多年度比较失败。主要集中在：

- 两道关键审计事项长答案漏项；
- 中国平安集团指标被寿险业务分部指标替换；
- 宁德时代分产品同比下降的同义表达；
- 两道事实正确但引用页未命中权威 Gold 页面的引用来源排序。

下一阶段应处理抽取任务的多事实清单和权威来源页排序，不应继续扩大比较检索器。

## 产物

- 增量汇总：`reports/agent/agent-hard-v3-p2b-metric-gap-improvement.json`
- 校准最终专项：`reports/agent/agent-hard-v3-calibration-deepseek-p2b-metric-gap-retrieval-v3.json`
- 开发专项：`reports/agent/agent-hard-v3-dev-deepseek-p2b-metric-gap-retrieval.json`
- 校准组合：`reports/agent/agent-hard-v3-calibration-deepseek-p2b-composed.json`
- 开发组合：`reports/agent/agent-hard-v3-dev-deepseek-p2b-composed.json`
- 可复现子集：`data/evaluation/agent-hard-v3-calibration-multi-metric-compare.json`、`data/evaluation/agent-hard-v3-dev-multi-metric-compare.json`
