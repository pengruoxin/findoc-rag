# Agent Hard-v3 P1：语料年份覆盖拒答

## 结论

P1 将“当前索引没有所请求年度的实际全年事实”从澄清改为可解释拒答。校准集和开发集的 4 道缺失年份题全部通过，安全拒答准确率由 0% 提升到 100%。

这次判断不使用系统当前年份。它只比较问题要求的事实年度与该公司在当前索引中的最高年报年度；因此即使运行时间是 2026 年，也不会据此断言 2025 年事实存在或不存在。

## 时间语义约束

- 用户独立说“去年/今年”：相对请求显式提供的 `as_of_date` 解析。评测不得读取运行时系统时钟。
- 用户引用“2022 年年报里的去年/上年”：相对文档报告年度解析，即 2021 年。
- 年报正文里的“去年/上年/本年”：DeepSeek 证据载荷携带 `report_year`，提示词明确要求相对该字段解释，不得相对系统当前年份。
- “上期”暂不机械映射为上一年，因为它也可能表示上季度或上一报告期；没有明确表头日期时应保留歧义并拒绝臆断。

## 拒答规则

当问题只要求一个事实年度，且该年度晚于公司在索引中的最高年报年度，同时问题没有明确指定另一份年报、也没有“预计/预测/目标/指引/展望”等前瞻性词语时，本地覆盖门直接返回 `abstain`：

- `stop_reason = no_new_evidence`
- `grounded = false`
- 引用为空，不伪造页码
- DeepSeek 请求数为 0
- 检索工具调用数为 0

为避免过度拒答，前瞻性问题会选择索引中最新年报继续检索；缺失的历史报告年度也不会被这个门直接拒答，因为较新年报可能包含历史比较数据。

## 评测设计

- 只重跑 Hard-v3 校准集与开发集中的 4 道 `expected_behavior=abstain` 题。
- 其余 44 条 P0 轨迹保持不变并组合计分。
- `frozen_test` 继续封存，未打开。
- 模型配置仍记录为 DeepSeek `deepseek-v4-flash`；本次 4 题均在模型调用前由确定性覆盖门完成。

## 结果

| 指标 | P0 | P1 | 变化 |
|---|---:|---:|---:|
| 校准集严格通过 | 16/24（66.67%） | 18/24（75.00%） | +2 题 |
| 开发集严格通过 | 14/24（58.33%） | 16/24（66.67%） | +2 题 |
| 合计严格通过 | 30/48（62.50%） | 34/48（70.83%） | +4 题 / +8.33 个百分点 |
| 安全拒答准确率 | 0% | 100% | +100 个百分点 |
| 任务覆盖率 | 100% | 100% | 不变 |
| 本次模型请求 | — | 0 | 未消耗 DeepSeek token |
| 本次工具调用 | — | 0 | 无无效检索 |

抽取事实准确率没有变化：校准集仍为 74.04%，开发集仍为 86.79%。这是预期结果，因为本阶段只修复拒答行为，没有替换回答题轨迹。

## 产物

- 汇总增量：`reports/agent/agent-hard-v3-p1-corpus-abstention-improvement.json`
- 校准专项：`reports/agent/agent-hard-v3-calibration-deepseek-p1-corpus-abstention.json`
- 开发专项：`reports/agent/agent-hard-v3-dev-deepseek-p1-corpus-abstention.json`
- 校准组合：`reports/agent/agent-hard-v3-calibration-deepseek-p1-composed.json`
- 开发组合：`reports/agent/agent-hard-v3-dev-deepseek-p1-composed.json`
- 可复现子集：`data/evaluation/agent-hard-v3-calibration-abstain.json`、`data/evaluation/agent-hard-v3-dev-abstain.json`
