# Agent hard-v3 P2-A：核验答案契约与跨年度引用

## 结论

P2-A 完成。Hard-v3 校准集与开发集共 4 道 `claim_verification` 题在最终代码上重跑，组合后的 48 题严格通过率由 34/48（70.83%）提升到 36/48（75.00%）。

本阶段没有修改 Gold、评分器、检索索引和 frozen test。只替换 4 条核验题轨迹，其余 44 条 P1 轨迹保持不变。

## 改动

核验任务现在携带结构化 `answer_contract`：

- 公司和年度 target 只限定证据范围，不再被误当成任务子项清单；
- 必须逐项覆盖原问题中的指标和方向；
- 优先明确输出“说法成立”或“说法不成立”；
- 如果没有固定结论短语，本地门禁至少要求完整复述原说法中的全部方向谓词；
- 同比下降同时输出文字方向和带符号数值，例如“下降 9.70%（-9.70%）”。

同时修正跨年度比较的引用归属：同一公司的比较 claim 可以引用另一个已规划年度的证据，但只有在 claim 明确写出该证据年度并包含比较谓词时才允许。没有写明年度的跨 target 引用仍被拒绝。

## 为什么需要本地门禁

第一次校准运行达到 2/2，但同一提示词复跑时出现 1/2，失败原因不是事实缺失，而是模型把 2023 年证据挂到 2024 target 后被引用门禁拒绝两次。只保留最好的一次会形成挑结果偏差，因此没有采用。

开发集第一次运行为 0/2：一个答案事实完整但引用页不覆盖 Gold；另一个因为结论措辞不在过窄白名单内，被本地门禁重复拒绝。后续将门禁改为“明确结论或完整方向谓词”，并增加有边界的跨年度引用支持。最终代码重新运行后，校准 2/2，开发 1/2，原来已通过的开发回归题没有退化。

## 指标

| 数据集 | P1 严格通过 | P2-A 严格通过 | 事实准确率变化 |
|---|---:|---:|---:|
| calibration | 18/24（75.00%） | 20/24（83.33%） | 74.04% → 75.96% |
| dev | 16/24（66.67%） | 16/24（66.67%） | 86.79% → 87.74% |
| 合计 | 34/48（70.83%） | 36/48（75.00%） | +2 题 / +4.17 个百分点 |

最终 4 题运行成本：

- DeepSeek 请求：8 次
- 输入 token：19,389
- 输出 token：2,853
- 合计 token：22,242
- 检索工具调用：8 次

所有校准、负实验与最终确认合计消耗 87,913 token，均记录在机器可读报告中，未只记录最好结果。

## 剩余边界

`v3_601318_verify_nbv` 已由 4/5 提升为 5/5 事实命中，结论也正确，但引用页没有与 Gold 指定页重叠，因此严格整题仍失败。下一阶段应处理“同一事实多个候选页时优先选择权威 Gold 类页面”的引用来源排序，而不是放宽引用评分。

其余失败还包括真正的跨年度多指标检索不足、集团与业务分部口径混淆，以及长审计事项漏项。它们没有被本阶段掩盖。

## 产物

- 增量汇总：`reports/agent/agent-hard-v3-p2a-answer-contract-improvement.json`
- 校准最终专项：`reports/agent/agent-hard-v3-calibration-deepseek-p2a-verdict-contract-v3.json`
- 开发最终专项：`reports/agent/agent-hard-v3-dev-deepseek-p2a-verdict-contract-v3.json`
- 校准组合：`reports/agent/agent-hard-v3-calibration-deepseek-p2a-composed.json`
- 开发组合：`reports/agent/agent-hard-v3-dev-deepseek-p2a-composed.json`
- 可复现子集：`data/evaluation/agent-hard-v3-calibration-verification.json`、`data/evaluation/agent-hard-v3-dev-verification.json`
