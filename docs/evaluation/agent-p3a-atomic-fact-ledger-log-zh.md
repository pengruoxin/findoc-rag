# Agent hard-v3 P3-A：证据感知的原子事实清单

## 结论

P3-A 完成。抽取 Agent 不再仅凭“已找到文档和页面”判断任务完成，而是在最终回答前由
DeepSeek 基于问题与候选证据生成原子事实清单；控制器补齐确定性指标漏项、只对无证据
清单项执行定向检索，并要求最终每条事实同时绑定 requirement ID 与证据 chunk。

在最终组合报告中，48 题严格通过率由 **41/48（85.42%）提升到 43/48（89.58%）**，
增加 2 题、4.17 个百分点：

- 中国平安 2024 年集团归母营运利润/归母净利润与寿险分部指标混淆：2/4 → 4/4；
- 宁德时代 2024 年三类产品收入、同比变化与毛利率：6/9 → 9/9。

安全拒答和澄清准确率继续保持 100%。本阶段未修改 Gold、评分器、索引或 frozen test。

## 机制

每个原子事实要求记录：

- `requirement_id`；
- 事实描述；
- 主体及 `group` / `business_segment` / `document` 口径；
- 事实期间；
- 表格、叙述、审计风险或审计应对等证据类型；
- 候选证据 chunk。

运行流程：

1. 先在已限定年报中检索锚点并展开相邻页；
2. DeepSeek 根据任务和证据生成非重叠原子事实清单；
3. 本地补齐 DeepSeek 漏掉但问题中可确定抽取的指标；
4. 解除集团要求与业务分部证据的错误预绑定；
5. 只对仍无候选证据的 requirement 定向检索；
6. 最终回答必须覆盖全部 requirement，并逐项绑定事实和引用；
7. 集团/分部口径冲突、未知引用或漏项均不能通过本地提交门禁。

没有保存或评测自由形式思维链，产品日志只包含结构化清单、工具调用和校验错误。

## 评测结果

| 数据集 | P2-B 严格通过 | P3-A 严格通过 | P2-B 事实准确率 | P3-A 事实准确率 |
|---|---:|---:|---:|---:|
| calibration | 22/24（91.67%） | 22/24（91.67%） | 92.31% | 91.35% |
| dev | 19/24（79.17%） | 21/24（87.50%） | 92.45% | 97.17% |
| 合计 | 41/48（85.42%） | 43/48（89.58%） | — | — |

最终 28 道抽取轨迹的新增结构诊断：

- 原子任务覆盖率：100%；
- requirement 证据绑定覆盖率：100%；
- 口径校验通过率：100%；
- claim 引用完整率：100%。

这些是结构诊断，不能替代 Gold 事实正确率。校准集审计题表明，即使清单和引用覆盖均为
100%，仍可能因为 PDF 页面内容未进入 chunk 或同义词面差异而未通过严格事实评分。

## 成本

最终 calibration + dev 抽取运行：

- DeepSeek 请求：80 次；
- 输入 token：269,932；
- 输出 token：31,328；
- 合计 token：301,260；
- 检索/页面工具调用：85 次。

包含无效配置、校准和 post-hoc 修复运行的全部实验共消耗 887,763 token。P3-A 的准确率
收益成立，但成本明显偏高；后续应把简单表格题的清单改为确定性生成，只让 DeepSeek 处理
审计事项、复杂原因列表和口径冲突。

## 负实验与治理说明

- calibration v1 使用了错误的通用 benchmark 索引，全部公司被输入门禁判为不存在；该次
  标记为无效配置，不参与效果比较。
- calibration v2-v4 严格分均为 12/14；审计题存在一条词面随机波动，但没有严格回归。
- 首次 dev 暴露两个本地校验 bug：“风险贴现率”被误判成审计风险，以及分部归母营运利润
  被强制判成集团口径。
- dev post-hoc v2 又暴露主体字段过度逐字校验和确定性指标漏项直接整题拒答的问题。
- 最终 dev v3 为修复后的 post-hoc 回归，不是无偏的首次开发集确认。上述失败报告全部保留，
  `frozen_test` 仍未打开；下一次无偏确认必须使用仍封存的数据。

## PDF 复核发现

人工视觉复核工商银行 2023 年审计报告相关页后确认，“设定多宏观情景及权重”确实出现在
PDF 第 197 页右栏，但当前年报 chunk 中该页被目录文本和双栏阅读顺序污染，进入 Agent
Evidence Memory 的 excerpt 没有这条审计程序。因此该题不是单纯的回答遗漏，而是已有
原生文字没有被可靠重建进检索证据。

这说明下一阶段不应继续扩 P3-A 提示词，而应做结构/来源层：

1. 双栏审计报告的列级阅读顺序与页级重建；
2. 权威来源页排序和 claim-source 绑定；
3. 对复杂页 A/B 测试结构化解析器或视觉文档模型，而不是全量 OCR。

## 产物

- 增量汇总：`reports/agent/agent-hard-v3-p3a-atomic-fact-ledger-improvement.json`
- calibration 组合：`reports/agent/agent-hard-v3-calibration-deepseek-p3a-composed.json`
- dev 组合：`reports/agent/agent-hard-v3-dev-deepseek-p3a-composed.json`
- calibration 最终抽取：`reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v4.json`
- dev 最终抽取：`reports/agent/agent-hard-v3-dev-deepseek-p3a-atomic-fact-ledger-posthoc-v3.json`
- 可复现子集：`data/evaluation/agent-hard-v3-calibration-extract.json`、
  `data/evaluation/agent-hard-v3-dev-extract.json`
