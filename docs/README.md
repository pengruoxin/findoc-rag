# FinDocRAG 文档索引

## 从哪开始

| 想知道 | 看 |
|---|---|
| **某个词是什么意思** | [glossary-zh.md](./glossary-zh.md) |
| 项目在什么水平、下一步做什么 | [roadmap-zh.md](./roadmap-zh.md) ← **新读者从这里** |
| Agentic RAG 怎么改、当前做到哪 | [agentic-rag-roadmap-zh.md](./agentic-rag-roadmap-zh.md) |
| Agent 任务怎么运行和评测 | [agent-tasks-zh.md](./agent-tasks-zh.md) |
| Agent P0 每次改动提升或回退了多少 | [evaluation/agent-p0-improvement-log-zh.md](./evaluation/agent-p0-improvement-log-zh.md) |
| Agent P1 跨页、计算、视觉工具逐次结果 | [evaluation/agent-p1-pdf-tools-log-zh.md](./evaluation/agent-p1-pdf-tools-log-zh.md) |
| Agent P2-A 整份文档盲测、来源复核与下一单变量 | [evaluation/agent-p2a-document-blind-baseline-zh.md](./evaluation/agent-p2a-document-blind-baseline-zh.md) |
| Agent P2-B1 文档年份/事实期间单变量提升 | [evaluation/agent-p2b1-document-scope-log-zh.md](./evaluation/agent-p2b1-document-scope-log-zh.md) |
| Agent P3-A 原子事实清单与口径绑定 | [evaluation/agent-p3a-atomic-fact-ledger-log-zh.md](./evaluation/agent-p3a-atomic-fact-ledger-log-zh.md) |
| Agent P3-B PDF 分栏重建与权威来源排序 | [evaluation/agent-p3b-pdf-layout-authority-log-zh.md](./evaluation/agent-p3b-pdf-layout-authority-log-zh.md) |
| Agent P4-A 独立证据审计与多 Agent 成本收益 | [evaluation/agent-p4a-evidence-verifier-log-zh.md](./evaluation/agent-p4a-evidence-verifier-log-zh.md) |
| Agent P4-B 本地 Claim 风险闸门与零 token 硬冲突拦截 | [evaluation/agent-p4b-claim-risk-gate-log-zh.md](./evaluation/agent-p4b-claim-risk-gate-log-zh.md) |
| Agent P4-C 未知语义错误的开放式风险路由 | [evaluation/agent-p4c-open-risk-routing-log-zh.md](./evaluation/agent-p4c-open-risk-routing-log-zh.md) |
| Agent P4-D 高语言重叠关系/会计符号风险 | [evaluation/agent-p4d-high-overlap-risk-log-zh.md](./evaluation/agent-p4d-high-overlap-risk-log-zh.md) |
| Agent P4-E 原子事实 Support Proof 与人工升级 | [evaluation/agent-p4e-support-proof-log-zh.md](./evaluation/agent-p4e-support-proof-log-zh.md) |
| Agent P4-F 不可变人工复核队列与审批闭环 | [evaluation/agent-p4f-human-review-workflow-log-zh.md](./evaluation/agent-p4f-human-review-workflow-log-zh.md) |
| Agent P4-G 表格单元格与 PDF 坐标证明 | [evaluation/agent-p4g-table-cell-geometry-proof-log-zh.md](./evaluation/agent-p4g-table-cell-geometry-proof-log-zh.md) |
| PDF 证据增强：假表清除、调整前后表头与有界区域证明 | [evaluation/pdf-evidence-enhancement-log-zh.md](./evaluation/pdf-evidence-enhancement-log-zh.md) |
| benchmark-v3 怎么独立标注 | [benchmark-v3-annotation-protocol-zh.md](./benchmark-v3-annotation-protocol-zh.md) |
| PDF 原生/扫描/混合页怎么评测 | [evaluation/pdf-extraction-benchmark-zh.md](./evaluation/pdf-extraction-benchmark-zh.md) |
| 当前分数是多少 | [evaluation/baseline-zh.md](./evaluation/baseline-zh.md) |
| 指标怎么定义、门禁怎么判 | [evaluation/benchmark-and-metrics-zh.md](./evaluation/benchmark-and-metrics-zh.md) |
| 系统怎么实现的 | [architecture/](./architecture/) |
| 怎么讲这个项目 | [interview/findoc-rag-interview-guide-zh.md](./interview/findoc-rag-interview-guide-zh.md) |
| 简历要写哪些数字 | [interview/phase-summaries-zh.md](./interview/phase-summaries-zh.md) |

## 系统设计 [architecture/](./architecture/)

按"从需求到运行时"顺序阅读：

| 文档 | 内容 |
|---|---|
| [product-scope.md](./architecture/product-scope.md) | 目标用户、业务任务、non-toy 判定标准 |
| [indexing.md](./architecture/indexing.md) | 持久化索引布局、chunk provenance、lexical/dense 分离 |
| [versioning.md](./architecture/versioning.md) | 文档身份、不可变版本、corpus generation 原子切换 |
| [scope-routing.md](./architecture/scope-routing.md) | 口径路由：季度/分部/合并/母公司/附注线索识别与重排 |
| [reranking.md](./architecture/reranking.md) | Cross-encoder 重排作为可选独立阶段 |
| [api.md](./architecture/api.md) | 配置优先级、环境变量、HTTP 契约 |
| [observability.md](./architecture/observability.md) | trace 身份、分阶段记录、指标 |
| [term-normalization-design-zh.md](./architecture/term-normalization-design-zh.md) | 同义词可扩展性：三层术语处理架构（LLM 改写 / 指标知识库 / 快速兜底） |

## 评测体系 [evaluation/](./evaluation/)

按"规则 / 数字 / 计划"分工，每个数字只存一处：

| 文档 | 回答什么问题 | 何时改 |
|---|---|---|
| [benchmark-and-metrics-zh.md](./evaluation/benchmark-and-metrics-zh.md) | **规则** —— 数据集怎么构造、指标怎么定义、门禁怎么判、报告怎么写 | 规则变时 |
| [baseline-zh.md](./evaluation/baseline-zh.md) | **数字** —— 当前规模、当前分数、当前薄弱点、迭代协议 | 每轮实验后 |
| [improvement-list-zh.md](./evaluation/improvement-list-zh.md) | **计划** —— P0–P4 改进清单与状态 | 完成一项时 |
| [experiment-summaries.md](./evaluation/experiment-summaries.md) | 每次实验完整分析的索引与规范 | 新实验时 |
| [pdf-extraction-benchmark-zh.md](./evaluation/pdf-extraction-benchmark-zh.md) | PDF 页级路由、OCR 兜底与消融基线 | 解析策略或视觉后端变化时 |
| [agent-p1-pdf-tools-log-zh.md](./evaluation/agent-p1-pdf-tools-log-zh.md) | 跨页窗口、受控计算、PDF 几何关系与真实 DeepSeek 增量 | Agent PDF 工具变化时 |
| [agent-p2a-document-blind-baseline-zh.md](./evaluation/agent-p2a-document-blind-baseline-zh.md) | 5 份未见年报盲测、严格分与来源复核、失败分类 | Agent 泛化基线或 P2 单变量变化时 |
| [agent-p2b1-document-scope-log-zh.md](./evaluation/agent-p2b1-document-scope-log-zh.md) | 文档范围/事实期间分离的 +4 配对结果、负面中间运行与成本 | P2-B1 规划逻辑变化时 |
| [agent-p3a-atomic-fact-ledger-log-zh.md](./evaluation/agent-p3a-atomic-fact-ledger-log-zh.md) | 原子事实清单、口径门禁与多事实遗漏修复 | 抽取清单或口径绑定变化时 |
| [agent-p3b-pdf-layout-authority-log-zh.md](./evaluation/agent-p3b-pdf-layout-authority-log-zh.md) | 异常 PDF 文字层分栏重建、来源权威排序、配对结果与成本 | PDF 页级证据或来源选择变化时 |
| [agent-p4a-evidence-verifier-log-zh.md](./evaluation/agent-p4a-evidence-verifier-log-zh.md) | 独立证据 Verifier、一次修复、故障注入、成本收益与是否默认多 Agent | 验证 Agent、修复循环或多 Agent 策略变化时 |
| [agent-p4b-claim-risk-gate-log-zh.md](./evaluation/agent-p4b-claim-risk-gate-log-zh.md) | Verifier 前的确定性主体/时期/数字/单位/引用风险闸门与成本配对 | 本地风险规则或高风险路由变化时 |
| [agent-p4c-open-risk-routing-log-zh.md](./evaluation/agent-p4c-open-risk-routing-log-zh.md) | 未知语义错误、通用 claim/evidence 异常信号、自动高风险路由与消融成本 | 开放风险信号、路由策略或 Verifier prompt 变化时 |
| [agent-p4d-high-overlap-risk-log-zh.md](./evaluation/agent-p4d-high-overlap-risk-log-zh.md) | 高语言重叠标签/关系/会计符号故障、contract/completeness 门禁、负实验与复跑波动 | 高重叠故障、会计符号、必要数字或独立 Verifier 策略变化时 |
| [agent-p4e-support-proof-log-zh.md](./evaluation/agent-p4e-support-proof-log-zh.md) | claim→cited quote 证明、最弱 contract 选择、表头 challenge、人工升级和七轮消融 | Support proof、challenge、人工审批或独立模型策略变化时 |
| [agent-p4f-human-review-workflow-log-zh.md](./evaluation/agent-p4f-human-review-workflow-log-zh.md) | 候选保留、不可变审核包、approve/correct/reject、stale 与包外证据门禁 | 人工复核状态机、审核权限或表格证据展示变化时 |
| [agent-p4g-table-cell-geometry-proof-log-zh.md](./evaluation/agent-p4g-table-cell-geometry-proof-log-zh.md) | 表格 row/column/value、PDF value bbox、cell proof hash 与文本 fallback 边界 | 表格 sidecar schema、坐标绑定或审核区域证明变化时 |
| [pdf-evidence-enhancement-log-zh.md](./evaluation/pdf-evidence-enhancement-log-zh.md) | 设计型年度表、伪表门禁、有界 PNG/hash region proof 与人工 hard-case 评测 | PDF 表头恢复、区域证明或审核视觉证据变化时 |
| [diagnostics-and-holdout.md](./evaluation/diagnostics-and-holdout.md) | benchmark-v2 之前的排序诊断集与 holdout 审核流程 | 少 |

## 面试材料 [interview/](./interview/)

- [findoc-rag-interview-guide-zh.md](./interview/findoc-rag-interview-guide-zh.md) —— 项目亮点、技术路线、问答准备、90 秒模板
- 评测集的讲法在 [baseline-zh.md §7](./evaluation/baseline-zh.md#7-面试怎么讲这套评测集)
- `interview-capability-map.md` —— 本地草稿（gitignored），对照另一项目的能力迁移

## 历史记录 [history/](./history/)

[optimization-log-zh.md](./history/optimization-log-zh.md) —— 逐次改动的可复现对比（代码、测试、指标、已知退化），最新在最前。

## 实验产物 [reports/](../reports/)

每次 run 的原始数字不在 docs 下：

| 目录 | 内容 |
|---|---|
| [ranking/](../reports/ranking/) | 检索评测与失败分析：holdout v2、结构诊断、**variant-regime-v1（P1 最新）** |
| [generation/](../reports/generation/) | 生成三轨 runs、RAGAS、数据集卡、逐题配对比较 |
| [processing/](../reports/processing/) | PDF / Document IR / chunking 处理基线 |
| [reranking/](../reports/reranking/) | Cross-encoder 重排验证记录 |
| [agent/](../reports/agent/) | Agent 困难集原始轨迹、P0/P1 增量、P2/P3 配对汇总与负实验 |

最新实验入口：[variant-regime-expanded-v2 分析](../reports/ranking/variant-regime-expanded-v2/analysis.md)（同义词查询改写，检索瓶颈归因完成）。

## UI 页面 [ui/](./ui/)

服务启动后挂载在 `/ui`，也可直接用浏览器打开：

| 页面 | 用途 |
|---|---|
| [workspace-v4.html](./ui/workspace-v4.html) | 当前证据工作台（`/` 默认跳转到此） |
| [holdout-review.html](./ui/holdout-review.html) | 候选证据审核 |
| [holdout-eval.html](./ui/holdout-eval.html) | provisional holdout manifest |
| [holdout-failures.html](./ui/holdout-failures.html) | 检索失败分类与样例 |
| [experiment-dashboard.html](./ui/experiment-dashboard.html) | 实验注册表与结论边界 |

`workspace-v4.html` 是可部署的证据优先 Agent 工作台：可上传并解析 PDF，在当前标签页配置
DeepSeek Key，运行问答、对比、精确抽取和计算任务，并处理高风险任务进入的人工审核队列。
抽取任务默认按风险决定是否开启独立 Evidence Verifier，不是所有问题都固定消耗第二次模型调用。

证据卡片优先展示公司、报告年份、页码、章节路径、支持的结论和可阅读原文；数字会自动强调，
chunk ID 与 SHA-256 只放在折叠的防篡改详情中。回答中的引用可直接定位对应证据，页面还会通过
`/v1/evidence:resolve` 校验证据是否仍属于当前索引。检索参数、Verifier 策略和运行时能力集中在
工作台抽屉中。DeepSeek Key 不写入浏览器存储或服务器文件，刷新页面即清除。

`workspace-v2.html`、`workspace-v3.html` 与 `workspace-wireframe.html` 是被 v4 取代的草稿，已删除。
