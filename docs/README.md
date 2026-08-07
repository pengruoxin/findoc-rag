# FinDocRAG 文档索引

## 从哪开始

| 想知道 | 看 |
|---|---|
| **某个词是什么意思** | [glossary-zh.md](./glossary-zh.md) |
| 项目在什么水平、下一步做什么 | [roadmap-zh.md](./roadmap-zh.md) ← **新读者从这里** |
| 当前分数是多少 | [evaluation/baseline-zh.md](./evaluation/baseline-zh.md) |
| 指标怎么定义、门禁怎么判 | [evaluation/benchmark-and-metrics-zh.md](./evaluation/benchmark-and-metrics-zh.md) |
| 系统怎么实现的 | [architecture/](./architecture/) |
| 怎么讲这个项目 | [interview/findoc-rag-interview-guide-zh.md](./interview/findoc-rag-interview-guide-zh.md) |

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
| [improvement-list-zh.md](./evaluation/improvement-list-zh.md) | **计划** —— P0–P3 改进清单与状态 | 完成一项时 |
| [experiment-summaries.md](./evaluation/experiment-summaries.md) | 每次实验完整分析的索引与规范 | 新实验时 |
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

最新实验入口：[variant-regime-expanded-v2 分析](../reports/ranking/variant-regime-expanded-v2/analysis.md)（同义词查询改写，检索瓶颈归因完成）。

## UI 页面 [ui/](./ui/)

服务启动后挂载在 `/ui`，也可直接用浏览器打开：

| 页面 | 用途 |
|---|---|
| [workspace-v3.html](./ui/workspace-v3.html) | 当前查询工作台（`/` 默认跳转到此） |
| [holdout-review.html](./ui/holdout-review.html) | 候选证据审核 |
| [holdout-eval.html](./ui/holdout-eval.html) | provisional holdout manifest |
| [holdout-failures.html](./ui/holdout-failures.html) | 检索失败分类与样例 |
| [experiment-dashboard.html](./ui/experiment-dashboard.html) | 实验注册表与结论边界 |

`workspace-v2.html` 与 `workspace-wireframe.html` 是被 v3 取代的草稿，未纳入版本控制。
