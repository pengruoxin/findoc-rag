# FinDocRAG → Agentic RAG 改造路线

> 状态：P0 工程与助手标注已完成，独立人工双审待补。本文是后续改造的唯一方向清单；原有
> [总体路线图](./roadmap-zh.md)继续记录传统 RAG、PDF 与表格能力。

## 目标边界

当前系统是证据工具完备的固定 RAG 流水线，不把 `Agent-ready` 接口表述为
已经具备 Agent 控制能力。目标是在保留确定性财务快路径的前提下增加：

1. 问题复杂度路由：直接查询、澄清、计算、多跳、跨文档；
2. 可审计的计划与工具调用；
3. Evidence Memory、证据充分性判断与缺口驱动的再次检索；
4. 有界循环、预算、停止条件和逐 claim 证据验证；
5. 对最终答案和完整执行轨迹同时评测。

第一版采用**单 Agent + 多工具**。只有工具、权限或策略确实需要隔离时，才拆分
表格、检索或语料写入 Agent；不以多 Agent、GraphRAG 或更换框架充当能力提升。

## P0：评测公信力（助手可执行部分已完成）

当前进度：语料分配、官方来源哈希、版本入库、split 隔离索引、60 条题目、129 个
原子事实和一次冻结评测均已完成。20 条可回答 frozen 题的 gold evidence 全部进入
retrieved context，但 strict 仅为 4/21，和 oracle-context 相同，瓶颈已定位到回答层
覆盖而不是本轮检索。独立人工双审尚未完成，因此仍不能对外声称 benchmark-v3
成绩。标注执行遵循
[Benchmark v3 独立标注协议](./benchmark-v3-annotation-protocol-zh.md)。

### 门禁

- split 必须按整份 `document_version_id` 隔离，而不是按题目随机切分；
- 同一公司同一报告年度不得同时出现在 dev 和 frozen test；
- 同一 `family_id` 不得跨 split；
- 每条题目、证据、参考答案和问法变体至少由两名独立人工审核者批准；
- 至少 4 家公司、2 个报告年度、24 条 frozen-test 题；
- `independent_gold=true` 且 `status=human_frozen` 才允许对外主张成绩；
- 报告必须按 provider、检索过滤来源、split、题型、难度和预期行为分层；
- strict、行为正确率和错误率同时给题目级与文档级 bootstrap 置信区间。

当前 `benchmark-v2` 保留作历史回归，不改写历史身份。它会被治理审计如实判为
`ready_for_external_claims=false`：2 家公司、1 个年度、文档跨 split、0 条双人审核。

运行审计：

```bash
uv run python scripts/audit_benchmark_governance.py \
  --output reports/validation/benchmark-governance-p0.json
```

准备好的 v3 语料还必须通过来源、版本和索引成员关系审计：

```bash
uv run python scripts/audit_corpus_plan.py \
  --plan data/evaluation/benchmark-v3-corpus-plan.json \
  --policy configs/evaluation-governance-p0.json

uv run python scripts/audit_prepared_corpus.py
```

新 benchmark 在发布或对外引用前必须追加：

```bash
uv run python scripts/audit_benchmark_governance.py \
  --dataset data/evaluation/<new-benchmark>.json \
  --require-external-claims-ready
```

人工审核记录写入每条 item 的 `annotation.human_reviews`：

```json
{
  "reviewer_id": "stable-pseudonymous-id",
  "reviewed_at": "2026-08-19T10:00:00+08:00",
  "verdict": "approve",
  "query_semantics_verified": true,
  "evidence_verified": true,
  "reference_answer_verified": true,
  "variants_verified": true,
  "notes": ""
}
```

审核者不能是题目生成者；`reviewer_id` 使用稳定匿名标识，不记录个人敏感信息。

## P1：工具化与复杂度路由

把现有能力收敛为稳定工具：`route_query`、`search_evidence`、
`search_sections`、`get_structured_table`、`resolve_evidence`、
`calculate_financial_metric`、`judge_sufficiency`、`verify_claims`。

简单单事实题继续走现有低成本快路径；歧义题澄清；只有比较、计算、跨页和
跨文档问题进入 Agent 循环。公司、股票代码、指标词表改为数据驱动注册表，
逐步移除回答层对贵州茅台和伊利股份的硬编码。

当前进度：已落地第一版只读 `compare` 任务及 `agent run/inspect` 命令。默认 runtime
现在是真实 DeepSeek tool calling；模型选择分目标检索和结构化提交，本地代码负责 target、
filter、证据归属和完整性门禁。没有 provider key 时默认命令拒绝运行，评测写
`status=not_run`。固定规则控制器只保留为显式 `deterministic-baseline`，不能当作 Agent
成绩。`agent-compare-v1` 的离线对照仍可证明计划和证据覆盖，但 grounded 覆盖为 0。

新增 `agent-hard-v1` 作为后续提升尺子：8 个 PDF 困难任务覆盖跨页表、视觉关系、口径
冲突、重述值、事实年度/文档年度错位、四目标完整性与安全拒答。首次真实
`deepseek-v4-flash` 运行的事实准确率、正确拒答率和端到端通过率均为 0，12 次模型请求
触发 25 次检索调用，用于冻结缺陷。

P0-A/P0-B 已在同一数据集、同一索引和同一本地硬评分器上完成：缺口重试、新鲜 finalizer、
事实年度/报告年份/报表口径/数值版本 target、财务表定位词、跨目标 provenance 证据组、
零证据本地拒答与受控引用补齐均已落地。最终 P0-B8 和复现实验都达到 5 个可执行 compare
难例 5/5、事实准确率 100%、安全拒答 100%；全量为 5/8，剩余 3 题仍是未实现的
`extract` / `calculate`，没有通过删题或改评分器隐藏。最终模型请求 10 次、输入 22,144
token、检索 13 次。命令与指标见 [Agent 任务与困难评测](./agent-tasks-zh.md)，完整消融见
[Agent P0 逐增量评测日志](./evaluation/agent-p0-improvement-log-zh.md)。

P1 PDF 工具也已完成：`extract` 用同文档 `get_page_window` 解决跨页章节状态；受控
`calculate` 要求每个操作数绑定 chunk 并由本地 `Decimal` 运算；权益结构图只在需要时
调用 `inspect_page_region`，从 manifest 白名单 PDF 的原生文字 bbox 和连接线重建关系，
DeepSeek确认关系后本地求和。阶段间使用独立证据上下文，避免并行工具调用污染后续协议。
最终同一困难集从 5/8 提升到 **8/8**，任务覆盖、事实、行为、引用和安全拒答均为 1.0000；
完整 8 题运行 19 次模型请求、输入 36,769 token。边界是样本小、视觉 gold 非独立盲标、
尚未覆盖真实扫描表。逐增量与负结果见
[Agent P1 PDF 工具日志](./evaluation/agent-p1-pdf-tools-log-zh.md)。

## P2：有界 Agent 控制器

控制器采用 `plan → act → observe → judge`：每轮把新增证据写入带来源的
Evidence Memory；Judge 返回 `sufficient`、`conflicting` 或结构化 gaps；gaps
转换为下一轮检索请求。

默认上限：3 轮、8 次工具调用；固定一次运行的 index ID；无新增证据、重复查询、
达到成本/时间预算或证据冲突无法消解时停止并澄清或拒答。只保存结构化决策和
工具结果，不把自由形式思维链作为产品日志。

当前 `compare`、跨页 `extract`、受控勾稽和视觉关系计算都采用有界控制器：比较任务按
gaps 重试；PDF 工具任务按检索、结构观察、提交/计算分阶段隔离；零证据或来源校验失败均
确定性拒答；每轮保存 token、函数名和本地 `validation_errors`。

P2-A 已把评测扩到 5 份开发阶段未见年报、34 题：冻结严格分 17/34，助手逐页来源复核
24/34；10 个真实失败集中在 4 个通用计算、4 个文档范围/事实期间规划和 2 个多事实遗漏。
P2-B1 已分离 extract 规划中的 `document_year` / `fact_period` 并增加唯一文档范围推断，
严格分提升到 **21/34**，4 fixed / 0 regressed，来源复核 28/34；两次运行和成本见
[P2-B1 单变量日志](./evaluation/agent-p2b1-document-scope-log-zh.md)。下一单变量 P2-B2 只处理
剩余两个多事实遗漏，不同时扩计算器。

扩展到 hard-v3 后，通用计算、语料年份拒答和核验答案契约已依次落地。最新 P2-B 将
多年度比较从“每个公司/年度有一个 chunk 即充分”升级为逐指标缺口判断：8 道目标题
校准与开发均为 4/4，48 题组合严格通过率达到 **41/48（85.42%）**，安全拒答和行为
准确率均保持 100%。中国平安多指标任务实际触发第二轮缺口检索，而非一次性扩大 top-k；
负实验和成本见 [P2-B 多指标缺口日志](./evaluation/agent-p2b-metric-gap-log-zh.md)。下一项转向
抽取任务的多事实清单与权威引用页排序，不再继续扩比较检索。

P3-A 已为抽取任务增加证据感知的原子事实清单：DeepSeek 在候选证据上拆分主体、指标、
期间、集团/分部口径和证据类型，本地补齐确定性指标漏项，只检索无证据 requirement，
最终逐事实绑定 requirement ID 与 chunk 引用。组合严格通过率由 **41/48（85.42%）**
提升到 **43/48（89.58%）**，修复平安集团/寿险分部混淆和宁德时代三产品同比表达，
0 严格回归；成本和 dev post-hoc 治理限制见
[P3-A 原子事实清单日志](./evaluation/agent-p3a-atomic-fact-ledger-log-zh.md)。视觉复核同时确认
工商银行 2023 年审计报告双栏页存在原生文字未进入正确 chunk 的问题，下一单变量转向
列级阅读顺序/页级重建与权威来源页排序，不继续对清单提示词过拟合。

P3-B 已把这两个来源层问题落地：关键审计事项按 manifest/hash 白名单渲染锚点页与续页，
左右栏独立 OCR 后作为页级证据进入 Agent；多指标抽取和核验任务扩大候选，并且只有更
权威页面包含已提交数值时才补入引用。组合严格通过率由 **43/48（89.58%）提升到
45/48（93.75%）**，修复 2 个引用来源失败；工商银行审计题 exact 6/13 → 7/13，人工
逐页复核为 13/13 grounded，但同义词差异仍按机器未命中保留。完整负实验、成本与限制见
[P3-B PDF 版面/来源日志](./evaluation/agent-p3b-pdf-layout-authority-log-zh.md)。frozen test 未打开，
下一步是一次性无偏确认，而不是继续看开发集调权重。

P4-A 已验证“主 Agent + 独立 Evidence Verifier”：正常 48 题仍为 **45/48**，0 提升、
0 回退，新增 80,817 token；15 个 claim 故障注入中安全处理 **14/15**，其中细项遗漏
5/5 完成一次修复并通过二次复核。结论是保留高风险复杂抽取的显式可选开关，不默认多
Agent，也不继续增加角色。实现、负实验和成本见
[P4-A Evidence Verifier 日志](./evaluation/agent-p4a-evidence-verifier-log-zh.md)。

P4-B 在 Verifier 前增加确定性 claim 风险闸门：主体/期间/引用范围的明确冲突零 token
拒答，数字/单位异常只升级复核。正常严格分保持 **45/48**；同一 15 个故障由 **14/15
提升到 15/15**，请求 25→20，token 100,941→80,882（-19.87%）。这进一步支持“便宜
规则先行、只有高风险才多 Agent”，而不是默认增加角色。实现与配对结果见
[P4-B Claim 风险闸门日志](./evaluation/agent-p4b-claim-risk-gate-log-zh.md)。

P4-C 进一步补“未知错误不在规则中”的开放集盲区：claim/evidence 语言覆盖异常只触发
DeepSeek 开放式证据审计，不本地拒答。同代码未见故障消融把路由和安全处理从 **9/15
提升到 15/15**；正常仍为 **45/48**，Verifier 请求仍为 18。抽取 CLI 现默认 `auto`，
只有原有高风险或开放风险命中才复核，可显式 `off/always`。完整负实验、成本和边界见
[P4-C 开放风险路由日志](./evaluation/agent-p4c-open-risk-routing-log-zh.md)。

P4-D 专门测试字符覆盖 0.824–1.000 的标签、关系和会计符号偷换，证明 P4-C 的低语言
覆盖并不足够：基线只安全处理 5/15。新增简单任务 requirement contract 偏离复核、明确
会计正负号本地冲突和必要数字完整性检查后，提升到 **14/15**；正常仍为 **45/48**、请求
仍为 18，token 增加 2.92%。剩余“同比下降→同比上升”仍被同模型 Verifier 接受，因此
不继续堆方向词规则；下一步应比较不同 provider/model 或人工升级。完整迭代、评测器修复
与失败方案见 [P4-D 高重叠风险日志](./evaluation/agent-p4d-high-overlap-risk-log-zh.md)。

P4-E 不再接受 Verifier 只返回 supported ID：非审计复核从 contract 相似度最低的一项生成
claim→cited quote support proof，低语言对齐时做一次带有限表头上下文的对抗挑战，证明契约
无效则进入结构化人工复核。同代码消融把 P4-D 剩余关系错误从 **14/15 提升到 15/15**，正常
仍为 **45/48**；代价是正常请求 18→22、token +19.05%。每项都生成 proof 的第一版曾把
calibration 压到 8/14，已否决。当前仍是同一 DeepSeek，不是模型独立裁判；人工状态也尚未
接工单/审批回写。完整迭代见
[P4-E Support Proof 日志](./evaluation/agent-p4e-support-proof-log-zh.md)。

P4-F 已把上述人工状态接成实际闭环：候选答案不再随拒答丢失；自动生成包含 atomic claim、
PDF 页码、chunk hash 和 excerpt 的待审包；提供 `agent review list/inspect/resolve`，支持批准、
证据绑定修正和驳回。task trace hash 漂移、审核包外证据与重复审批都会 fail closed，resolution
作为追加记录存在，不覆盖原模型 trace。基于一条已存真实 DeepSeek trace 的工作流契约
**10/10**，额外 provider 请求/token 为 0；这只证明人机状态机，不证明人审准确率。实现与
边界见 [P4-F 人工复核闭环日志](./evaluation/agent-p4f-human-review-workflow-log-zh.md)。下一步
回到 PDF 表格 cell/geometry proof，而不是继续增加 Agent 角色。

P4-G 已完成这一步：PDF IR 的数值 token page/bbox 不再在 sidecar 转换时丢失；每个表格
cell 生成绑定 table/chunk/row/column/value/geometry 的 SHA-256 proof，并由 AgentEvidence
传给人工审核包。calibration + dev 的 11 表/111 格语义 111/111 不变，proof 111/111；
有精确 value bbox 的单元格从 **0/111 提升到 102/111（91.89%）**，坐标原生路径 72/72，
另有 30 格通过唯一 `row+column+value` 匹配保守补坐标，9 格保持 text-only。全程 0 模型
调用且 frozen 未打开。见
[P4-G Cell/Geometry Proof 日志](./evaluation/agent-p4g-table-cell-geometry-proof-log-zh.md)。

## P3：验证、轨迹评测与生产边界

- 逐 claim 检查主体、期间、口径、数字、单位、引用正确性和引用完整性；
- 增加 complexity route、tool selection、subquery completeness、hop recall、
  sufficiency precision/recall、unsupported-claim rate、调用次数、成本和延迟；
- 回答 Agent 与语料写入 Agent 分权，写操作要求鉴权、幂等键和人工批准；
- Web 进程不承担索引构建，改为持久任务队列和独立 worker；
- 增加 native-first PDF 工具链：原生坐标文本为快路径，扫描/混合区域选择性 OCR，
  复杂表格和结构冲突才升级到视觉文档模型；三层来源分别落盘和评测。当前页级路由与
  native/hybrid 消融见 [PDF 提取基准](./evaluation/pdf-extraction-benchmark-zh.md)。单页
  退化集和海尔真实三页连续表已进入硬评测；跨页 7 题已由工作区用户完成候选标注核验
  并升级为 `human_verified`，但不是独立盲标。当前 2,614 页语料审计仍未发现真实扫描
  财务表，因此不能宣称真实扫描表能力已闭环。

## 对照实验

所有 Agent 改动必须在同一个 `agent-hard-v2` document-blind frozen candidate 和
`agent-hard-v1` 回归集上对比：

1. 固定单次 RAG；
2. 复杂度路由但无迭代；
3. 有充分性/缺口判断的有界迭代 Agent。

只有第三组在事实正确率、证据充分性、口径选择或拒答校准上改善，且额外调用成本可接受，
才能称为 Agentic RAG 能力提升。特别要单列 `document_vintage` 与 `fact_period`，避免
2024 年报中的 2023 调整后比较数被 2023 文档 filter 永久挡在 Agent 之外。
