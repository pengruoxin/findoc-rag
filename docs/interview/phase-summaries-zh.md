# FinDocRAG 分阶段成果摘要（简历/面试用，最精简版）

> 每阶段一行定位 + 可量化的指标提升。**对外口径**与**内部口径**分开标：带 ⚠️ 的数字只在内部迭代用，简历上不要写。
> 完整证据与受控实验记录见 [baseline-zh.md](../evaluation/baseline-zh.md)、[optimization-log](../history/optimization-log-zh.md)。

## 双层 RAG 评估体系（面试必讲）

RAG 评估分两层：**检索侧保证能找到，生成侧保证用得好**。没有评估的优化，就是拿着锤子找钉子。

### 检索侧：4 个指标全部落地

| 指标 | 实现位置 | 最新数字示例（variant-regime，query_parser 过滤） | 口径 |
|---|---|---|---|
| Recall@K | `run_retrieval_variant_eval.py` / `diagnostics.py` | canonical lexical Recall@5 = **0.8108** | gold 在 top-K 中的覆盖率 |
| Precision@K | 同上 | canonical lexical Precision@5 = **0.2216** | 部分判定（见下） |
| MRR@K | 同上 + `evaluation/retrieval.py` | canonical lexical MRR@5 = **0.6937** | 首个 gold 排名倒数 |
| NDCG@K | `run_retrieval_variant_eval.py`（`ndcg_at_k`） | canonical lexical NDCG@5 = **0.7089** | 当前只有"相关/不相关"两档 |

另外还有两个超出标准清单、专门用于失败归因的指标：**候选池召回 candidate_recall**（区分"根本没捞到"和"捞到了排序砸了"）和**前 K 中干扰段落数 negative_count_in_top5**。

### 生成侧：4 个指标全部落地（RAGAS + 确定性打分双轨）

| 指标 | 实现位置 | 最终 index-bound 三轨（Oracle / Retrieved / Robustness） | 备注 |
|---|---|---|---|
| Faithfulness | `run_ragas_generation_eval.py`（`Faithfulness`） | **0.761** / **0.884** / **0.796** | LLM judge，`independent_judge=false` 自评 |
| Answer Relevancy | `ResponseRelevancy` | **0.908** / **0.907** / **0.894** | 同上 |
| Context Relevancy | `ContextRelevance`（产物字段 `nv_context_relevance`） | **1.000** / **1.000** / **0.944** | RAGAS 0.3.1 非原生实现，字段带 nv_ 前缀 |
| Context Recall | `LLMContextRecall` | **0.986** / **1.000** / **1.000** | Retrieved 另有确定性 37/37 gold-context 检查 |

生成侧还有一层**确定性打分器**（不依赖 LLM judge）：fact recall、数值/单位准确率、引用有效性、行为准确率（答/拒答/澄清）、context recall（gold chunk 是否在检索上下文里）——这些是 RAGAS 之外更硬的契约检查。

### 边界（面试主动说）

- Precision/NDCG 是部分判定下界（只有 gold 和 hard negatives 被标注，其余按不相关计）；
- NDCG 当前为二元相关，未做分级相关度；
- RAGAS 为 DeepSeek 自评（`independent_judge=false`），只作语义诊断；
- 每个 RAGAS metric coverage 和完整行 coverage 都是 100%，但完整覆盖不等于独立裁判；
- 冻结集 48 题 / 2 家公司 / 1 个年度，绝对分数需带边界声明。

简历一句话：

> 构建双层 RAG 评测体系：检索侧覆盖 Recall@K / Precision@K / MRR@K / NDCG@K 及候选召回与干扰污染归因，生成侧覆盖 RAGAS 四指标与确定性事实/单位/引用/行为门禁；所有优化以受控实验和逐题配对为准。

⚠️ 边界：RAGAS 为 DeepSeek 自评（`independent_judge=false`）；Precision/NDCG 是部分判定下界；NDCG 当前为二元相关；冻结集为 48 题 / 2 家公司 / 1 个年度。

## A 阶段：检索定论（✅ 完成）

**定位**：用受控实验证明"该用哪种检索"，并沉淀生产链路。

可写进简历的量化：

- 口语/相对时间问法 Hit@5 **0.73 → 0.92**（同义词改写，零回归）；
- 词表外口语改写 Hit@5 **0.194 → 0.694**（LLM 查询改写）；
- 权重扫描结论：**纯关键词检索全面优于任何 BM25×Dense 融合**，默认策略改为 lexical-only；
- 中文专用 dense 模型（bge-small-zh-v1.5）对照：三种问法全部低于 E5，语义路暂不启用；
- 生产 `/v1/query`：相对时间 / 公司别名 / 股票代码路由 + 确定性改写（LLM 可选、带缓存与质量门控），路由评测 **18/18 精确匹配**；
- LLM 改写进入 canonical 检索链路为**阴性结论**（strict 持平、3 个证据回归），因此只用于生产自由文本、不用于评测链路。

简历一句话：

> 通过融合权重扫描与 OOV 改写实验确定 lexical-only + 同义词改写策略，口语问法命中率提升 19 个百分点，词表外改写提升 50 个百分点，并落地到生产问答（别名/代码路由、改写缓存与质量门控）。

## B 阶段：表格重建与 index-bound 消费（✅ 冻结范围闭环）

**定位**：把"证据对了也抽不出数字"的瓶颈打掉，并让表格答案确定性化。

可写进简历的量化：

- 五类表型单元格抽取：**28/149 → 146/149（98.0%）**，另 concentration 8/8；
- 坐标级表格重建（整页输入、无区域裁剪）：**92/157 → 154/157**；安全选择把预测数 165→157，使 Precision / Recall 均为 **0.981**，PDF 直读与持久化 IR v2 一致；
- 五类表型变成 index-bound sidecar：真实两份年报自动发现 **15 表 / 195 cells**，12 表走坐标、3 表安全回退；artifact 与 index / snapshot / source chunk SHA 绑定，不改冻结 chunk identity；
- schema-aware 证据路由 + 自适应候选池 + 财务勾稽接入 API 与评测共用链路，最终三轨 strict / 行为均 1.0、错误率 0；Retrieved 37/37 gold context；
- 相对各 lane 历史最佳：Retrieved 6 strict fixed / 0 regressed，Robustness 1 / 0（0.9545→1.0000）。该跨度含多项工程变化和 API 随机性，不冒充单变量实验。

简历一句话：

> 自研中文财报坐标级表格重建与 index-bound 结构证据 sidecar，在 157-cell 冻结尺子上达到 98.1% Precision/Recall，并将结构路由、口径/年份绑定和财务勾稽接入同一 RAG 链路，使最终 Oracle / Retrieved / Robustness strict 与行为均达到 1.0。

## C 阶段：行为与时间（🟡 部分完成）

**定位**：让系统"知道什么时候该答、该拒、该追问"。

已完成：

- 远程拒答检测（打分口径修正）：应拒答题的拒答被如实计分、带数字的伪拒答不再刷分——Retrieved strict **0.571 → 0.800**、行为 **0.833 → 0.896**；Robustness strict **0.636 → 0.864**、行为 **0.793 → 0.828**（⚠️ 这是测量修正，不是模型能力提升，简历上不要写成提升）；
- 生产相对时间解析（"去年/前年/今年"）+ 时间对齐：语料外年份正确拒答；
- 未完成：行为拒答策略（可答题误拒答）、实时数据新鲜度、多轮评测。

简历一句话（可选，只讲已完成的）：

> 建立可审计的拒答/回答判定与时间对齐，区分"测量口径"与"模型能力"，避免伪提升。

## D 阶段：公信力（🟡 基础设施完成，外部验证待做）

已完成：冻结 benchmark 外部 SHA 锁、38 个最小源证据块、显式 `10fb...→9898...` migration、错索引 fail-closed、dirty-worktree code fingerprint、远程部分失败非零退出、RAGAS coverage 审计。

待做：多公司多年度 document-blind 评测、第二人独立复核（Cohen's kappa）、不同 provider 独立 judge、置信区间（题目层/文档层 bootstrap）、OOV 实例人工审核。

## Agent-ready 工程闭环（✅ 单机可演示）

- `/v1/query` 返回版本化 `outcome`、route/filter、index/trace identity 与 claim→citation，Agent 不用解析自然语言猜控制流；
- `/v1/capabilities` 只声明当前索引和配置真实具备的能力；`/v1/evidence:resolve` 用 index ID + chunk SHA 防止索引切换后证据漂移；
- 上传默认无写权限，显式授权后才进入 `validating → ingesting → indexing → ready/failed`，任务落盘、质量门禁和不可变索引切换均 fail-closed；
- 边界：当前是单机 BackgroundTasks 闭环，不冒充分布式 Agent 平台；生产还需认证、外部 worker、租约/心跳、重试和多进程 reload。

面试一句话：

> 我没有只做一个 `/chat` 接口，而是把 RAG 输出设计成 Agent 可消费的证据契约：控制流、能力发现、证据解析、索引身份和引用哈希都可机读，并把语料写入做成显式授权的持久化状态机。

## 对外/对内口径（重要）

✅ 简历可写（有受控实验 + 冻结集支撑）：

- 表格抽取 98.0%、坐标安全重建 Precision/Recall 98.1%、路由精确匹配 100%；
- 口语问法 Hit@5 0.92、OOV LLM 改写 0.694、最终 DeepSeek 三轨 strict / 行为均 1.0；strict 分母必须同时说明为 35 / 35 / 22；
- 最终三轨共享 migration / index / code fingerprint，远程错误率 0，Retrieved 37/37 gold context；历史单变量实验与多改动最终收口分开讲。

⚠️ 只做内部诊断、简历不写：

- RAGAS 四项（DeepSeek 自评，`independent_judge=false`）；
- Precision/NDCG（部分判定下界）；
- 任何"拒答检测导致 strict 提升 X"的表述（是测量修正）；
- 48 题 / 2 家公司 / 1 个年度的绝对分数，必须带边界声明；
- "96 变体 / OOV 实例"未经人工审核，不作为独立标注。

## 使用建议

- 简历正文：每阶段用一行量化句（上文的"简历一句话"）；
- 面试深聊：从对应阶段的实验记录里选 1–2 个"失败→定位→修复"的故事（如坐标重建从 58.6% 到 98.1%、LLM 改写 canonical lane 阴性结论、拒答检测的测量修正）；
- 所有数字引用时补一句数据边界："冻结集 48 题、两家公司 2024 年报、assistant-reviewed provisional"。
