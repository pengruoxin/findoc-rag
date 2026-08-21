# FinDocRAG 评测基线（更新于 2026-08-21）

> 本文档只写**数字**：当前规模、当前分数、当前薄弱点。每轮实验后更新此处，是所有优化对比的唯一权威来源。
> 规则（指标定义、数据集构造、门禁、评分策略）见 [benchmark-and-metrics-zh.md](./benchmark-and-metrics-zh.md)；改进计划见 [improvement-list-zh.md](./improvement-list-zh.md)。
> 看不懂的词见 [术语表](../glossary-zh.md)。

## 0. 当前统一口径

首页和后续报告按“主 Agent 集 → 通用 RAG 集 → PDF/表格专项 → 历史回归”排序，不再把
旧的 2 家公司、1 个年度 benchmark-v2 称为当前主基线。

| 层级 | 完整范围 | 当前已执行范围 | 当前结果 | Gold 状态 |
|---|---|---|---|---|
| 主 Agent：`agent-hard-v3` | 96 题、8 家公司、16 份 2023–2024 年报 | calibration + dev，48 题 | 严格自动通过率 94%；行为准确率 100% | 来源与页码已核对的 provisional gold；独立双审待完成 |
| 通用 RAG：`benchmark-v3` | 60 题、6 家公司、10 份 2023–2024 年报 | frozen test，24 题，离线确定性基线 | strict 19%；行为准确率 33%；未启用远程模型 | `independent_gold=false`；PDF 视觉复核待完成 |
| 真实扫描 PDF | 4 页、30 个视觉探针、17 个结构单元格 | 全部开发探针 | 自适应结构召回 100% | 开发集；正式独立复核 gold 完成率 0%（目标 70 个） |
| 历史回归：`benchmark-v2` | 48 题、2 家公司、2 份 2024 年报 | 全部旧集 | DeepSeek 三轨 strict/行为 100%；检索 Recall@5 81% | `independent_gold=false`；只用于回归 |

`agent-hard-v3` 的另外 48 题 frozen test 尚未运行，当前 94% 不能外推成 96 题全量成绩。
`benchmark-v3` frozen 的 19% 是扩大公司与年度覆盖后的离线起始基线，不是 DeepSeek 最终
成绩。对应产物分别为：

- `reports/agent/agent-hard-v3-calibration-deepseek-p4e-composed.json`；
- `reports/agent/agent-hard-v3-dev-deepseek-p4e-composed.json`；
- `reports/generation/benchmark-v3-frozen-run/retrieved_context-benchmark-v3-deterministic-p0-freeze-frozen_test/summary.json`；
- `reports/pdf-extraction/pdf-hard-v2-stage7-summary.json`。

以下第 1–6 节保留 benchmark-v2 的完整历史实验明细，用于解释旧索引和旧指标，不再代表
项目当前最大评测覆盖。

## 历史 benchmark-v2 版本钉（复现旧数字需要的三个版本号）

| 项 | 值 |
|---|---|
| 评测集 | `benchmark-v2`（派生自 `generation-eval-v1-b7f4d6113c96`） |
| 检索回归集 | holdout v2，16 题 |
| 冻结源索引身份 | `10fb50419145d56720c9`（历史制品已不可恢复，不伪造） |
| 当前正式目标索引 | `9898c95e13d01c51c156`（958 个切片，E5-small 384 维，经 migration 显式绑定） |
| 目标 snapshot | `c3f15772dcb8b44898e415e93eb27aec17d56a95190837366f41b500b2730bff` |
| 最终三轨代码指纹 | `5f02074f2fd4e08042a2c9a05123c83d199d325959e271178a7ff849f40aff06` |
| 切片格式版本 | 3 |

历史 `10fb...` 仍是冻结基线身份，不被覆盖。由于原 snapshot/manifest/embeddings 已不可恢复，已建立迁移候选 `9898c95e13d01c51c156`：snapshot `c3f157...`、同一 E5-small、384 维、958 chunks。迁移清单逐文件绑定 SHA，并证明 38/38 judged chunks 的文本、章节、页码、计数等语义核心完全一致；原问题、答案与指标未变。

## 1. 数据规模

> 背后是 2 份真实的 2024 年年报：贵州茅台 143 页、伊利股份 270 页。

| 项目 | 数值 |
|---|---:|
| 总题数（该答 / 该拒答 / 该追问） | 48（37 / 9 / 2） |
| 变体问句 | 96（股票代码 47、财务简称 26、相对时间 43、口语改写 96） |
| 原子事实 / 正确证据 / 涉及的切片 | 120 / 67 / 35 |
| 需要多段独立证据的题 / 跨页证据 | 13 / 26 |
| 干扰段落（5 类） | 53（错公司 15、错期间 14、错口径 15、证据不全 5、因果硬凑 4） |
| 需要算的事实（带公式） | 13 |
| 数据划分 | 调参 12 / 开发 24 / 冻结测试 12 |
| 公司覆盖 | 茅台 18、伊利 22、跨公司对比 8 |
| 题型 | 多事实表 11、对比 8、叙述 8、计算 7、单事实 2、政策 1、拒答 / 追问 11 |
| 文档来源 | 2 份 2024 年报（143 页 / 270 页，切成 958 个切片） |
| 随仓最小证据目录 | 38 个证据块（35 个 unique gold，覆盖全部 53 个 hard negatives） |

67 条正确证据全部翻回 PDF 原页人工看过，其中 26 条跨页；复核告警为 0。

## 2. 检索基线（holdout v2，16 题，取前 5 名，候选池 20）

> 当前默认策略：**只用关键词检索**。2026-08-07 从"混合 2:1"改过来，依据是 fusion-sweep-v1（见 §2.1 末条），真实检索赛道回归无退化。

| 检索方式 | 不过滤 Hit@5 / MRR@5 | 先按公司+年份过滤 Hit@5 / MRR@5 |
|---|---:|---:|
| 关键词检索（BM25） | 0.6875 / 0.4531 | 0.8125 / 0.6667 |
| 语义检索（E5-small） | 0.2500 / 0.0802 | 0.2500 / 0.0802 |
| 混合 2:1（旧默认） | 0.6875 / 0.3948 | 0.8125 / 0.4750 |

混合 2:1 的完整四指标（不过滤）：Recall@5 0.6875、Precision@5 0.1375、MRR@5 0.3948、NDCG@5 0.4666。

上下文预算（111 个问句，公司+年份过滤，`avg_evidence_tokens_top5`）：关键词 307、语义 386、混合 2:1 348。语义分支不仅准确率低，送进下游的证据量还更大——用更贵的上下文换更差的结果。

另有 10 条结构诊断题（开公司年份过滤 + 口径路由）：关键词 0.80 / 0.80，加自适应候选池 1.0 / 1.0。这组只作开发期诊断——标注和路由用了同一批结构线索，等于自己给自己出题，不作独立结论。

### 2.1 按问法类型分组（variant-regime-v1，111 个问句 / 37 道题）

同一道题用三种问法各考一遍，看系统对不同问法的能力差异。已开启公司+年份过滤，下表是 Hit@5（前 5 个结果里有没有正确证据）：

| 问法类型 | 例子 | 关键词检索 | 语义检索 | 混合 2:1 |
|---|---|---:|---:|---:|
| **原题**（照年报原文问） | 贵州茅台 2024 年营业收入是多少 | **0.838 → 0.892**（改写后） | 0.216 | 0.649 |
| **代码 / 简称** | 600519 2024 年营收是多少 | 0.811 → **0.892**（改写后） | **0.676** | 0.730 |
| **口语 / 相对时间** | 去年营收多少、毛利水平怎么样 | 0.730 → **0.919**（改写后） | 0.162 | 0.459 |

> 查询改写（[variant-regime-expanded-v2](../../reports/ranking/variant-regime-expanded-v2/analysis.md)）：从 10 条失败题提取 7 组财务同义词映射（营收→营业收入、毛利水平→毛利率、净资产回报率→净资产收益率、主要风险→可能面对的风险、前五大客户→前五名客户、同比增幅→比上年同期增减、一定实现→计划实现），命中后三档 Hit@5 全面提升、零回归；剩余 3 条 miss 归因：1 条评测时间对齐 bug（P2-15）、1 条排序差一名、1 条结构性。

完整六指标 × 两种过滤态见 [summary.md](../../reports/ranking/variant-regime-v1/summary.md)。四条结论都有逐题证据（见 [analysis.md](../../reports/ranking/variant-regime-v1/analysis.md)）：

- **语义检索只在"代码 / 简称"这类问法上有用**（0.676，其余两类只有 0.2 左右）。股票代码短、指向唯一，向量编码稳；问句一长或换了说法就迅速失效。说明主要影响因素是**问句的形态**，不是模型本身不行。
- **混合检索现在是负优化**：4 道口语问法的题，正确证据被关键词检索排在第 1–2 位，融合之后掉到第 6–7 位，跌出前 5。这 4 道题的语义检索连候选池都没捞到正确证据，等于只往融合里灌噪声。
- **口语问法是全员弱区**：37 道题里有 11 道三种检索方式全都没找到。主因是专业同义词改写——问的是"毛利水平"，年报里写的是"毛利率"；问"净资产回报率"，年报写"加权平均净资产收益率"；问"营收"，年报写"营业收入"。
- **权重扫描已经定论**（[fusion-sweep-v1](../../reports/ranking/fusion-sweep-v1/analysis.md)）：**只用关键词检索、完全不掺语义检索，在三种问法上都碾压任何融合权重**（Hit@5 0.838 / 0.811 / 0.730，MRR 0.694 / 0.724 / 0.633）。只要给语义路任何正权重都是拖累，逐问法挑最优也全都选"不掺"。结论是默认策略应改成纯关键词检索，等语义模型换掉之后再重新验证融合。

43 个相对时间问句全部正确解析成 2024。唯一的例外 `yili_2025_plan_bounded` 暴露了一个真实 bug：这题的提问时点设在 2026-04-30，"去年"解析成 2025 是对的，但系统直接拿 2025 去过滤年报年份，把正确证据排除了——那份证据是**2024 年报里写的 2025 年经营计划**。教训是**问题问的年份 ≠ 文档的报告年份**，中间需要一层时间对齐判断。

### 2.2 索引迁移候选（2026-08-13，`9898...`）

- 迁移门禁绑定：E5 revision `614241f...`、模型权重 SHA、dense embeddings、dense chunk IDs、BM25 SQLite、structured-table sidecar、snapshot 与 38 条逐块 payload/semantic-core SHA。
- 与历史 `10fb...` 的 expanded-v2 做 111 条正向实例配对：lexical + query-derived filter 的三类问法 Hit@5/MRR@5 **完全相同**，0 fixed / 0 regressed；109/111 全指标逐项相同，另 2 条仅因生产路由统一而改善跨 chunk Recall/NDCG。
- 新索引 raw → deterministic rewrite + 生产路由：Hit@5 分别 `0.838→0.892`、`0.811→0.892`、`0.730→0.919`；12 fixed / 0 Hit@5 regressions。MRR 有 14 提升 / 3 小幅降位，均未跌出 top-5。
- dev split 上按生产 deterministic rewrite 重跑 6 组权重：三个 regime 的最优仍全部为 lexical-only；dense/hybrid 保持实验能力，但不提升为默认。
- no-remote Retrieved 确定性诊断：strict `0.6000`、行为 `0.5000`、48/48 无运行错误；这是本地规则回归，不与 DeepSeek 历史主分混报。

证据：[迁移清单](../../data/evaluation/benchmark-v2-e5-migration-v1.json)、[历史→迁移配对](../../reports/ranking/paired-10fb-to-9898-lexical-v1/summary.md)、[迁移内改进配对](../../reports/ranking/paired-9898-improvement-v1/summary.md)、[融合扫描](../../reports/ranking/fusion-sweep-e5-migration-rewrite-v1/summary.md)。

### 2.3 词表外改写（OOV）与中文 dense 对照（2026-08-11）

OOV 集 36 实例 / 12 题，全部是 deterministic 词表外的口语改写（如"扣非净利润与净利润的差额""风险因素有哪些"），top_k=5 / candidate_k=20 / query_parser 过滤：

| rewrite | lexical Hit@5 / MRR@5 | dense Hit@5 | hybrid Hit@5 |
|---|---:|---:|---:|
| none | 0.194 / 0.148 | 0.139 | 0.222 |
| deterministic（7 组词表） | 0.194 / 0.148 | 0.139 | 0.222 |
| **LLM 改写** | **0.694 / 0.498** | 0.167 | 0.472 |

中文 dense 对照（bge-small-zh-v1.5，索引 `6a951f4e8b7bd913d918`）：OOV dense 0.083，LLM 改写后仍 0.083；变体三问法 0.162 / 0.568 / 0.162，全部低于 E5（0.216 / 0.703 / 0.189）。**结论：换中文小 dense 模型没有收益，问题在问句形态与上游表格线性化；LLM 查询改写是 OOV 的决定性杠杆，但改写需持久化缓存 + 确定性词表兜底**（剩余 miss 归因见 [oov-eval-llm-v1 分析](../../reports/ranking/oov-eval-llm-v1/analysis.md)）。

## 3. 生成评测（DeepSeek 主结果 + no-LLM 确定性诊断）

### 3.1 DeepSeek 真实基线（2026-08-07，`deepseek-chat`，48 题）——主基线

三条赛道分别给不同上下文，用来定位错误出在哪一层（赛道定义见 [术语表](../glossary-zh.md#三条评测赛道)）：

| 赛道 | 题数 | 全对率（可计分） | 行为准确率 | 平均上下文 token | p95 延迟 | RAGAS（faith/rel/ctx-rel/ctx-rec） | 报错 |
|---|---:|---:|---:|---:|---:|---|---:|
| **直接给答案**（生成上限） | 48 | **0.9714**（35） | 1.0000 | 303 | 1668ms | 0.68 / 0.98 / 1.00 / 0.99 | 0 |
| **真实检索**（端到端） | 48 | **0.5429**（35） | 0.8333 | 1536 | 1843ms | 0.84 / 0.73 / 0.93 / 0.82 | 0 |
| **抗干扰** | 29 | **0.5455**（22） | 0.7931 | 784 | 2441ms | 0.68 / 0.71 / 0.94 / 1.00 | 0 |

"全对率"要求一道题的所有原子事实全部正确才算过，所以分数看着低；括号里是符合计分条件的题数（叙述类答案交给语义评测，不进这个指标）。

解读：

- **直接给答案 97% 全对**：只要证据给对了，DeepSeek 基本都能答对——生成不是瓶颈。
- **真实检索 54%**：与直接给答案差 43 个百分点，差距全部来自检索 / 路由 / 证据选择——这就是检索侧优化（同义词、路由、表格结构化）的目标空间。
- **抗干扰 55%**：gold 保证在上下文里但混入真实干扰后掉 43 个百分点——模型会被高相似干扰带偏，行为层的拒答 / 澄清还有很大空间。
- RAGAS 为同模型自评（`independent_judge=false`），只作语义诊断，不作为对外主张。

**查询改写后 retrieved strict 仍为 0.5429（持平）**。逐题归因 16 个 strict 失败：行为拒答（该拒没拒）8 题、表格 / 生成（gold 在 top5 仍答错）5 题、检索（gold 不在 top5）3 题——检索指标的提升尚未传导到端到端，下一步瓶颈是行为拒答与表格抽取。

### 3.2 no-LLM 确定性链路（回归对照）

同 48 题、clarify-v1 策略，只验证评测闭环与门禁：直接给答案 0.3143 / 真实检索 0.0857 / 抗干扰 0.2273（strict）；行为 1.0 / 0.8333 / 0.7931。确定性链路的表格正则抽取明显弱于 DeepSeek 直接读表，因此 strict 远低于真实模型。

#### 3.2.1 表格确定性回答接入后（2026-08-11，`no-llm-table-v1`）

把 `extract_cells` 接入 answer_generation 的确定性表格路径（季度 / 附注收入成本 / 分部毛利率 / 年度营业收入 + 跨公司 / 合并母公司口径，带 `[n]` 引用），同一 dataset 逐题配对：

| Lane | clarify-v1 基线 | table-v1 | Δ | 修复题数 |
|---|---:|---:|---:|---:|
| Oracle | 0.3143 | **0.6571** | +0.3428 | 12 / 0 回归 |
| Retrieved | 0.0857 | **0.2571** | +0.1714 | 6 / 0 回归 |
| Robustness | 0.2273 | **0.3636** | +0.1364 | 3 / 0 回归 |

Oracle 12 题全部命中预期：季度现金流 / 季度归母 / 两个季度-年度核对 / 附注成本 / 两个成本核对 / 产品毛利率（茅台 / 伊利）/ 渠道毛利率差 / 年度营收同比 / 跨公司营收差 / 合并-母公司营收差。Retrieved 修复 6 题，剩余 6 题卡在检索证据不在 top-5。行为准确率不变（1.0 / 0.8333 / 0.7931），零回归。配对报告：`reports/generation/comparisons/{oracle,retrieved,robustness}-table-v1`。

#### 3.2.2 DeepSeek 三轨重跑（2026-08-11，`deepseek-chat-table-v2`，真实 api_model 已落盘）

| Lane | 8/7 基线 strict | 重跑 strict | Δ | 配对（修复/回归） |
|---|---:|---:|---:|---:|
| Oracle | 0.9714 | 0.9714 | 0.0000 | +moutai_annual_deducted_profit / -yili_quarterly_profit_reconcile（单题互换） |
| Retrieved | 0.5429 | **0.5714** | +0.0286 | +2 / -1（含单题波动） |
| Robustness | 0.5455 | **0.6364** | +0.0909 | +2 / -0 |

行为准确率不变（1.0 / 0.8333 / 0.7931）。RAGAS（`independent_judge=false`，元数据已修正）：

| Lane | Faithfulness | Answer Relevancy | Context Relevance | Context Recall |
|---|---:|---:|---:|---:|
| Oracle | 0.778（旧 0.682） | 0.977 | 1.000 | 0.973 |
| Retrieved | 0.845（旧 0.836） | 0.845（旧 0.733） | 0.973（旧 0.932） | 0.892（旧 0.824） |
| Robustness | 0.849（旧 0.680） | 0.765（旧 0.708） | 0.944 | 1.000 |

说明：DeepSeek 同模型自评，只作语义诊断；单题级存在 ±1 波动（同温度下 API 非确定性），结论看趋势而非单题。修复 `api_model` 元数据记录（`--model` 只是 run 标签），新产物 `api_model_recorded=true`、`independent_judge=false`。

> ⚠️ 本表与 8/7 基线的对比**不是受控实验**：8/7 基线跑在同义词改写落地之前的 runner 上，之后还叠加了 `api_model` 元数据修正。按[实验协议](./experiment-protocol-zh.md)，这组数字只能作趋势参考，不能作为单变量结论；受控对比见 §3.2.1（功能开关 A/B）与 §3.2.3（单一测量变更）。

#### 3.2.3 拒答检测修正后的真实基线（2026-08-11，`deepseek-chat-abstain-v2`）

问题：远程回答无论内容是否拒答，`grounded` 一律为 True，导致"无法回答……数字……"这种自带数字的拒答被算成正确回答（如 yili_concentration），且 9 条应拒答题的拒答永远计为"answer"→ 行为与 strict 双输。修复：`_is_abstention` 检测明确拒答措辞（无法/不能+回答/确认/给出等、证据不足等），命中则 `grounded=false`、provider=`remote-abstention`。

| Lane | strict（修正后） | 行为准确率（修正后） | 相对 table-v2 的严格变化 |
|---|---:|---:|---:|
| Oracle | 0.9429（v1 曾 1.0，波动） | 0.9792 | +1 / -2（可答题拒答被如实计分） |
| Retrieved | **0.8000** | **0.8958** | **+8 应拒答修复 / -0** |
| Robustness | **0.8636** | **0.8276** | +6 应拒答修复 / -1（yili_concentration 打分怪癖消除） |

解读：

- strict 的大幅上升主要是**计分修正**：8 条应拒答题的拒答此前被误计为"answer"（0 分），现在如实计为 abstain（满分）。不是检索/生成能力突变。
- 行为准确率上升同理；同时可答题的拒答（每轨 4–5 条，如 moutai_product_margin、yili_consolidated_parent_revenue）现在如实计为行为失败——这是更诚实的数字，也是下一步"行为拒答策略"要修的样本。
- Oracle 出现 0.94–1.0 波动：取决于当次运行模型是否对个别可答题拒答（单题级 API 非确定性）。
- RAGAS（`api_model_recorded=true`、`independent_judge=false`）：Oracle faithfulness 0.796 / Retrieved 0.887 / Robustness 0.834。

#### 3.2.4 远程确定性表格优先（2026-08-11，`deepseek-chat-table-remote-v1`）

受控实验：单变量为 `FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES=1`（远程模式下表格题优先走确定性抽取，不再交给 DeepSeek）；dataset / index / api_model / runner 均与 abstain-v2 一致，代码版本 `640ab99`（工作区脏）。

| Lane | strict（abstain-v2 → 新） | 行为（abstain-v2 → 新） | 配对 |
|---|---:|---:|---:|
| Oracle | 0.9429 → **1.0000** | 0.9792 → **1.0000** | +2 / -0（其中 moutai_annual_deducted_profit 为模型波动，yili_consolidated_parent_revenue 为确定性修复） |
| Retrieved | 0.8000 → **0.8286** | 0.8958 → **0.9583** | +1 / -0（strict）；+3 / -0（行为：moutai_quarterly_cashflow、yili_consolidated_parent_revenue 确定性修复 + yili_2025_plan_bounded 模型波动） |
| Robustness | 0.8636 → 0.8636 | 0.8276 → **0.8621** | +0 / -0（strict）；+1 / -0（行为：yili_consolidated_parent_revenue） |

解读：

- 确定性表格路径在远程模式下的效果符合预期：**表格类可答题的误拒答被消除**（moutai_quarterly_cashflow、yili_consolidated_parent_revenue 三轨全中），零回归；
- 剩余行为失败集中在**未被四类抽取器覆盖的表格**（moutai_concentration / yili_concentration 等）与个别模型波动；
- 新 run 已带 `code_revision` / `code_dirty`；RAGAS：Oracle faithfulness 0.783 / Retrieved 0.868 / Robustness 0.813。

#### 3.2.5 concentration 表型抽取器（2026-08-11，`deepseek-chat-concentration-v2`）

新增第五类表型：前五名客户/供应商集中度（句子式披露，正则抽取金额与占比）。table-eval-concentration-v1：**8/8 单元格**（茅台 + 伊利，含"关联方第二次占比"不误取）。生成链路支持单公司与跨公司客户/供应商集中度对比（带引用）。

受控实验（单变量：新增 concentration 表型 + 修复"单公司查询按查询公司选题"）：基线 table-remote-v1（同 dataset / index / api_model / runner）：

| Lane | strict（前 → 后） | 行为（前 → 后） | 配对 |
|---|---:|---:|---:|
| Oracle | 1.0 → 1.0 | 1.0 → 1.0 | 持平 |
| Retrieved | 0.8286 → 0.8286 | 0.9583 → 0.9375 | strict 持平；行为 -1 为模型波动（yili_2025_plan_bounded，与 concentration 无关） |
| Robustness | 0.8636 → **0.9545** | 0.8621 → **0.9655** | **+2 strict（moutai_concentration、yili_concentration）/+3 行为**，零回归 |

过程中发现并修复一个真 bug：Robustness 负例前置时，单公司 concentration 题可能取到另一家公司的值（v1 run 保留为历史记录）；修复后按查询中的公司显式选题。RAGAS（clean revision `f8a1be8`、`code_dirty=false`）：Oracle faithfulness 0.773 / Retrieved 0.804 / Robustness 0.731（robustness answer_relevancy 0.909，较前显著提升）。

#### 3.2.6 LLM 查询改写 retrieved lane 实验（2026-08-11，`deepseek-chat-rewrite-llm-v1`，**阴性结果，不落地**）

受控实验：单变量为 retrieved lane 改写模式 deterministic → LLM（含持久化缓存 `rewrites.json`）；其余固定。结论：

- **端到端无净收益**：Retrieved strict 0.8286 → 0.8286（+1/-1）、行为 0.9375 → 0.8958（-2，均为无关模型波动）；Oracle/Robustness 持平。
- **证据层更清楚**：改写改变了 38/48 条的 top-5，但三个遗留检索 miss（revenue_cross_company、moutai_product_margin、yili_quarterly_cashflow_reconcile）**仍未修好**，反而新增 3 个证据回归（moutai_revenue_yoy、moutai_disclosed_risks、yili_disclosed_risks 的 gold 掉出 top-5）。
- **机制**：LLM 把"同比增幅"改写为"同比增长率"，覆盖了确定性词表里专门从失败案例提炼的"同比增幅→比上年同期增减"，与年报措辞反而更远；RAGAS context relevance 0.973 → 0.892、context recall 0.901 → 0.811。
- **为什么与 OOV 实验（0.194→0.694）不矛盾**：OOV 是词表外口语，LLM 归一化收益大；benchmark canonical 问句已接近年报措辞，通用改写反而破坏精选映射；剩余 miss 是结构性问题（跨公司双表共现、segment chunk 排序），不是同义词问题。
- **决策**：retrieved lane 默认保持 deterministic 词表；LLM 改写仅用于生产 `/v1/query` 的用户自由文本，且必须叠加"改写后确定性兜底 + 检索质量门控"再上线（待做）。

DeepSeek 与 RAGAS 语义评测见 §3.1（8/7 基线）与 §3.2.2（2026-08-11 表格路径重跑，均 `independent_judge=false` 自评）。旧 32 题的 0.9583 只作历史记录——那是另一个版本的数据集，不能搬过来用。

### 3.3 Index-bound 最终三轨（2026-08-14，`deepseek-index-bound-final`）——当前主结果

三轨共享同一个 migration `benchmark-v2-to-e5-c3f157-v1`、目标索引 `9898c95e13d01c51c156`、代码指纹 `5f02074f...aff06`、结构证据路由 `structured-table-v1` 和候选池策略 `scope-adaptive-20-to-100`。Oracle、Retrieved、Robustness 都通过 migration 从目标索引解析带 `statement_scope` / `structured_tables` 的 chunk；后两轨不再读取裸 evidence JSONL。所有远程调用均成功，`run_error_rate=0`。

| Lane | items | strict（eligible） | 行为准确率 | 平均上下文 token | p95 | 远程错误率 |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 48 | **1.0000（35）** | **1.0000** | 303 | 1.63s | 0 |
| Retrieved | 48 | **1.0000（35）** | **1.0000** | 1533 | 2.08s | 0 |
| Robustness | 29 | **1.0000（22）** | **1.0000** | 784 | 2.23s | 0 |

strict 只覆盖有确定性数值/单位评分资格的 35 / 35 / 22 题；语义叙述题不进入 strict 分母，禁止对外写成“48 条数值题全部 strict”。Retrieved 的独立确定性上下文检查为 **37/37 gold context 完整，Context Recall=1.0**。

RAGAS 完整覆盖结果：

| Lane | Faithfulness | Answer Relevancy | Context Relevance | Context Recall |
|---|---:|---:|---:|---:|
| Oracle | 0.7612 | 0.9077 | 1.0000 | 0.9865 |
| Retrieved | 0.8843 | 0.9066 | 1.0000 | 1.0000 |
| Robustness | 0.7963 | 0.8938 | 0.9444 | 1.0000 |

每个 metric 的 coverage 与 complete-row coverage 均为 100%，behavior mismatch 为空；但回答模型和 judge 都是 DeepSeek，产物明确记录 `api_model_recorded=true`、`independent_judge=false`，因此这些分数只作自评语义诊断，不是独立裁判结论。

相对各 lane 历史最佳配对：Retrieved strict 0.8286→1.0000（6 fixed / 0 regressed）、行为 0.9583→1.0000；Robustness strict 0.9545→1.0000（1 fixed / 0 regressed）、行为 0.9655→1.0000；Oracle 双 1.0000 持平。该对比同时包含 E5 migration、生产元数据/预测年份路由、index-bound sidecar、结构证据路由、自适应候选池、财务勾稽和 scorer/runner 修复，并受 DeepSeek 随机性影响，**不是严格单变量因果实验**。较早的 table-remote Robustness 0.8636→1.0000 配对保留作阶段审计，但不再称为“历史最佳”。

`runs-e5-migration-remote-v1` 中曾出现 Oracle `run_error_rate=0.4375`；这是沙箱网络权限失败造成的无效运行，runner 修复后已保留审计产物并以失败状态退出，不纳入任何能力结论。

## 4. 当前薄弱点（优化起点）

按杠杆排序，每条都指向已落盘的实验证据：

| 层 | 观察 | 证据 |
|---|---|---|
| **表格抽取** | 五类表型确定性抽取已通过 index-bound sidecar 接入在线回答：主表 146/149、集中度 8/8；真实两份年报自动发现 15 表 / 195 cells，IR v2 下 12 表走坐标、3 表因安全门禁回退文本。sidecar 不进入 chunk serialization，启动时校验 schema/generator/index/source/content/chunk SHA；无 sidecar 的旧索引继续使用文本 fallback | table-eval-v10-safe + coordinate-safe-v10 + sidecar contract tests |
| **复杂 PDF / 坐标表格** | 文本主表 146/149、集中度 8/8；safe coordinate 为 154/157、157 predictions，PDF 直读与持久化 IR v2 完全一致。3 格残差来自页面/文字层为“其他”而 gold 为“其他地区”，不做题目特判 | coordinate-safe-v10-pdf / coordinate-safe-v10-ir |
| **同义词** | 词表内改写（7 组映射）把口语问法 Hit@5 提到 0.92；词表外改写原为 0.194，LLM 改写提到 0.694；剩余 miss 主要是简称未归一（扣非→扣除非经常性损益）、行内换行和文档措辞未知 | OOV 36 实例：none / deterministic 0.194 → LLM 0.694；剩余 11 个 miss 归因见 [oov-eval-llm-v1 分析](../../reports/ranking/oov-eval-llm-v1/analysis.md) |
| **时间对齐** | 已把事实期间、报告年份和预测目标年份分离；2025 经营目标不再错误过滤到 2025 年报。剩余风险是当前路由评测仅 18 条、单一语料年度 | `yili_2025_plan_bounded` runner/API 回归；query-routing-v1 18/18 |
| ~~融合~~ | 已定论：混合检索确实是负优化，默认已改成纯关键词检索 | fusion-sweep-v1，见 §2.1 |
| 语义检索 | 只在"股票代码 / 简称"这类短问句上有用；**换中文专用小模型（bge-small-zh-v1.5）全面退化**，证明主因是问句形态 + **上游 PDF 表格线性化**（数字密集无结构文本放大表面误导）。决策：暂用关键词检索 + LLM 查询改写，表格结构化后与更强模型（如 bge-m3）一起重验 | E5 0.216 / 0.703 / 0.189 vs bge-zh 0.162 / 0.568 / 0.162；OOV dense 0.083 |
| 查询路由 | 已支持全名、别名、股票代码、相对时间、事实期间/预测目标年份分离，并在 `/v1/query` 返回 route 与 applied filters；当前路由小集 18/18，薄弱点是公司/年度覆盖太窄 | query-routing-v1；Agent contract tests |
| **Agent 文档盲测** | P1 定向小集 8/8；5 份未见年报的 34 题冻结基线严格 17/34、来源复核 24/34。P2-B1 只修文档年份/事实期间规划后严格 21/34（4 fixed / 0 regressed）、来源复核 28/34，行为和安全拒答 100%；剩余 4 个计算未支持和 2 个多事实遗漏 | `agent-hard-v2-p2b1-summary.json`；[P2-A](./agent-p2a-document-blind-baseline-zh.md)；[P2-B1](./agent-p2b1-document-scope-log-zh.md) |
| 生成 | 最终三轨 strict / 行为均 1.0，错误率 0，Retrieved 37/37 gold context；RAGAS 完整覆盖但为 DeepSeek 自评。当前薄弱点已从冻结集正确性转为外部有效性、独立裁判和更难行为覆盖 | §3.3；`runs-e5-migration-remote-final`；`ragas-index-bound-final-*` |
| 行为 | 拒答 / 追问只有 11 题，且偏"因果推断、投资建议"这类明显该拒的，缺"数字确实存在但口径或期间不对"的近失拒答 | |
| 覆盖 | 37 道该答的题里只有 13 道需要拼多段证据，跨段推理的覆盖偏低 | |

PDF 解析的源元素覆盖率 100%，958 个切片，切片长度中位数约 238 token。IR v2 已持久化 line/span 坐标、字体、字号、粗体与旋转信息，并有 OCR/低文本/乱码/图片页质量报告；表格结构已具备确定性文本与坐标双路径，但尚未覆盖扫描版年报、多公司和多年度盲测。

## 5. 迭代协议

每轮优化固定流程：

1. 从 §4 选定一个薄弱点，一次只动一层（**控制变量纪律见 [实验协议](./experiment-protocol-zh.md)**）
2. 改代码 + 补单元测试
3. 跑完整矩阵：检索 3 种方式 × 2 种过滤态 + 生成 3 条赛道 + 语义评测；变体按问法类型分组报告
4. 与本文档的基线做逐题配对对比（哪些修好了、哪些退化了），更新 §2 §3 表格；配对报告必须记录 `code_revision` 与 `controlled_change`
5. 记录到实验注册表和 [变更日志](../history/optimization-log-zh.md)；结论必须带数据集 / 索引 / 代码版本 / 配置四重身份，否则无法复现

每步最低验收：单元测试覆盖新增逻辑；Ruff 通过；跑回归并记录指标变化；至少记录一个未解决问题或潜在退化——只报成功的记录没有参考价值。

优先级：评测基础设施（已完成）→ 问法分组与失败归因 → 行为 / 多轮 / 表格 → 对外公信力（多公司多年度、整份文档留出盲测、人工复核、独立打分模型、CI）。

## 6. 当前待办

> 完整清单见 [improvement-list-zh.md](./improvement-list-zh.md)。

- [x] 评测集定版（`benchmark-v2.json`，含 96 个变体问句、赛道资格标记、检索视图）
- [x] 报告校验器（`scripts/validate_eval_report.py`，必须记录过滤条件来源，一票否决）
- [x] 旧 holdout 迁移清单（16 条：2 条精确映射，14 条原文已恢复但排除在正式成绩之外）
- [x] 数据集完整性门禁（`scripts/validate_benchmark_dataset.py`，一票否决，已接入启动检查）
- [x] 切片格式版本绑定（当前 3）；配置枚举与默认权重已与代码对齐
- [x] 评测脚本支持注入提问时点和跑变体问句（`--variant`）
- [x] 变体首轮检索评测 + 融合权重扫描（variant-regime-v1 / fusion-sweep-v1；结论：纯关键词检索全面优于融合，默认策略已改）
- [ ] 96 个变体问句的语义保真人工审核（现在全是助手生成，没有逐条审核状态）
- [x] 48 题 DeepSeek 三赛道 + 语义评测已有历史正式基线和表格改造配对结果
- [x] `10fb...` 不可恢复后建立显式 `9898...` migration manifest；迁移检索矩阵与 paired review 已完成，旧 benchmark 身份未改写
- [x] 在迁移索引上完成同 migration / index / code fingerprint 的 DeepSeek 三轨与 RAGAS，远程错误率 0、coverage 100%
- [x] 建立 5 份 Agent 未见年报、34 题的 `agent-hard-v2` 候选盲测并冻结首次 DeepSeek 运行
- [ ] 给 `agent-hard-v2` 增加第二人 gold/语义复核、真实扫描年报和文档聚类置信区间后再升级为对外正式集
- [ ] 引入不同 provider 的独立 judge

## 7. 面试怎么讲这套评测集

### 一句话

> 这不是"能答对几题"的题库，而是一套能告诉你"系统到底弱在哪一层"的评测体系。

### 30 秒讲法

中文年报里同一个数字会出现很多次——年度汇总、季度表、附注、分部报告里都有，光靠关键词根本分不清该用哪个。所以我们的评测集做了四件事：用真实用户的问法出题（股票代码、财务简称、"去年"）、用真实年报段落当干扰、把答案拆到数字和单位逐项核对、分三条赛道回答"问题到底出在检索、表格还是生成"。

### 五个可辩护的点

**1. 同一道题用三种真实问法考。** 真实用户不会照着年报原文提问。48 题各配 2 个变体共 96 种问法：报股票代码（"贵州茅台 2024 年营业收入"→"600519 2024 年营收"，47 题）、用财务简称（"经营活动现金流量净额"→"经营现金流净额"，16 题）、说相对时间（"2024 年"→"去年"，43 题；提问时点冻结在 2025-04-30，所以"去年"永远=2024，不用系统当前日期解析，否则评测会随运行日期漂移）、口语改写（全部 96 个）。营收=营业收入、600519=贵州茅台这些都要靠语义理解，纯关键词会失灵。可以分别测"照原文问"和"换种说法问"的成绩，量化对专业表达的真实鲁棒性。

**2. 干扰项从真实年报里挑，不是编的。** 53 个干扰段落全部来自那两份年报：茅台的问题里混进伊利的营收、季度数字冒充年度数字、合并口径冒充母公司口径。干扰项和正确答案同版式、同数量级，唯一区别是用错了地方。专门测**近失拒答**——资料里确实有数字但答的不是你问的，系统敢不敢说"我不确定"。

**3. 三条赛道把错误定位到层。** 早期 no-LLM 基线中，直接给正确证据只有 0.31、走真实检索只有 0.09；接入确定性表格后分别提升到 0.6571 / 0.2571，证明表格能力带来的是可配对的真实增益。远程 DeepSeek 主基线另账记录，不能和确定性诊断混报。

**4. 答案拆到原子事实，逐项核对。** 37 道该答的题拆成 120 个原子事实，每个事实的数值、单位、期间、口径全对，而且要能在证据原文里找到（有页码、有引用）。13 条要算的事实带公式（四季度加总=全年、跨公司差额、扣非勾稽），不是"这个数字在文里出现过"就算对。单位错了（元 vs 万元 vs 亿元）直接判错。

**5. 评测自己有防作弊设计。** 用正确答案帮忙过滤的报告直接判无效；同一考点的题不跨数据划分；来源文件 SHA-256 锁定；同一道题的多种问法先聚合再平均，防止虚高样本量；没有全量标注的指标明确标"部分判定"。面试官最担心的"评测是不是自己编的、是不是泄题了"，这里有工程化答案。

### 边界要主动说

本节讲的是历史 benchmark-v2，因此只有两家公司、一个年度；标准答案是助手自审的
（`independent_gold: false`），没有第二人独立复核。它只用于历史回归，不代表当前最大
评测覆盖，也不支持通用 SOTA 或跨文档泛化主张。
