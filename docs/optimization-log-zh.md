# FinDocRAG 优化实验记录

本文件用于记录每次工程改动的可复现对比。任何优化必须同时记录代码变更、测试、评测指标和已知退化。

## 记录模板

```text
日期：
目标问题：
修改前基线：
修改内容：
测试命令：
测试结果：
评测结果变化：
新增能力：
已知退化/未覆盖：
结论：
```

## 2026-08-06：P0 回答可信度修复

### 1. UTF-8 编码修复

- 修改前基线：`answer_generation.py` 的拒答文案、季度指标名和 DeepSeek prompt 出现乱码；原有季度测试使用乱码文本，无法证明中文链路正常。
- 修改内容：以 UTF-8 重写回答生成模块和测试样例，保留 extractive、deterministic-table、DeepSeek 三种 provider。
- 测试结果：季度抽取测试通过；Ruff 通过。
- 评测结果变化：从“中文输出不可用/无法验证”变为可读中文和可执行回归测试。
- 未覆盖：历史索引中的乱码文本需要重新解析后才能彻底消除。

### 2. 公司/年份 metadata routing

- 修改前基线：`/v1/query` 只有用户显式传 filters 时才过滤，存在跨公司、跨年份召回风险。
- 修改内容：自动识别贵州茅台、伊利股份和 `20xx` 年份；显式 filters 优先。
- 测试结果：通过回答生成测试；API 全量测试需在 uv trampoline 环境问题修复后重跑。
- 评测结果变化：增加了查询级路由约束；尚未宣称 Hit@K 提升，需使用同一 holdout 重跑对比。
- 未覆盖：公司别名、简称和多公司联合问题。

### 3. 证据支持门禁

- 修改前基线：远程 LLM HTTP 成功即标记 `grounded=true`，不检查引用、公司、年份和指标。
- 修改内容：回答前校验公司、年份及营业收入/营业成本/现金流/净利润；远程回答必须包含 `[1]`、`[2]` 或 `[3]` 引用，否则拒答。
- 测试结果：`tests/test_answer_generation.py`：3 passed；Ruff：All checks passed。
- 评测结果变化：增加了“错误证据拒答”和“无引用拒答”能力；准确率和拒答率需要在 holdout 重跑后记录。
- 未覆盖：回答中的具体数字与证据数字的逐项比对。

## 后续每步的最低验收要求

1. 单元测试覆盖新增逻辑。
2. Ruff 静态检查通过。
3. 相关 holdout 或真实问题回归，记录 Hit@5/MRR@5、拒答率或延迟变化。
4. 至少记录一个未解决问题或潜在退化，避免只保留成功案例。

## 2026-08-06：单路检索与 RRF Hybrid 对比

- 目标：回答“BM25、multilingual-e5 和 RRF 融合相对单独 BM25 的收益”。
- 控制变量：同一 index、16 条 holdout v2、top-k=5、candidate-k=20，关闭 metadata、scope、adaptive 和 reranker。
- 结果：BM25 Hit@5 0.6875/MRR 0.4531；Dense 0.2500/0.0802；RRF Hybrid 0.6250/0.3917。
- 结论：Hybrid 相对 BM25 的 Hit@5 为 -0.0625，MRR 为 -0.0614；当前没有提升。主要风险是弱 dense 分支通过等权 RRF 稀释强 BM25 信号。
- 报告：`reports/ranking/retrieval-comparison-v3.md`。

### BM25 加权 RRF（2:1）

- 修改：RRF 支持可配置 `lexical_weight` 和 `dense_weight`，默认 2.0 与 1.0；API service 使用配置传入权重。
- 测试：Ruff 通过；index/config/diagnostics 共 14 passed。
- 结果：加权 Hybrid Hit@5 0.6875、MRR 0.3948。
- 相对等权 Hybrid：Hit@5 +0.0625，MRR +0.0031。
- 相对 BM25：Hit@5 持平，MRR -0.0583。
- 结论：2:1 权重消除了等权融合的 Hit@5 退化，但尚未改善首个相关结果排名；下一步应做权重 sweep，而不是直接宣称 Hybrid 已优于 BM25。

## 2026-08-06：双层 RAG 评估体系

- 检索侧新增逐题和汇总 Precision@K、Recall@K、NDCG@K，保留 Hit@K、MRR@K 和 candidate recall。
- 加权 Hybrid 实测：Recall@5 0.6875、Precision@5 0.1375、MRR@5 0.3948、NDCG@5 0.4666。
- 生成侧新增 Faithfulness、Answer Relevancy、Context Relevancy、Context Recall 数据契约及聚合器，强制保存 human/LLM judge 来源；LLM judge 必须记录模型。
- 验证：相关测试 5 passed；Ruff 通过。
- 边界：尚无人工复核生成集，因此不伪造四项生成指标分数。

## 2026-08-06：生成评测集 v1

- 建立 32 条 source-verified 生成回归集：24 可回答、7 应拒答、1 应澄清，共 72 个原子事实和 39 组 evidence。
- 按 calibration/dev/frozen_test 分组；同一 family 和 gold chunk 不跨 split。
- 所有 39 组 evidence 已回到原始 PDF 页视觉复核，validator 警告为 0。
- 新增确定性 scorer：财务数字规范化、fact recall、单位、引用有效性、context recall、严格成功和拒答正确性。
- 导出 24 条 RAGAS Oracle Context 样本；evaluation extra 固定 RAGAS 0.3.1 与兼容 LangChain 0.3 系列，真实 import 已验证。
- W&B 默认为 disabled，可切换 offline/online 记录 dataset artifact、配置和四项 RAGAS 指标。
- 验证：generation evaluation 定向测试 6 passed；首次全量测试 53 passed；Ruff 全仓检查通过。

### 首轮无 LLM 基线发现的问题

- 初始 Oracle run：strict success 0.2188、expected behavior accuracy 0.2500、error rate 0；暴露注册表原始 chunk 缺公司/年份 metadata，正确 evidence 被 guardrail 误拒。
- 为 Oracle gold chunk 注入已核验 company/year 后，expected behavior accuracy 提升到 0.9375（+0.6875）。
- 进一步修复 PDF 在“经营活动产生的现金流量净额”内部换行导致的证据门禁/季度抽取失败，expected behavior accuracy 提升到 0.9688（+0.0313）；剩余 1 条是系统尚无 clarify 行为。
- 新增 metric 内换行回归测试；answer generation 当前 4 passed。
- 修复后全量回归：54 passed；Ruff 全仓检查通过。
- 当前 Codex 进程未继承用户终端中的 `DEEPSEEK_API_KEY`，因此没有伪造 DeepSeek/RAGAS 分数；真实 Oracle/Retrieved 生成基线需在包含 token 的同一环境运行。

### DeepSeek Oracle Context 基线

- 32 条全部完成，API error rate 0。
- 初版 scorer：strict success 0.8438、expected behavior accuracy 0.9688、numeric accuracy 1.0000、unit accuracy 1.0000、citation validity 1.0000、deterministic context recall 1.0000。
- 逐条复核发现四个 narrative “失败”实际语义正确，是完整字符串匹配造成的评分器假失败。
- scorer v2 将叙述事实移交 RAGAS/人工语义复核，确定性 strict gate 只汇总有资格的数字/单位/引用/拒答样本：eligible 24，strict success 0.9583；8 条需要语义复核；行为准确率仍为 0.9688。
- 唯一明确行为失败：模糊问题“茅台利润是多少”应该澄清年份和利润口径，当前系统只会拒答。

## 2026-08-06：BM25 财务科目整词增强

- 修改前基线：中文仅使用单字/二元切分；“营业收入”“经营活动产生的现金流量净额”等长会计科目会被拆散，精确科目匹配信号不足。
- 修改内容：在 tokenizer 中保留高价值财务科目整词，同时保留原有二元 token，兼容旧索引查询逻辑。
- 测试命令：`uv run ruff check src/findoc_rag/indexing.py tests/test_indexing.py`；`python -m pytest tests/test_indexing.py -q`。
- 测试结果：Ruff 通过；索引测试 9 passed。
- 评测结果变化：本次改动会改变新建 lexical index 的 token 统计；旧索引未重建前不宣称 Hit@5/MRR 提升。下一步重建同一 holdout index，记录 BM25 baseline 与 term-enhanced 的差值。
- 潜在退化：词表目前是有限的财务科目集合，后续需要从真实失败案例扩充并防止过拟合 holdout。

### 实际 holdout 对比结果

- 已使用新 tokenizer 重建 corpus lexical index：index `b384205bc3a39550f64d`。
- 同一 holdout v2、16 条问题、top-k=5、candidate-k=20、关闭 metadata/scope/adaptive 条件下：
  - baseline BM25：Hit@5 0.6875，candidate recall 0.6875，MRR@5 0.4531。
  - term-enhanced BM25：Hit@5 0.6875，candidate recall 0.6875，MRR@5 0.4531。
- 结论：本次改动没有可观测提升，也没有退化。说明当前 holdout 的查询已有足够二元词信号，整词 token 尚未改变排序；下一步应转向失败案例驱动的字段权重或查询改写。
- 完整对比：[holdout-eval-v2-bm25-term-enhanced-comparison.md](../reports/ranking/holdout-eval-v2-bm25-term-enhanced-comparison.md)。

## 2026-08-06：生成评测深度与 Robustness 闭环

### 修改前基线

- 数据集 32 条：24 可回答、7 应拒答、1 应澄清，72 个原子事实、39 条 evidence。
- `tracks` 虽声明 robustness，但 hard negative 为 0，runner 只支持 Oracle/Retrieved。
- 17/36 条扩充后可回答题曾存在“绑定 quote 不包含全部直接数值事实”的风险；validator 只能验证 quote 是 chunk 子串，不能验证语义契约。
- 同一 chunk 拆成多个 quote 时，`required_citation_count` 错按 span 数计算。
- 季度利润核对题参考答案只覆盖 2/5 required facts。
- 生成器最多使用 3 个上下文，无法容纳 4 个关键审计事项 gold chunk。

### 修改内容

- 数据集扩展并重审为 48 条：37 可回答、9 应拒答、2 应澄清，40 个 family、120 个原子事实、67 条 evidence、35 个唯一 gold chunk。
- 新增利润口径—非经常性损益勾稽、季度—年度核对、合并—母公司口径、分红实施状态、审计比较等 hard cases。
- 29 条问题绑定 53 个真实年报 hard negatives；frozen_test 12/12 全覆盖。
- 新增可运行的 `robustness` lane，保存 `gold / retrieved / hard_negative:<type>` 上下文标签。
- 上下文预算由 3 提升至 5；Retrieved runner 同步使用 top-5。
- Validator 新增直接数值 fact→quote containment、reference required-fact coverage、引用上下文计数、hard-negative 存在性/去重和完整 PDF 页范围门禁。
- 新增生成 run 配对比较脚本，拒绝跨 dataset/lane 比较并输出修复/回归 case。
- 增加通用成本/利润口径澄清策略，不依赖具体公司或固定问题全文。

### 数据覆盖变化

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| 问题数 | 32 | 48 | +16 |
| 原子事实 | 72 | 120 | +48 |
| Gold evidence | 39 | 67 | +28 |
| Hard negatives | 0 | 53 | +53 |
| 可运行评测 lane | 2 | 3 | +1 |
| Frozen robustness 覆盖 | 0/9 | 12/12 | 全覆盖 |
| PDF visual warnings | 0 | 0 | 保持为 0，改为检查完整跨页范围 |

### Clarify 策略配对结果（同一 Dataset ID）

| Lane | 行为准确率：前 | 行为准确率：后 | 变化 | 行为修复/回归 |
|---|---:|---:|---:|---:|
| Oracle | 0.9583 | 1.0000 | +0.0417 | 2 / 0 |
| Retrieved | 0.7917 | 0.8333 | +0.0417 | 2 / 0 |
| Robustness | 0.7241 | 0.7931 | +0.0690 | 2 / 0 |

修复 case：`u_moutai_profit_ambiguous`、`u_yili_cost_scope_ambiguous`。完整 paired artifacts 位于 `reports/generation/comparisons/`。

### 结论边界

- 上述结果是无 LLM 确定性链路基线，只证明 runner、行为策略和回归门禁工作，不代表 DeepSeek 生成质量。
- 旧 32 条 DeepSeek 结果不能迁移到当前 Dataset ID；需要重新运行三条 lane 与 RAGAS。
- 当前仍只有两份 2024 年报且没有人工独立 gold，因此定位为“非 toy 的冻结回归集原型”，不宣称生产级或跨文档泛化 benchmark。

## 2026-08-06：评测覆盖审计与单位契约修复

### 修改前

- RAGAS runner 固定要求 run 包含全部 37 条可回答题，因此 29 条受控干扰子集的 Robustness run 无法执行语义评测。
- 分红题以 `元/股` 作为标准单位，但参考答案采用“每股 x 元”，参考答案自检的单位准确率仅为 0.5。
- 数据卡将 25 道多 evidence-span 问题统称为“多证据问题”，没有区分同一 chunk 多 span 与多个独立上下文。

### 修改后与验证

- RAGAS 按 lane 绑定不可变 run：Oracle/Retrieved 必须完整覆盖 48 条，Robustness 必须精确覆盖 29 条 hard-negative 问题；Robustness 可评 18 条可回答题，`coverage=18/37`、`lane_coverage=1.0`，并保存完整 query ID 审计字段。
- 单位评分支持 `1.20元/股`、`每股1.20元`、`1.20元每股` 等等价写法，同时保持元、万元、亿元之间的严格尺度边界。37 条可回答参考答案单位自检由 1 个失败降为 0。
- 数据卡分别记录 25 道多 evidence-span 问题和 13 道多独立 gold chunk 问题，防止在项目陈述中夸大跨上下文推理覆盖。
- 定向验证：生成与 RAGAS 评测 17 passed；全量测试 68 passed；Ruff 全仓检查通过；artifact gate 通过（16 条检索 holdout、48 条生成问题、2 个实验 run）。
- Windows PowerShell artifact loader 显式使用 UTF-8 读取中文 JSON，消除了文件内容正确但启动门禁因默认编码误判失败的问题。

### 结论边界

- 当前评测集已经足以验证 within-corpus 的证据、行为和干扰鲁棒性，但 37 道可回答题中只有 13 道需要多个独立 chunk。
- 下一轮不应继续从同两份年报机械增题；优先新增 4–6 家公司、2–3 个年度以及 OCR/扫描和跨页表格文档，建立 document-level blind split，再将独立多 chunk 占比提升到至少 50%。
