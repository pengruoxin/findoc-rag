# FinDocRAG 优化实验记录

本文件记录每次工程改动的可复现对比，**最新的在最前**。任何优化必须同时记录代码变更、测试、评测指标和已知退化。

最低验收要求与迭代流程见 [评测基线 §5](../evaluation/baseline-zh.md#5-迭代协议)；当前基线数字见同一文档；每次实验的完整分析见 [实验总结索引](../evaluation/experiment-summaries.md)。看不懂的词见 [术语表](../glossary-zh.md)。

## 记录模板

```text
日期：
目标问题：
基线（run_id + code_revision + 是否脏）：
受控变量（只允许一个；同版本重复 run 用于噪声估计）：
固定变量（dataset / index / runner 参数 / api_model / remote / 功能开关）：
修改前基线：
修改内容：
测试命令：
测试结果：
评测结果变化：
新增能力：
已知退化/未覆盖：
结论：
```

## 2026-08-11：坐标重建迭代 v1→v9（92/157 → 154/157，追平文本基线）

- 目标问题：外部模型交付的坐标重建真实数据仅 92/157（R=0.586），远低于文本基线 154/157。
- 基线：coordinate-smoke-v1（92/157）；固定变量：同一评测脚本（整页输入、无区域裁剪、单元格三元组规则）。
- 受控变量：仅修改 `table_reconstruction.py` 与评测脚本的页面传递。
- 修改内容（6 项，每步用 `evaluate_coordinate_reconstruction.py` 回归）：
  1. 区域锚点取"最后一个匹配"并排除"报告期末公司前三年…说明"；
  2. 拆开标题（"2" + ")营业…"）边界识别；修复标题正则误判小数（"15.71"→"15."）；
  3. 季度扣非标签尾部跨行带回填（规范指标名前缀/后缀修复）；
  4. 散文标签过滤（主要原因/说明/标点/年份）；
  5. segment 列分配改为固定行结构前 3 数值列（表头几何被"营业收入比"碎片污染）；
  6. **跨页隔离**：token 增加 page，区域过滤与行带聚类按页隔离。
- 测试命令：`pytest tests/test_table_reconstruction.py`（13 passed，新增 2 个回归：季度拆标签修复、跨页不串行）；全量 144 passed；ruff 通过。
- 评测结果变化：
  - **92/157 → 154/157（R=0.981）**；quarterly 32/32、note_cost 24/24、segment 51/54（仅"其他地区"）、annual_data 39/39、concentration 8/8；
  - 与文本基线持平；坐标路径对阅读顺序倒置/跨行标签/跨页表格的几何鲁棒性优于文本正则。
- 新增能力：几何层坐标重建达到可用召回；跨页表格处理；评测尺子完整。
- 已知退化/未覆盖：yili_segment"其他地区"需 OCR 或标注分歧；季度/伊利 segment 各有 1–3 个多余 junk 行（Precision 0.94/0.85）；坐标路径未接入生产生成链路。
- 结论：合成级 → 真实级落差已消除，坐标路径可作为 `extract_cells` 的几何增强候选；下一步接入生成链路前先做一次全量三轨回归（需 key）。

## 2026-08-11：坐标级表格重建集成与真实数据评测（P0，外部模型交付）

- 目标问题：把外部模型交付的坐标重建模块集成进仓库，并在真实 PDF 数据上验证，而不是只信合成夹具。
- 修改内容：
  - 集成 `src/findoc_rag/table_reconstruction.py`（span→token→行带聚类→列对齐→标签-数值配对 + 文本回退 + `merge_pages`），legacy 回退改指向 `findoc_rag.table_extraction`；
  - 集成 11 个单测（含"标签后置""跨行标签"夹具）；
  - 新增 `scripts/evaluate_coordinate_reconstruction.py`：把标注表覆盖页的完整 pymupdf blocks 喂给坐标路径，按同一单元格三元组打分；
  - 修复 ruff 12 处（import 排序、盲异常、getattr 常量、隐式拼接等）。
- 测试命令：`pytest -q`（142 passed）；`ruff check` 通过。
- 评测结果变化：
  - 文本基线（生产路径）不变：154/157（98.1%）；
  - 坐标路径 P0（整页无裁剪）：**92/157（Recall 58.6%）**——concentration 8/8、note_cost R=1.0（P 仅 0.27–0.32）、annual R≈0.85、quarterly R=0.75（扣非行未配对）、segment R=0.06（最差）；
  - 外部模型自评的 150/157 是其内置文本回退在独立环境的结果，接入仓库后未复现，不作为坐标路径成绩。
- 新增能力：坐标重建的合成级与真实数据级双轨回归尺子。
- 已知退化/未覆盖：整页输入下精度/召回低于文本基线，**坐标路径暂不接入生产**；瓶颈为表格区域定位缺失（prose 行误当数据行）、segment 子表隔离、quarterly 扣非行配对；"其他地区"维持 OCR/标注分歧结论。
- 结论：坐标路径 P0 不可替代文本路径；下一步按 analysis.md 优先级迭代（区域定位 → segment 隔离 → 扣非行配对），每步用 `evaluate_coordinate_reconstruction.py` 回归。

## 2026-08-11：LLM 查询改写 retrieved lane 受控实验（阴性结果，不落地）

- 目标问题：OOV 实验证明 LLM 改写有效（Hit@5 0.194→0.694），能否直接用于 benchmark retrieved lane 补检索证据？
- 基线（concentration-v2，同 dataset / index / api_model / runner）：Retrieved strict 0.8286 / 行为 0.9375。
- 受控变量：retrieved lane 改写模式 deterministic → LLM（`--rewrite llm`，新增持久化缓存 `rewrites.json`）；其余固定。
- 修改内容：`LLMQueryRewriter` 支持持久化缓存（原子写）；`run_generation_eval` 增加 `--rewrite none|deterministic|llm` 并记录 `search_query`；新增缓存单测。
- 测试命令：`pytest -q`（131 passed）；ruff 通过。
- 评测结果变化（rewrite-llm-v1 vs concentration-v2）：
  - Retrieved strict 0.8286 → 0.8286（+1 yili_annual_deducted_profit / -1 moutai_revenue_yoy），行为 0.9375 → 0.8958（-2 为无关模型波动）；
  - 证据层：38/48 条 top-5 改变，遗留 3 个 miss 未修复，新增 3 个证据回归（moutai_revenue_yoy、moutai_disclosed_risks、yili_disclosed_risks）；
  - RAGAS：Retrieved context relevance 0.973→0.892、context recall 0.901→0.811；
  - 机制：LLM 把"同比增幅"改写为"同比增长率"，覆盖了精选映射"同比增幅→比上年同期增减"。
- 新增能力：改写缓存持久化 + rewrite 模式 A/B 基础设施（对 OOV/生产接入仍有用）。
- 已知退化/未覆盖：三个证据回归已确认由改写引入；生产 `/v1/query` 的 paraphrase 门控与"改写+确定性兜底"未做。
- 结论：**retrieved lane 默认保持 deterministic 词表；LLM 改写不进入 canonical 评测链路**。OOV 收益与 canonical 无收益并不矛盾——问句分布不同。此阴性结论防止了一次会伤害 3 题的发布。

## 2026-08-11：concentration 表型抽取器 + 单公司选题修复（受控实验，`deepseek-chat-concentration-v2`）

- 目标问题：moutai_concentration / yili_concentration 是剩余行为失败集中点（Robustness 全中），且未被四类表型覆盖。
- 基线（table-remote-v1，同 dataset / index / api_model / runner）：Oracle 1.0 / Retrieved 0.8286 / Robustness 0.8636（strict）；行为 1.0 / 0.9583 / 0.8621。
- 受控变量：新增 concentration 表型（抽取 + 生成格式化 + 单公司按查询公司选题修复）；其余固定。
- 修改内容：
  - `table_extraction.py`：`concentration` 表型，正则抽取前五名客户/供应商的销售额（万元）、销售/采购占比（%），关联方第二次占比不误取；新增 table-eval-concentration-v1（8 单元格）；
  - `answer_generation.py`：`_concentration_answer` 单公司与跨公司（客户/供应商集中度差）格式化；v1 暴露 bug 后修复为按查询公司选题；
  - 新增 3 个抽取 + 4 个生成单测（35 passed）。
- 测试命令：`pytest tests/test_table_extraction.py tests/test_answer_generation.py`（35 passed）；全量 127 passed；ruff 通过。
- 评测结果变化（concentration-v2 vs table-remote-v1，零回归）：
  - Robustness strict **0.8636 → 0.9545**（+2：moutai_concentration、yili_concentration），行为 0.8621 → 0.9655（+3，含 1 个无关模型波动 audit_opinion_comparison）；
  - Oracle 持平 1.0 / 1.0；Retrieved strict 持平 0.8286，行为 0.9583 → 0.9375（-1 为 yili_2025_plan_bounded 模型波动，与 concentration 无关）；
  - RAGAS（clean `f8a1be8`、`code_dirty=false`）：Oracle 0.773 / Retrieved 0.804 / Robustness 0.731。
- 新增能力：第五类表型抽取 + 集中度对比的确定性回答；单公司选题不再依赖 hit 顺序。
- 已知退化/未覆盖：Retrieved 行为波动（非表格题）仍在；坐标级表格重建（PDF 文字层丢字）未做；LLM 改写仍未落地。
- 结论：concentration 表型按预期消灭剩余集中度误拒答；过程中受控对比抓到一个真实 bug（负例前置取错公司），已修复并保留 v1 run 作审计记录。

## 2026-08-11：远程模式确定性表格优先（受控实验，`deepseek-chat-table-remote-v1`）

- 目标问题：远程模式下确定性表格路径被跳过，表格类可答题交给 DeepSeek 后出现"证据齐全仍误拒答"（moutai_quarterly_cashflow facts=1.0 拒答、yili_consolidated_parent_revenue 拒答）。
- 基线（abstain-v2，同 dataset / index / api_model / runner）：Oracle 0.9429 / Retrieved 0.8000 / Robustness 0.8636（strict）；行为 0.9792 / 0.8958 / 0.8276。
- 受控变量：`FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES=1`（只动这一个开关；默认关）。
- 固定变量：benchmark-v2 / index `10fb50419145d56720c9` / api_model=deepseek-chat / remote=true / top-k=5 / candidate-k=20 / hybrid。
- 修改内容：`generate()` 中远程模式默认仍跳过表格路径；开关开启时表格题优先返回确定性答案（带引用）。
- 测试命令：`pytest tests/test_answer_generation.py`（21 passed）；新增 2 个开关单测。
- 评测结果变化（配对，零回归）：
  - Oracle strict **0.9429 → 1.0000**（+2：yili_consolidated_parent_revenue 确定性修复、moutai_annual_deducted_profit 模型波动），行为 1.0000；
  - Retrieved strict **0.8000 → 0.8286**（+1：moutai_quarterly_cashflow），行为 **0.8958 → 0.9583**（+3 / -0）；
  - Robustness strict 0.8636（持平），行为 **0.8276 → 0.8621**（+1 / -0）；
  - RAGAS：Oracle faithfulness 0.783 / Retrieved 0.868 / Robustness 0.813（`independent_judge=false`，带 code_revision）。
- 新增能力：远程模式下表格题的确定性兜底，消除表格类误拒答与 API 波动。
- 已知退化/未覆盖：concentration 类表格（未被四类抽取器覆盖）仍会误拒答；非表格题的行为波动仍在（yili_2025_plan_bounded 等）；工作区未提交，`code_dirty=true`。
- 结论：单变量验证成立——确定性表格优先在远程模式零回归地提升行为与 strict；下一步扩展抽取器覆盖 concentration 表型或做拒答 prompt 策略。

## 2026-08-11：远程拒答检测（打分口径修正，`deepseek-chat-abstain-v2`）

- 目标问题：远程回答 `grounded` 恒为 True，"无法回答……数字……"的拒答被算成正确回答（yili_concentration 实证）；9 条应拒答题的拒答永远计为 answer → 行为与 strict 被低估；可答题的拒答被掩盖。
- 修改前基线（table-v2）：Oracle 0.9714 / Retrieved 0.5714 / Robustness 0.6364（strict），行为 1.0 / 0.8333 / 0.7931。
- 修改内容：
  - `answer_generation.py` 新增 `ABSTENTION_PATTERNS` + `_is_abstention`（保守规则：无法/不能+回答/确认/给出/判断等、证据不足、信息不足等；不含"不确定性"等中性词），`_generate_remote` 命中后返回 `provider=remote-abstention`、`grounded=false`；
  - `run_generation_eval.py` 将 `remote-abstention` 计入远程调用（model/api_model 正确落盘）；
  - 新增 3 个拒答检测单测（19 passed）。
- 测试命令：`pytest tests/test_answer_generation.py`（19 passed）；全量 118 passed；ruff 通过。
- 评测结果变化（abstain-v2 vs table-v2，逐题配对）：
  - Retrieved strict **0.5714 → 0.8000**（+8 应拒答修复 / -0 回归），行为 0.8333 → 0.8958；
  - Robustness strict **0.6364 → 0.8636**（+6 / -1：yili_concentration 打分怪癖消除），行为 0.7931 → 0.8276；
  - Oracle strict 0.9714 → 0.9429（+1 / -2：个别可答题本轮拒答被如实计分；abstain-v1 曾 1.0，属单题波动）；
  - RAGAS：Oracle faithfulness 0.796 / Retrieved 0.887 / Robustness 0.834（`independent_judge=false`、`api_model_recorded=true`）。
- 新增能力：拒答/回答边界可审计（provider 区分 remote-abstention），行为准确率不再虚高。
- 已知退化/未覆盖：拒答检测是启发式（可能漏检极少数措辞变体）；可答题拒答（每轨 4–5 条）成为行为层当前主要失败，属模型行为问题而非评测问题；Oracle 单题波动仍存在。
- 结论：这是打分口径修正而非能力突变——应拒答被正确计分、自带数字的伪拒答不再刷分；真实行为基线为 Retrieved 0.8958 / Robustness 0.8276，下一步修可答题误拒答与检索缺证据。

## 2026-08-11：DeepSeek 三轨重跑 + RAGAS 元数据修复（表格路径端到端验证）

- 目标问题：表格确定性路径只验证了 no-LLM；DeepSeek 三轨仍是 8/7 旧基线；且 RAGAS 产物 `independent_judge` 用 run 标签（`--model`）判断，真实 API 模型未落盘，自评被误标为独立评测。
- 修改前基线：Oracle 0.9714 / Retrieved 0.5429 / Robustness 0.5455（8/7）；RAGAS 元数据 `independent_judge=true`（错误）。
- 修改内容：
  - `GenerationRunItem` 增加 `api_model` 字段；`run_generation_eval.py` 落盘真实 API 模型（`--api-model`），`--model` 保持纯 run 标签；
  - `run_ragas_generation_eval.py` 用 `api_model` 判断独立性，输出 `api_model_recorded` 审计字段；
  - 重跑 DeepSeek 三轨（`deepseek-chat-table-v2`，本地 key）并跑 RAGAS。
- 测试命令：全量 pytest 待跑；ruff 待跑。
- 评测结果变化（配对 vs 8/7 基线）：
  - Oracle strict 0.9714 → 0.9714（修复 moutai_annual_deducted_profit / 回归 yili_quarterly_profit_reconcile，单题互换，v1 run 曾到 1.0——属 API 非确定性）；
  - Retrieved strict 0.5429 → **0.5714**（+2/-1）；Robustness strict 0.5455 → **0.6364**（+2/-0）；
  - RAGAS faithfulness：Oracle 0.682→0.778、Retrieved 0.836→0.845、Robustness 0.680→0.849；answer_relevancy 三轨全部提升；产物 `independent_judge=false`、`api_model_recorded=true`。
- 新增能力：DeepSeek 三轨 + RAGAS 的真实当前基线（表格路径、修正元数据）；run 级单题波动可审计。
- 已知退化/未覆盖：单题 ±1 波动（同模型温度 0 仍非确定），结论看趋势；Retrieved 仍 15 题 strict 失败（检索 top-5 缺证据为主）；RAGAS 仍是同模型自评，独立 judge 归公信力阶段。
- 结论：表格路径在真实模型下无退化且有提升（Retrieved +1、Robustness +2），元数据可信度问题已修复；下一步转向行为拒答与检索侧（LLM 改写落地）。

## 2026-08-11：表格确定性回答接入生成链路（no-LLM 三轨验证，清单 P1-11 收口）

- 目标问题：单元格抽取器只停留在评测层，生成链路仍用旧季度正则；no-LLM 确定性链路 strict 只有 0.3143 / 0.0857 / 0.2273。
- 修改前基线（benchmark-v2，disable 表格路径复跑确认）：Oracle 0.3143 / Retrieved 0.0857 / Robustness 0.2273。
- 修改内容：
  - `table_extraction.py`：`ExtractedCell` 增加 `section` 字段（区分分行业/分产品/分地区/分销售模式）；新增 `extract_annual_rows`（保留同比列），`extract_annual_data` 改为其投影，table-eval 数字不变（146/149）；
  - `answer_generation.py`：新增 `_deterministic_table_answer` 及季度（含季度-年度核对）、附注成本（含核对差额）、分部毛利率（产品/渠道、最高项）、年度营收（同比）、跨公司营收差、合并-母公司营收差六个格式化路径，全部带 `[n]` 引用；
  - A/B 开关：`FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES=1` 可关闭表格路径，用于复现旧基线；
  - 新增 10 个单测（24 passed）。
- 测试命令：`pytest tests/test_answer_generation.py tests/test_table_extraction.py`（24 passed）；全量 `pytest -q` 待跑。
- 评测结果变化（逐题配对，零回归）：
  - Oracle strict **0.3143 → 0.6571**（12 题修复：季度现金流/归母、季度-年度核对×2、附注成本、成本核对×2、产品毛利率×2、渠道毛利率差、年度营收同比、跨公司营收差、合并-母公司营收差）；
  - Retrieved strict **0.0857 → 0.2571**（6 题修复）；Robustness strict **0.2273 → 0.3636**（3 题修复）；
  - 行为准确率不变（1.0 / 0.8333 / 0.7931），error rate 0。
- 新增能力：确定性表格回答成为 no-LLM 链路的一部分；`extract_annual_rows` 供同比等派生事实使用。
- 已知退化/未覆盖：Retrieved 剩余 6 题因证据不在 top-5 未受益（检索侧）；计算题如非经常性损益核对未纳入确定性路径；表格回答对"表格格式变化"仍脆弱，需坐标级重建兜底。
- 结论：B 阶段表格抽取层已传导到端到端（no-LLM 基线翻倍以上），下一步重跑 DeepSeek 三轨验证真实模型下的端到端 strict（需要 key）。

## 2026-08-11：B 阶段表格重建第一步——四类表型单元格抽取器（清单 P1-11）

- 目标问题：季度、附注收入成本、分部、年度数据四类表只有 quarterly 正则基线（28/32），其余三类 0/149；表格数字抽不出来是端到端 strict（0.54）与 Oracle（0.97）差距之外的最大单点瓶颈。
- 修改前基线：table-eval-v1 全表型 28/149（18.8%）；quarterly 茅台扣非行 4 格全错。
- 修改内容：
  - `table_extraction.py` 新增 note_cost / segment / annual_data 抽取器，重写 quarterly（按"4 个数值一组 × 出现指标"对齐，修好标签在数值后的扣非行；剔除季度区间"（1-3 月份）"与标题年份）；
  - 关键实现：保留空白做数字分隔（避免压缩后数字粘连）、小节头部按"最后一个表头词之后"剔除、行标签监管科目集合校验、annual_data 跳过同比列并兼容 3 数值行（股本）；
  - `evaluate_table_extraction.py` 移除"只有 quarterly 算 implemented"的旧标记；
  - 新增 5 个抽取器单测（共 8 passed）。
- 测试命令：`pytest tests/test_table_extraction.py -q`（8 passed）；`ruff check` 通过。
- 评测结果变化：table-eval-v2 **28/149 → 146/149（98.0%）**：quarterly 32/32、note_cost 24/24、segment 51/54、annual_data 39/39。
- 新增能力：四类表型可回归的单元格抽取层，`extract_cells` 统一入口。
- 已知退化/未覆盖：伊利 segment"其他地区"行 3 格不匹配——chunk 文本层只有"其他"（PDF 文字层丢"地区"），gold 按视觉复核标注"其他地区"；这是上游 Document IR 文字层丢失，需坐标级重建修复，不在抽取器内硬编码改名。抽取器尚未接入 answer_generation，端到端指标未重跑。
- 结论：表格抽取层从 18.8% 提到 98.0%；下一步接入 answer_generation 确定性表格路径，重跑 no-LLM 三轨，再跑 DeepSeek 三轨验证端到端。

## 2026-08-11：OOV 评测 + 中文 dense 对照（roadmap 阶段 0 / 清单 P2-17）

- 目标问题：词表外改写（OOV）能否被 LLM 查询改写救回？更强的中文 dense（bge-small-zh-v1.5）能不能替代/补充查询改写？
- 修改前基线：OOV 36 实例（12 题），rewrite=none，lexical Hit@5 0.194 / MRR 0.148；dense 0.139；hybrid 0.222。
- 修改内容：
  - `indexing._dense_text` 支持 BGE v1.5 查询指令前缀（`为这个句子生成表示以用于检索相关文章：`），E5 / BGE-M3 / 其他模型路径不变；新增 4 个单测；
  - 用同一份 958 chunks 构建第二个语料索引 `data/indexes/corpus-bge-zh`（`6a951f4e8b7bd913d918`，bge-small-zh-v1.5，512 维），不覆盖 E5 索引；
  - 新跑 4 个实验：`oov-eval-bge-zh-v1`、`oov-eval-llm-v1`、`oov-eval-llm-bge-zh-v1`、`variant-regime-bge-zh-v1`，另补 `oov-eval-deterministic-v1` 对照。
- 测试命令：`pytest tests/test_indexing.py -q`（13 passed）；全量 `pytest -q`（97 passed）。
- 评测结果变化：
  - LLM 改写：lexical Hit@5 0.194 → **0.694**（MRR 0.148 → 0.498），候选召回 0.333 → 0.861；
  - deterministic 改写对 OOV 无效果（0.194，符合设计）；
  - bge-zh dense：OOV 0.083（< E5 0.139）；变体三问法 0.162 / 0.568 / 0.162（均 < E5 0.216 / 0.703 / 0.189）；LLM 改写后 dense 仍 0.083；
  - hybrid 依旧负优化：LLM 改写后 0.472（E5）/ 0.444（bge）均低于 lexical 0.694 / 0.667。
- 新增能力：OOV 评测三档改写（none / deterministic / llm）在 E5 与 bge-zh 两套索引上的完整矩阵；BGE 指令前缀支持。
- 已知退化/未覆盖：
  - LLM 改写跨 run 不稳定（36 条中 8 条输出不同、1 条改变命中），改写缓存未持久化；
  - 剩余 11 个 miss 归因：财务简称未归一（扣非净利润、同比变化）、PDF 行内换行导致 bigram 断裂（其他系列 酒）、文档措辞未知（风险因素→可能面对的风险）、LLM 改写引入原问题没有的指标；
  - bge-m3 等更大模型未测；未做人工审核。
- 结论：**默认 lexical-only 不变；LLM 查询改写是 OOV 的决定性杠杆，应进入生产链路；换中文小 dense 模型在当前输入上没有收益，待表格结构化后与更强模型一起重验**。

## 2026-08-07：同义词查询改写（清单 P1-5 收口 / roadmap A 阶段）

- 目标问题：口语问法 10 条检索全 miss，根因是用户措辞与年报措辞不重叠。
- 修改前基线：semantic Hit@5 0.730（lexical，query_parser 过滤）。
- 修改内容：
  - 新增 `src/findoc_rag/query_expansion.py`：7 组财务同义词映射，全部从失败案例提取（营收→营业收入、毛利水平→毛利率、净资产回报率→净资产收益率、主要风险→可能面对的风险、前五大客户→前五名客户、同比增幅→比上年同期增减、一定实现→计划实现）；
  - 检索评测脚本加 `--expand-synonyms`；生成 runner `retrieved_hits` 默认启用改写；
  - 新增 8 个单测。
- 测试命令：`pytest -q`（93 passed）；`ruff check .`；variant-regime-expanded-v2；retrieved lane（`--model deepseek-chat-expanded`）。
- 评测结果变化：
  - 检索侧（lexical，query_parser 过滤）：canonical 0.838 → 0.892、ticker 0.811 → 0.892、semantic **0.730 → 0.919**；9 题救回、0 回归；
  - 端到端 retrieved strict **0.5429 持平**。逐题归因 16 个 strict 失败：行为拒答 8、表格/生成 5、检索 3（其中 1 条实为评测时间对齐 bug）。
- 新增能力：专业同义词鲁棒性从"全员弱区"变为三档最高；端到端瓶颈定位从"检索"转移为"行为拒答 + 表格抽取"。
- 已知退化 / 未覆盖：词表为 7 组定向映射，覆盖评测内已知改写，长尾改写仍需失败驱动扩展；`/v1/search` 服务默认未启用改写（待集成）。
- 结论：A 阶段（检索定论）完成——检索指标已达上限附近，下一步主战场是行为拒答与表格抽取。完整分析见 `reports/ranking/variant-regime-expanded-v2/analysis.md`。

## 2026-08-07：DeepSeek 三轨真实基线 + RAGAS（清单 P2-14）

- 目标问题：48 题新评测集从没跑过真实模型；所有生成侧数字都是确定性链路，无法对外主张。
- 修改前基线：no-LLM 确定性链路 strict 0.3143 / 0.0857 / 0.2273；旧 32 题 DeepSeek Oracle 0.9583 不迁移。
- 修改内容：
  - 修复 `run_generation_eval.py`：`--model` 之前同时充当 run 标签和 API 模型名，标签一改（如 `deepseek-chat-v2`）就导致全部请求 400 Bad Request；新增 `--api-model` 分离两者；
  - `answer_generation.py` 远程调用加 3 次指数退避重试（网络超时 / 5xx / 429 类错误），timeout 提到 connect 30s / total 120s；
  - RAGAS 脚本默认数据集对齐到 `benchmark-v2.json`。
- 测试命令：三轨 `run_generation_eval.py --lane <lane> --model deepseek-chat-v2 --api-model deepseek-chat --require-remote`；RAGAS 三个 run。
- 测试结果（`deepseek-chat`，48 题，2026-08-07）：

  | 赛道 | strict（eligible） | 行为准确率 | avg_context_tokens | p95 延迟 | RAGAS faith/rel/ctx-rel/ctx-rec | error |
  |---|---|---:|---:|---:|---:|---:|
  | oracle | 0.9714（35） | 1.0000 | 303 | 1668ms | 0.68 / 0.98 / 1.00 / 0.99 | 0 |
  | retrieved | 0.5429（35） | 0.8333 | 1536 | 1843ms | 0.84 / 0.73 / 0.93 / 0.82 | 0 |
  | robustness | 0.5455（22） | 0.7931 | 784 | 2441ms | 0.68 / 0.71 / 0.94 / 1.00 | 0 |

- 新增能力：第一份可对外主张的真实模型基线；"给对证据就 97% 全对"证明瓶颈在检索 / 路由 / 证据选择，不在生成；上下文效率指标在真实 LLM 链路上有了数字（1536 token / p95 ≈1.8s）。
- 已知退化 / 未覆盖：RAGAS 是 DeepSeek 自评（`independent_judge=false`）；仅 2 家公司 1 年度；网络偶发超时（已加重试缓解）。
- 结论：生成侧瓶颈定位完成（Oracle 97% vs Retrieved 54%）；检索侧优化（同义词、路由、表格结构化）是下一步的主战场。

## 2026-08-07：变体问句首轮检索评测（清单 P1-5 / P1-6）

- 目标问题：96 个变体问句建好之后从没评测过；系统对股票代码、财务简称、相对时间这三种真实问法的检索能力完全未知。
- 修改前基线：只有 16 题旧回归集的原题结果（关键词 0.6875 / 语义 0.25 / 混合 0.6875，均未过滤）。
- 修改内容：新增 `scripts/run_retrieval_variant_eval.py`，对检索视图的 111 个该答问句（37 道题 × 3 种问法）跑关键词 / 语义 / 混合三路 × 不过滤 / 按问句解析过滤两态，输出逐题结果、汇总、分析和完整配置（可复现）。
- 测试命令：`python scripts/run_retrieval_variant_eval.py --output-dir reports/ranking/variant-regime-v1`；ruff 通过。
- 测试结果：脚本产出 `summary.json` / `per_query.jsonl` / `analysis.md` / `config.json`。
- 评测结果变化（开过滤，Hit@5 / MRR@5）：

  | 问法类型 | 关键词 | 语义 | 混合 2:1 |
  |---|---|---|---|
  | 原题（照原文问） | 0.838 / 0.694 | 0.216 / 0.146 | 0.649 / 0.365 |
  | 代码 / 简称 | 0.811 / 0.724 | 0.676 / 0.469 | 0.730 / 0.671 |
  | 口语 / 相对时间 | 0.730 / 0.633 | 0.162 / 0.113 | 0.459 / 0.321 |

- 新增能力：
  - 专业表达鲁棒性变成可量化指标：语义检索只在"代码 / 简称"问法上有效（0.676），另两类极弱；
  - 找到主要失败模式：37 道口语问法的题里 11 道三路全没找到，全是专业同义词改写（毛利水平 vs 毛利率、净资产回报率 vs 净资产收益率）；
  - 拿到"混合检索被弱语义分支拖累"的量化证据：4 道题的正确证据被关键词检索排在第 1–2 位，融合后被推到第 6–7 位；
  - 时间对齐 bug 实锤：`yili_2025_plan_bounded` 的提问时点是 2026，"去年"解析成 2025，系统就拿 2025 去过滤，把 2024 年报里的正确证据排除了（这是清单 P2-15 的评测证据）；
  - 相对时间解析 43/43 正确；行为组 33 个问句的检索污染基线：关键词和混合各有 11 条前 5 名里出现了干扰段落。
- 已知退化 / 未覆盖：只有 2 家公司 1 个年度；Precision / NDCG 是部分判定；还没做权重扫描；行为赛道的正式指标还没跑。
- 结论：优化优先级从"换更强的语义模型"转向"同义词鲁棒性 + 融合策略 + 时间对齐"。完整分析见 `reports/ranking/variant-regime-v1/analysis.md`。

## 2026-08-07：融合权重扫描（P1-5 下一轮）

- 目标问题：上一轮发现混合 2:1 被弱语义分支拖累。能不能靠调权重，或者按问法类型动态选路，把这个负优化消掉？
- 修改前基线：混合 2:1 开过滤后 Hit@5 分别是 0.649 / 0.730 / 0.459。
- 修改内容：新增 `scripts/run_retrieval_fusion_sweep.py`。两路的排名只算一次，然后离线融合 6 组权重（1:1 / 2:1 / 3:1 / 4:1 / 1:0 / 0:1），输出每组权重的汇总和逐问法最优（仅供开发期参考）。
- 测试命令：`python scripts/run_retrieval_fusion_sweep.py --output-dir reports/ranking/fusion-sweep-v1`；ruff 通过。
- 测试结果：脚本产出 `summary.json` / `summary.md` / `config.json` / `analysis.md`。
- 评测结果变化（开过滤，Hit@5 / MRR@5；权重是"关键词 : 语义"）：

  | 权重 | 原题 | 代码 / 简称 | 口语 / 相对时间 |
  |---|---|---|---|
  | 1:1 | 0.595 / 0.357 | 0.730 / 0.640 | 0.432 / 0.306 |
  | 2:1 | 0.649 / 0.365 | 0.730 / 0.671 | 0.459 / 0.321 |
  | 4:1 | 0.703 / 0.417 | 0.730 / 0.689 | 0.514 / 0.350 |
  | **1:0（纯关键词）** | **0.838 / 0.694** | **0.811 / 0.724** | **0.730 / 0.633** |

- 新增能力：用数据证明了"融合不是默认就更好"。逐问法挑最优也全都选 1:0，说明按问法动态选路在当前模型下没有增益。
- 已知退化 / 未覆盖：逐问法最优是在评测集上选的，有过拟合风险；默认策略改动当时还没落地验证；语义模型没做升级对照。
- 结论：默认检索策略应改成纯关键词检索；语义模型升级后用同一套扫描重新验证融合；同义词增强仍是口语问法的下一项。完整分析见 `reports/ranking/fusion-sweep-v1/analysis.md`。

## 2026-08-07：默认检索策略落地为纯关键词检索

- 目标问题：上一轮已经证明纯关键词全面优于融合，但代码默认还是混合 2:1，文档说的和系统做的不一致。
- 修改前基线：默认混合 2:1；真实检索赛道全对率 0.0857 / 行为准确率 0.8333。
- 修改内容：默认值统一改成关键词检索——`config.py` 的 `RetrievalSettings.default_mode`、`findoc-rag.example.toml`、`indexing.PersistentIndex.search`、CLI 三处命令、生成评测脚本的检索调用、`eval-config-template.json`、诊断评测默认值。检索对比实验脚本保留三路模式不变，因为它们本来就要做对照。
- 测试命令：`pytest -q`（82 passed）；`ruff check .`；`python scripts/run_generation_eval.py --lane retrieved_context --model no-llm-lexical-default`。
- 测试结果：真实检索赛道全对率 0.0857 / 行为准确率 0.8333 / 报错 0 —— 与混合基线**完全持平，无退化**。
- 新增能力：系统默认行为和评测结论一致了。语义检索仍可通过显式 `--mode dense/hybrid` 或配置文件开启。
- 已知退化 / 未覆盖：不接大模型的链路里，全对率被表格抽取的上限卡住，所以检索排名的提升没有体现到生成侧（直接给答案只有 0.31 这个瓶颈还在）。语义模型升级后需要重新验证融合。
- 结论：落地完成，验收通过（roadmap A 阶段"生成侧回归无退化"达成）。

## 2026-08-07：表格抽取评测层 v1（清单 P1-11）

- 目标问题：表格题答不对（Oracle strict 0.31）但无法定位错在行、列还是值；表格重建没有可回归的尺子。
- 修改前基线：无单元格级评测，只知道整题 0.31。
- 修改内容：
  - 标注 `data/evaluation/table-eval-v1.json`：8 张真实表格、149 个单元格三元组（行 / 列 / 值 / 单位），覆盖季度表、附注收入成本表、分行业表、主要会计数据表；
  - 新增 `src/findoc_rag/table_extraction.py`：抽取器接口 `extract_cells(text, table_type)` + quarterly 正则基线（其他表型留待 B 阶段实现）；
  - 新增 `scripts/evaluate_table_extraction.py`：单元格级 Precision / Recall / 错误行诊断；
  - 新增 3 个单测。
- 测试命令：`pytest -q`（85 passed）；`ruff check .`；`python scripts/evaluate_table_extraction.py --output-dir reports/ranking/table-eval-v1`。
- 测试结果：quarterly 28/32（0.875）；note_cost / segment / annual_data 未实现（0）；全表型正确 28/149（18.8%）。
- 新增能力：抓到具体 bug——茅台季度表扣非行 4 格全错（PDF 线性化顺序是数值在前、标签在后，正则把下一行现金流当成扣非值）；伊利同表全对（顺序正常）。这直接解释了 Oracle 0.31 的表格侧原因。
- 已知退化 / 未覆盖：标注为 assistant 级；未覆盖增减率列和"减少 x 个百分点"文本列；其他表型抽取器未实现。
- 结论：尺子已就位，B 阶段表格重建有了精确起点（149 单元格中 121 个待恢复）与回归接口。

## 2026-08-06：P0 回答可信度修复

三项来自同日的问题审计，逐条记录现象、影响、根因、修复与验收。

### 1. 回答生成源文件中文编码损坏

- 现象：`src/findoc_rag/answer_generation.py` 中拒答文案、季度指标、提示词和表格输出出现 `褰撳墠`、`缁忚惀` 等乱码；原有季度测试也使用乱码文本，无法证明中文链路正常。
- 影响：用户看到乱码；季度抽取条件无法稳定匹配；发送给 DeepSeek 的 system prompt 语义失真。
- 根因假设：文件曾被以错误编码读写，导致 UTF-8 中文被二次转码。
- 修改内容：以 UTF-8 重写中文常量、prompt 和测试样例，保留 extractive、deterministic-table、DeepSeek 三种 provider；禁止非 UTF-8 文件进入源码。
- 验收与结果：三个真实问题返回可读中文；季度抽取测试通过；Ruff 通过。
- 未覆盖：历史索引中的乱码文本需要重新解析后才能彻底消除。

### 2. `/v1/query` 未自动应用公司和年份过滤

- 现象：`SearchRequest.filters` 只有用户显式传入时才生效。
- 影响：查询"贵州茅台 2024 年……"时可能召回伊利或其他公司的相似章节（不同公司年报章节名高度重复），LLM 仍会在错误证据上生成看似合理的答案。
- 修改内容：从 query 推断公司名和 `20xx` 年份，与显式 filters 合并；冲突时以显式过滤为准并记录 trace。
- 验收与结果：真实三问的 evidence 公司、年份与问题一致；跨公司证据触发拒答。API 全量测试需在 uv trampoline 环境问题修复后重跑。
- 评测结果变化：增加了查询级路由约束；尚未宣称 Hit@K 提升，需用同一 holdout 重跑对比。
- 未覆盖：公司别名、简称和多公司联合问题。

### 3. 远程 LLM 输出没有二次证据校验

- 现象：只要 HTTP 请求成功就标记 `grounded=true`，没有检查回答是否引用 `[1]` 或是否包含证据外数字。
- 影响：DeepSeek 可能返回未被证据支持的数字，系统却把答案标为 grounded。
- 修改内容：回答前校验公司、报告年份及营业收入 / 营业成本 / 现金流 / 净利润等核心指标是否出现在证据中；远程回答必须包含 `[1]`、`[2]` 或 `[3]` 引用，否则返回 `guardrail-abstention`；针对"经营活动现金流量净额 / 经营活动产生的现金流量净额"增加别名匹配。
- 验收与结果：错误引用、无引用和证据外数字均不能标记 grounded。`tests/test_answer_generation.py` 3 passed；新增错误公司与匹配公司年份指标的回归测试；Ruff 通过。
- 未覆盖：回答中的具体数字与证据数字的逐项比对；上传后的异步解析建索引。

> 审计当时运行 `uv run pytest -q; uv run ruff check .` 时测试进程未能启动完成（`uv` 报告 trampoline 无法 canonicalize script path）。该问题属运行环境 / 工具链问题，当时未记录为项目测试通过；上述结果为环境恢复后重新执行所得。

## 2026-08-06：单路检索与融合的首次对比

- 目标：回答"关键词检索、语义检索、融合三者相比单用关键词检索，收益如何"。
- 控制变量：同一索引、16 题回归集、取前 5 名、候选池 20，关掉元数据过滤、口径路由、自适应候选池和重排。
- 结果：关键词 Hit@5 0.6875 / MRR 0.4531；语义 0.2500 / 0.0802；等权融合 0.6250 / 0.3917。
- 结论：融合相比纯关键词，Hit@5 掉了 0.0625，MRR 掉了 0.0614，没有提升。主要风险是弱的语义分支通过等权融合把强的关键词信号稀释了。
- 报告：`reports/ranking/retrieval-comparison-v3.md`。

### 给关键词路加权（2:1）

- 修改：融合支持配置两路权重，默认关键词 2.0、语义 1.0；服务层从配置读取。
- 测试：Ruff 通过；索引 / 配置 / 诊断相关共 14 passed。
- 结果：加权融合 Hit@5 0.6875、MRR 0.3948。相比等权融合 Hit@5 +0.0625、MRR +0.0031；相比纯关键词 Hit@5 持平、MRR -0.0583。
- 结论：2:1 消除了等权融合的 Hit@5 退化，但没改善"第一个正确结果排第几"。下一步应该做权重扫描，而不是直接宣称融合已经优于纯关键词。

## 2026-08-06：双层评估体系

- 检索侧新增逐题和汇总的 Precision@K、Recall@K、NDCG@K，保留原有的 Hit@K、MRR@K 和候选池召回。
- 加权融合实测：Recall@5 0.6875、Precision@5 0.1375、MRR@5 0.3948、NDCG@5 0.4666。
- 生成侧新增四项语义指标（答案有没有编、答案切不切题、上下文相不相关、上下文覆盖不覆盖）的数据契约和聚合器，强制记录打分来源是人还是模型；模型打分必须记下具体模型名。
- 验证：相关测试 5 passed；Ruff 通过。
- 边界：生成集还没有人工复核，所以这四项语义指标不编造分数。

## 2026-08-06：生成评测集 v1

- 建立 32 题的生成回归集，证据全部核验过原文：24 该答、7 该拒答、1 该追问，共 72 个原子事实、39 组证据。
- 按调参 / 开发 / 冻结测试划分；同考点、同证据的题不跨划分。
- 39 组证据全部翻回原始 PDF 页人工看过，校验器警告为 0。
- 新增确定性评分器：财务数字归一化、事实召回、单位、引用有效性、上下文召回、全对判定和拒答正确性。
- 导出 24 条给语义评测用的样本；固定 RAGAS 0.3.1 和兼容的 LangChain 版本，真实导入已验证。
- W&B 默认关闭，可切到离线 / 在线记录数据集产物、配置和四项语义指标。
- 验证：生成评测定向测试 6 passed；首次全量 53 passed；Ruff 全仓通过。

### 首轮不接大模型的基线暴露的问题

- 首次跑"直接给答案"赛道：全对率 0.2188、行为准确率 0.2500、报错 0。暴露一个真问题——注册表里的原始切片缺公司和年份元数据，导致正确的证据被证据门禁误拒。
- 给正确切片补上已核验的公司和年份后，行为准确率升到 0.9375（+0.6875）。
- 再修一个 PDF 问题："经营活动产生的现金流量净额"在原文里换了行，导致证据门禁和季度抽取都失败。修完行为准确率升到 0.9688（+0.0313）。剩下 1 条是系统当时还不会追问。
- 新增指标名内部换行的回归测试；回答生成测试 4 passed。
- 修完全量回归：54 passed；Ruff 全仓通过。
- 当时的进程没继承终端里的 `DEEPSEEK_API_KEY`，所以没有编造 DeepSeek 和语义评测的分数。真实基线需要在有 token 的同一环境里跑。

### DeepSeek"直接给答案"赛道基线

- 32 题全部跑完，API 报错率 0。
- 初版评分器：全对率 0.8438、行为准确率 0.9688、数值准确率 1.0、单位准确率 1.0、引用有效性 1.0、上下文召回 1.0。
- 逐条复核发现四条叙述类"失败"其实语义正确，是评分器要求完整字符串匹配造成的假失败。
- 评分器 v2 把叙述类事实交给语义评测和人工复核，确定性的全对率只统计有计分资格的数值 / 单位 / 引用 / 拒答样本：可计分 24 条，全对率 0.9583；8 条需要语义复核；行为准确率仍是 0.9688。
- 唯一确定的行为失败：模糊问题"茅台利润是多少"应该追问年份和利润口径，当时系统只会拒答。

## 2026-08-06：关键词检索保留财务科目整词

- 修改前基线：中文只按单字和二字组合切词，"营业收入""经营活动产生的现金流量净额"这类长会计科目会被拆散，精确匹配的信号不足。
- 修改内容：切词时保留高价值财务科目的整词，同时保留原有的二字组合，兼容旧索引的查询逻辑。
- 测试命令：`uv run ruff check src/findoc_rag/indexing.py tests/test_indexing.py`；`python -m pytest tests/test_indexing.py -q`。
- 测试结果：Ruff 通过；索引测试 9 passed。
- 评测结果变化：这个改动会改变新建索引的词频统计，但旧索引没重建之前不宣称任何提升。下一步重建索引再比。
- 潜在退化：词表目前是一批固定的财务科目，后续要从真实失败案例扩充，同时防止为了迁就回归集而过拟合。

### 重建索引后的实际对比

- 用新切词逻辑重建了索引 `b384205bc3a39550f64d`。
- 同一 16 题回归集、取前 5 名、候选池 20、关掉过滤和路由的条件下：
  - 改动前：Hit@5 0.6875，候选池召回 0.6875，MRR@5 0.4531
  - 改动后：Hit@5 0.6875，候选池召回 0.6875，MRR@5 0.4531
- 结论：没有可观测提升，也没有退化。说明当前回归集的问句已经有足够的二字组合信号，整词还没改变排序。下一步应转向由失败案例驱动的字段权重或查询改写。
- 完整对比：[holdout-eval-v2-bm25-term-enhanced-comparison.md](../../reports/ranking/holdout-eval-v2-bm25-term-enhanced-comparison.md)。

## 2026-08-06：生成评测加深 + 抗干扰赛道闭环

### 修改前基线

- 数据集 32 题：24 该答、7 该拒答、1 该追问，72 个原子事实、39 条证据。
- 虽然声明了抗干扰赛道，但干扰段落数是 0，脚本也只支持前两条赛道。
- 扩充后的 36 道该答题里有 17 道存在风险："绑定的原文引用没包含全部直接数值事实"。当时的校验器只能验证引用是切片的子串，验证不了语义上的契约。
- 同一段切片被拆成多段引用时，"需要几个引用"算错了——按引用段数算，而不是按独立上下文数算。
- 季度利润核对题的参考答案只覆盖了 5 个必需事实里的 2 个。
- 生成器最多用 3 段上下文，装不下需要 4 段证据的关键审计事项题。

### 修改内容

- 数据集扩充并重审为 48 题：37 该答、9 该拒答、2 该追问，40 个考点、120 个原子事实、67 条证据、35 段涉及的切片。
- 新增几类难题：利润口径与非经常性损益的勾稽、季度与年度核对、合并与母公司口径、分红实施状态、审计比较。
- 29 道题绑定 53 个真实年报干扰段落；冻结测试集 12/12 全覆盖。
- 新增可运行的抗干扰赛道，保存每段上下文的标签（正确证据 / 检索来的 / 哪一类干扰）。
- 上下文预算从 3 段提到 5 段；真实检索赛道同步取前 5 名。
- 校验器新增五项：直接数值事实必须包含在绑定引用里、参考答案必须覆盖全部必需事实、引用按独立上下文计数、干扰段落存在性和去重、跨页证据的完整页范围。
- 新增配对比较脚本，拒绝跨数据集或跨赛道比较，并输出哪些修好了、哪些退化了。
- 增加通用的成本 / 利润口径追问策略，不依赖具体公司或固定问句全文。

### 数据覆盖变化

| 项目 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| 题数 | 32 | 48 | +16 |
| 原子事实 | 72 | 120 | +48 |
| 正确证据 | 39 | 67 | +28 |
| 干扰段落 | 0 | 53 | +53 |
| 可运行赛道 | 2 | 3 | +1 |
| 冻结集的抗干扰覆盖 | 0/9 | 12/12 | 全覆盖 |
| PDF 复核告警 | 0 | 0 | 保持 0，且改成检查完整跨页范围 |

### 追问策略的配对结果（同一数据集版本）

| 赛道 | 行为准确率：前 | 行为准确率：后 | 变化 | 修好 / 退化 |
|---|---:|---:|---:|---:|
| 直接给答案 | 0.9583 | 1.0000 | +0.0417 | 2 / 0 |
| 真实检索 | 0.7917 | 0.8333 | +0.0417 | 2 / 0 |
| 抗干扰 | 0.7241 | 0.7931 | +0.0690 | 2 / 0 |

修好的两题：`u_moutai_profit_ambiguous`、`u_yili_cost_scope_ambiguous`。完整配对产物在 `reports/generation/comparisons/`。

### 结论边界

- 上述结果来自不接大模型的确定性链路，只能证明脚本、行为策略和回归门禁能工作，不代表 DeepSeek 的生成质量。
- 旧 32 题的 DeepSeek 结果不能搬到当前数据集版本上，三条赛道和语义评测都要重跑。
- 只有两份 2024 年报，标准答案也没有第二人独立复核，所以定位是"不算玩具的冻结回归集原型"，不宣称生产级或跨文档泛化。

## 2026-08-06：评测覆盖审计 + 单位判定修复

### 修改前

- 语义评测脚本死板地要求每次运行都包含全部 37 道该答题，所以只有 29 题的抗干扰赛道根本跑不了语义评测。
- 分红题把 `元/股` 当标准单位，但参考答案写的是"每股 x 元"，导致参考答案自检的单位准确率只有 0.5——自己的标准答案都过不了自己的判定。
- 数据卡把 25 道"引用了多段原文"的题统称为"多证据题"，没区分"同一段切片里引了好几处"和"真的需要好几段独立证据"。后者才是跨上下文推理。

### 修改后与验证

- 语义评测改成按赛道绑定：前两条赛道必须完整覆盖 48 题，抗干扰必须精确覆盖那 29 题。抗干扰里有 18 题该答，所以覆盖率 `18/37`、赛道内覆盖率 `1.0`，并保存完整的题号审计字段。
- 单位判定接受 `1.20元/股`、`每股1.20元`、`1.20元每股` 这些等价写法，同时严格区分元、万元、亿元的量级边界。37 道该答题的参考答案自检从 1 个失败降到 0。
- 数据卡分开记录 25 道"多段引用"和 13 道"多段独立证据"的题，防止在项目陈述里夸大跨上下文推理的覆盖面。
- 定向验证：生成与语义评测 17 passed；全量 68 passed；Ruff 全仓通过；启动门禁通过（16 题检索回归集、48 题生成集、2 次实验记录）。
- Windows 上的产物加载脚本显式用 UTF-8 读中文 JSON，消除了"文件内容其实是对的，但启动门禁因为默认编码误判而失败"的问题。

### 结论边界

- 当前评测集足以验证语料范围内的证据、行为和抗干扰能力，但 37 道该答题里只有 13 道需要多段独立证据。
- 下一轮不该继续从这两份年报里机械加题。优先加 4–6 家公司、2–3 个年度，以及 OCR / 扫描件和跨页表格文档，把整份文档留出来做盲测，再把"需要多段独立证据"的比例提到至少 50%。
