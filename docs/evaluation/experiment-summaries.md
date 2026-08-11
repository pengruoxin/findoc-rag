# FinDocRAG 实验分析总结索引

> 每个实验的完整分析放在 `reports/<stage>/<experiment-id>/analysis.md`，与 config / per-query 数据同目录，保证可复现。
> 本文件是分析总结的统一入口：一行摘要 + 链接 + 状态。
> 变更记录（改了什么、基线、结果、退化）见 [history/optimization-log-zh.md](../history/optimization-log-zh.md)；最新基线数字见 [baseline-zh.md](./baseline-zh.md)。

## 实验记录

| 实验 | 日期 | 想回答什么 | 关键结论 | 完整分析 |
|---|---|---|---|---|
| variant-regime-v1 | 2026-08-07 | 96 个变体问句的首轮检索评测：系统对三种真实问法的能力差异（清单 P1-5 / P1-6） | 语义检索只在"股票代码 / 简称"问法上有效（0.676，原题只有 0.216）；口语和相对时间问法全员弱（同义词改写导致 11 题三路全没找到）；混合检索被弱语义分支拖累（4 题正确证据从第 1–2 位被挤出前 5）；时间对齐 bug 实锤——问句解析出的年份不等于文档报告年份时过滤会出错 | [analysis.md](../../reports/ranking/variant-regime-v1/analysis.md) |
| fusion-sweep-v1 | 2026-08-07 | 融合权重扫描：调权能不能消除语义分支的负优化（P1-5 下一轮） | **只用关键词检索、完全不掺语义检索，在三种问法上都碾压任何融合权重**（0.838 / 0.811 / 0.730）；逐问法挑最优也全都选"不掺"；给语义路任何正权重都是拖累 → 默认策略已改成纯关键词检索 | [analysis.md](../../reports/ranking/fusion-sweep-v1/analysis.md) |
| table-eval-v1 | 2026-08-07 | 表格抽取评测层（P1-11）：8 张表 149 个单元格三元组标注 + 抽取器接口 + 正则基线 | 季度表基线 28/32（0.875）；茅台扣非行 4 格全错——标签在数值后面，正则把下一行现金流当成了答案，直接解释 Oracle 只有 0.31；其余表型尺子已就位、抽取器未实现 | [analysis.md](../../reports/ranking/table-eval-v1/analysis.md) |
| context-metrics-v1 | 2026-08-07 | 把上下文效率与延迟变成每个实验的标准指标 | 生成侧 `avg_context_tokens` ≈1536（真实检索赛道）、`p95_latency_ms` ≈20（确定性链路）；检索侧 `avg_evidence_tokens_top5`：关键词 307 / 语义 386 / 混合 348——语义分支用更贵的上下文换更差的结果 | 指标定义见 [benchmark-and-metrics-zh.md](./benchmark-and-metrics-zh.md)；基线数字见 [baseline-zh.md](./baseline-zh.md) |
| deepseek-baseline-v1 | 2026-08-07 | 48 题新评测集第一次跑真实 DeepSeek 三轨 + RAGAS（P2-14） | **给对证据就 97% 全对**（Oracle 0.9714），真实检索只有 0.5429——瓶颈明确在检索 / 路由 / 证据选择；抗干扰 0.5455，说明模型会被高相似干扰带偏；上下文真实数字：1536 token / p95 ≈1.8s | 基线数字见 [baseline-zh.md](./baseline-zh.md) §3.1；变更见 [optimization-log](../history/optimization-log-zh.md) |
| variant-regime-expanded-v2 | 2026-08-07 | 同义词查询改写：从 10 条失败题提取 7 组财务同义词映射，能否救回口语问法 | **口语问法 Hit@5 0.73 → 0.92、原题 0.84 → 0.89、代码/简称 0.81 → 0.89，零回归**；但真实检索 strict 持平（0.5429）——逐题归因：8 行为拒答 + 5 表格/生成 + 3 检索，端到端瓶颈已转移到行为与表格 | [analysis.md](../../reports/ranking/variant-regime-expanded-v2/analysis.md) |
| oov-eval-llm-v1 | 2026-08-11 | 词表外改写（OOV）对 deterministic 词表免疫，LLM 查询改写能否救回？中文 dense（bge-small-zh-v1.5）能否替代/补充？（清单 P2-17 / 阶段 0） | **LLM 改写：lexical Hit@5 0.194 → 0.694（MRR 0.148 → 0.498）**；deterministic 0.194 无效果；bge-zh dense 0.083 < E5 0.139，三种问法全面退化；LLM 改写后 hybrid 仍低于 lexical（0.472 vs 0.694）→ 默认 lexical-only 不变 | [analysis.md](../../reports/ranking/oov-eval-llm-v1/analysis.md) |
| variant-regime-bge-zh-v1 | 2026-08-11 | bge-small-zh-v1.5 在变体矩阵上的 dense 对照 | dense 三问法 0.162 / 0.568 / 0.162，均低于 E5（0.216 / 0.703 / 0.189）；ticker 唯一优势形态也退化 → 换中文小 dense 模型在当前输入上无收益 | [analysis.md](../../reports/ranking/variant-regime-bge-zh-v1/analysis.md) |
| table-eval-v2 | 2026-08-11 | B 阶段表格重建第一步：四类表型（quarterly / note_cost / segment / annual_data）的确定性单元格抽取器 | **28/149 → 146/149（98.0%）**：quarterly 32/32（修好茅台扣非行）、note_cost 24/24、annual_data 39/39、segment 51/54（唯一残差为 PDF 文字层丢"地区"）；8 个单测 | [analysis.md](../../reports/ranking/table-eval-v2/analysis.md) |
| table-generation-v1 | 2026-08-11 | 把单元格抽取接入 answer_generation 确定性表格路径（带引用），no-LLM 三轨是否提升（清单 P1-11 收口） | **Oracle strict 0.3143 → 0.6571（+12 题）、Retrieved 0.0857 → 0.2571（+6 题）、Robustness 0.2273 → 0.3636（+3 题），行为不变、零回归**；Retrieved 剩余 6 题卡在证据不在 top-5 | [配对报告](../../reports/generation/comparisons/oracle-table-v1.json) |
| deepseek-table-v2 | 2026-08-11 | 表格路径接入后重跑 DeepSeek 三轨 + RAGAS，并修复 `api_model` 元数据（`--model` 只是 run 标签，之前 `independent_judge` 被误标 True） | **Oracle strict 0.9714（持平，单题互换）、Retrieved 0.5714（+1）、Robustness 0.6364（+2，零回归）**；RAGAS faithfulness 0.778 / 0.845 / 0.849 全部提升；新产物 `api_model_recorded=true`、`independent_judge=false` | [配对报告](../../reports/generation/comparisons/robustness-deepseek-table-v2.json) |
| deepseek-abstain-v2 | 2026-08-11 | 远程拒答检测修正打分口径：伪拒答（拒答文本里带数字）不再刷分，应拒答题的拒答被如实计分 | **Retrieved strict 0.5714 → 0.8000（+8 应拒答 / -0）、Robustness 0.6364 → 0.8636（+6 / -1 怪癖消除）**；行为 0.8958 / 0.8276 为真实基线；Oracle 0.9429（单题波动） | [配对报告](../../reports/generation/comparisons/retrieved-abstain-v2.json) |
| deepseek-table-remote-v1 | 2026-08-11 | 受控实验：远程模式启用确定性表格优先（单变量开关），消除表格类可答题误拒答 | **Oracle strict 0.9429→1.0、Retrieved 0.80→0.8286、Robustness 0.8636（持平）**；行为 1.0 / **0.9583** / 0.8621，零回归；新 run 带 code_revision | [配对报告](../../reports/generation/comparisons/retrieved-table-remote-v1.json) |
| concentration-table-v1 | 2026-08-11 | 新增 concentration 表型（前五名客户/供应商集中度），消除剩余集中度误拒答 | 抽取 8/8 单元格；Robustness 行为 0.8621→0.9655；v1 run 暴露"负例前置取错公司"bug（保留审计） | [表格评测](../../reports/ranking/table-eval-concentration-v1/summary.md) |
| concentration-bugfix-v2 | 2026-08-11 | 修复单公司 concentration 按查询公司选题（受控 bugfix） | **Robustness strict 0.8636→0.9545（+2）、行为 0.9655，零回归**；Oracle 1.0；Retrieved -1 行为为无关模型波动 | [配对报告](../../reports/generation/comparisons/robustness-concentration-v2.json) |

## 分析总结规范

每个 `analysis.md` 必须包含：

1. 实验配置与重跑命令，保证可复现
2. 关键发现：每条都带量化证据（表格 / 数字 / 具体是哪几道题）
3. 失败归因：定位到具体哪一层（检索 / 路由 / 时间解析 / 融合），并说明根因
4. 下一步建议：按优先级排，指向改进清单的对应条目
5. 结论边界：样本量多大、标注是否完整、哪些结论不能外推

新增实验时依次做四件事：把行加入上表 → 在[变更日志](../history/optimization-log-zh.md)记录改了什么 → 如果数字成为新基线就更新[基线文档](./baseline-zh.md) → 更新实验注册表 `reports/ranking/experiment-registry-v1.json`。
