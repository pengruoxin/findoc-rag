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

## 分析总结规范

每个 `analysis.md` 必须包含：

1. 实验配置与重跑命令，保证可复现
2. 关键发现：每条都带量化证据（表格 / 数字 / 具体是哪几道题）
3. 失败归因：定位到具体哪一层（检索 / 路由 / 时间解析 / 融合），并说明根因
4. 下一步建议：按优先级排，指向改进清单的对应条目
5. 结论边界：样本量多大、标注是否完整、哪些结论不能外推

新增实验时依次做四件事：把行加入上表 → 在[变更日志](../history/optimization-log-zh.md)记录改了什么 → 如果数字成为新基线就更新[基线文档](./baseline-zh.md) → 更新实验注册表 `reports/ranking/experiment-registry-v1.json`。
