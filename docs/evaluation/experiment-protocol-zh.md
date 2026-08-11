# 控制变量实验协议

> 本协议是 FinDocRAG 所有优化实验的强制纪律。**没有控制变量的对比不是实验，只是猜测。**
> 导航：返回 [文档索引](../README.md) · 基线数字 [baseline-zh.md](./baseline-zh.md) · 改进清单 [improvement-list-zh.md](./improvement-list-zh.md)

## 1. 为什么需要这个协议（本项目的真实教训）

2026-08-11 出现过一组"看起来提升、实际不可归因"的对比：8/7 的 DeepSeek 基线跑在**同义词改写落地之前**的 runner 上，8/11 重跑时 runner 和打分口径都变了。逐题反推后才发现：Retrieved +1 里只有 1 题是真检索提升，其余是模型措辞波动；Robustness +2 里有 1 题是"拒答文本带数字"被打分器误判成全对。

结论：**分数变化必须能逐题解释，且解释必须指向唯一的受控变量。**

## 2. 单变量原则

一次实验只改一个变量，其余全部钉死：

| 必须钉死的变量 | 说明 |
|---|---|
| Dataset ID | 同一冻结集；跨版本数据集禁止直接对比 |
| Index ID | `data/indexes/corpus/current.json` 的 `index_id`；换索引 = 换实验 |
| 代码版本 | `summary.json` 的 `code_revision` + `code_dirty`；工作区有未提交改动也算脏 |
| Runner 参数 | top_k / candidate_k / rrf_k / 权重 / filters / 是否 query_parser |
| 模型身份 | `--model` 只是 run 标签；真实 API 模型看 `api_model`；判 judge 独立性也用它 |
| 远程开关 | `remote_generation`：LLM 链路 vs no-LLM 链路是两个世界，禁止互比 |
| 功能开关 | 例如 `FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES=1` 做 A/B 时只动这一个开关 |
| 测量口径 | scorer 版本、grounded/abstain 判定、RAGAS judge 模型；**测量变更不能与能力变更混在同一对比里** |

## 3. 对比合法性规则（脚本强制）

`scripts/compare_generation_runs.py` 现在强制执行：

1. 两个 run 的 `dataset_id`、`lane` 必须一致；
2. `remote_generation` 必须一致（LLM vs no-LLM 拒绝对比）；
3. `code_revision` 不同（或未记录）时，必须传 `--change "<单变量说明>"` 声明这次改了哪一个变量；否则脚本拒绝出报告；
4. 同一 `code_revision` 的重复 run 用于**稳定性/噪声估计**（`controlled_change=null`），不用于宣称能力变化。

报告里会落盘 `baseline_code_revision` / `candidate_code_revision` / `code_revision_match` / `controlled_change`。

## 4. 非确定性处理

远程模型即使 temperature=0 仍非确定（实测单题 ±1–2 波动）。因此：

- 单题 fixed/regressed 列表必须随报告输出，结论按题集看，不按单题看；
- 宣称"某改动提升 X 题"时，必须同时给出同版本重复 run 的噪声水平（例如 abstain-v1 与 abstain-v2 之间的波动）；
- 对临界结论（±1 题），跑两次取区间或明确写"在噪声范围内，不宣称"。

## 5. 测量与能力分离

当改动只影响"怎么算分"（如拒答检测、`grounded` 判定、scorer 规则）时：

- 这属于测量变更，产物必须标注（provider=`remote-abstention`、`api_model_recorded` 等）；
- 测量修正后的分数与修正前**不可直接对比**（如 strict 0.57 → 0.80 的"提升"其实是 8 条应拒答被正确计分）；
- 修正后需要重新建立基线（如 abstain-v2），后续能力改动与它对比。

## 6. 实验记录模板（与 optimization-log 共用）

```text
日期：
目标问题：
基线（run_id + code_revision + 是否脏）：
受控变量（只允许一个）：
固定变量清单（dataset / index / runner 参数 / api_model / remote / 功能开关）：
修改内容：
测试命令与结果：
评测结果（配对 fixed / regressed 列表 + 指标变化）：
已知退化/未覆盖：
结论：
```

## 7. 回填义务

每完成一个受控实验，必须：

1. 配对报告写入 `reports/generation/comparisons/`（含 `controlled_change`）；
2. 数字更新到 [baseline-zh.md](./baseline-zh.md)，旧基线保留并注明适用范围；
3. 一行摘要加入 [experiment-summaries.md](./experiment-summaries.md)；
4. 完整记录追加到 [optimization-log](../history/optimization-log-zh.md)；
5. 实验注册表 `reports/ranking/experiment-registry-v1.json` 增加 run 条目。
