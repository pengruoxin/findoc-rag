# FinDocRAG 双层评估体系

## 检索侧

- Recall@K：所有已标注相关 chunk 中，被 top-k 找回的比例。
- Precision@K：top-k 中相关 chunk 的比例。
- MRR@K：第一个相关 chunk 排名倒数的均值。
- NDCG@K：考虑相关结果位置的排序质量；当前 holdout 为二元 relevance，未来可扩展 graded relevance。

加权 RRF 当前结果（16 条 holdout，K=5）：

| Recall@5 | Precision@5 | MRR@5 | NDCG@5 |
|---:|---:|---:|---:|
| 0.6875 | 0.1375 | 0.3948 | 0.4666 |

过去报告中的 Hit@5 只表示“至少命中一个”，不是严格 Recall@5。当前数据每题通常只有一个 relevant chunk，因此两者暂时数值相同；多证据题会自然拉开二者。

## 生成侧数据集

当前 `generation-eval-v1-b7f4d6113c96` 包含：

- 48 条问题：37 可回答、9 应拒答、2 应澄清。
- 120 个原子事实、67 条 gold evidence、35 个唯一 gold chunk。
- 29 条受控 robustness 问题、53 个真实年报 hard negatives。
- calibration/dev/frozen_test 为 12/24/12，frozen_test 12/12 带干扰。

每个直接财务数值必须出现在绑定 quote 中；每个 required 数值/布尔事实必须出现在 reference 中。引用数量按唯一 gold context 计算，跨页 evidence 必须完成整段页码范围的 PDF 视觉复核。

## 三条生成轨道

1. Oracle Context：只给 gold evidence，衡量生成与回答策略上限。
2. Retrieved Context：走真实 Hybrid、routing 和 guardrail，衡量端到端系统。
3. Robustness：把错公司、错期间、错口径、部分证据和错误因果段落插入 gold 之间；拒答题也会得到诱导性证据。

Oracle 好、Retrieved 差通常指向检索或路由；Oracle 也差通常指向生成、表格解析或 prompt；Robustness 单独下降说明模型容易被高相似干扰污染。

## 指标与门禁

确定性主指标：

- Expected behavior accuracy
- Strict success
- Gold fact recall / numeric accuracy
- Unit accuracy
- Citation validity
- Deterministic context recall
- Abstention / clarification accuracy
- Run error rate

语义诊断使用 RAGAS：Faithfulness、Answer Relevancy、Context Relevance、LLM Context Recall。拒答题不进入普通 Faithfulness 均值，避免靠大量拒答刷高分。

当 DeepSeek 同时是 answer model 与 judge 时，产物必须记录 `independent_judge=false`。同模型自评只作为辅助诊断，不能覆盖精确数值、单位、公司、年份、口径和引用门禁。

### RAGAS 运行范围与审计门禁

RAGAS 不再假定每条 run 都包含数据集中的全部可回答问题，而是先验证 run 的 lane，再取“该 lane 实际问题集合”和“数据集可回答问题”的交集：

- `oracle_context` 和 `retrieved_context` 必须完整包含当前数据集 48 条问题，少一条或混入未知问题都会在调用 Judge 前失败。
- `robustness` 必须精确包含带 hard negative 的 29 条问题；它是受控子集，其中 18 条可回答，因此语义评测覆盖率为 `18 / 37`。
- 重复 query ID、数据集外 query ID、错误 Dataset ID、错误 Index ID、run error、lane 范围不一致，以及 answer/grounded/context 结构自相矛盾，都会在远程调用前失败。
- `observed_behavior` 与 answer contract 不一致属于模型质量失败，不属于产物损坏。错误回答、错误拒答和错误澄清仍保留在 gold-answerable eligible 集合中，同时写入 `behavior_mismatch_query_ids`；如果因此中止或剔除样本，会制造幸存者偏差并虚高 RAGAS 分数。

RAGAS 输出保存以下审计字段：`run_id`、`lane`、`scope_policy`、`dataset_query_count`、`dataset_answerable_count`、`lane_query_count`、`lane_answerable_count`、`run_item_count`、`eligible_count`、`coverage`、`lane_coverage`、`behavior_mismatch_query_ids`，以及 run/lane/eligible/excluded 的完整 query ID 列表。每条 RAGAS row 也回填 `query_id`，便于与确定性分数和失败案例逐题关联。

三条 DeepSeek run 完成后可以分别运行语义评测，例如 Robustness：

```powershell
.venv\Scripts\python.exe scripts\run_ragas_generation_eval.py `
  reports\generation\runs\robustness-generation-eval-v1-b7f4d6113c96-deepseek-chat\items.jsonl `
  --output reports\generation\ragas-robustness-b7f4d6113c96.json
```

这里的 `coverage` 以数据集全部 37 条可回答题为分母，便于比较三条 lane 的覆盖面；`lane_coverage` 以该 lane 的可回答题为分母，用来证明本次 run 没有在 lane 内漏题。Robustness 的正常值分别为 `18/37` 和 `1.0`。

## 当前确定性基线

| Lane | 版本 | Strict success | 行为准确率 |
|---|---|---:|---:|
| Oracle | baseline | 0.2571 | 0.9583 |
| Oracle | clarify-v1 | 0.3143 | 1.0000 |
| Retrieved | baseline | 0.0286 | 0.7917 |
| Retrieved | clarify-v1 | 0.0857 | 0.8333 |
| Robustness | baseline | 0.1364 | 0.7241 |
| Robustness | clarify-v1 | 0.2273 | 0.7931 |

这只是无 LLM 链路基线，用于验证评测闭环；旧 32 条 DeepSeek 分数不属于当前 Dataset ID。新版 DeepSeek 三轨与 RAGAS 结果需要重新运行后再发布。

## 结论边界

当前可称为“带 source-verified gold、受控干扰和三轨 runner 的中文财务 RAG 冻结回归集原型”。由于仍只有两家公司、一个年份且没有独立人工 gold，不能称为跨文档泛化或生产级 benchmark。
