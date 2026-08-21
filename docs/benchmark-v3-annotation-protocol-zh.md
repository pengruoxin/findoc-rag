# Benchmark v3 独立标注协议

## 隔离边界

- 语料分配已在出题和规则调优前密封，分配单位为公司；公司不得跨 split。
- 日常开发只能使用 `calibration` 和 `development` 索引。`frozen_test` 索引只在候选版本、提示词、阈值和规则全部锁定后运行。
- frozen-test 标注者不得查看被测系统的回答、检索结果或失败案例；开发者不得根据 frozen-test 结果修改系统后继续把同一批题称为冻结测试。
- `benchmark-v2` 只做历史回归，不得并入 v3 的独立 gold。

## 题量与覆盖

每份文档目标 6 题。每个 split 内应覆盖单事实、多事实/表格、计算、叙述或会计政策，以及不可回答或需要澄清行为；不能用六个同模板数字题凑数。每题另写两个语义等价但词面不同的问法变体。

总目标为：calibration 12 题、dev 24 题、frozen test 24 题。跨文档题只能引用同一 split 中的文档，且涉及的每家公司都必须属于该 split。

## Gold 生产

1. 出题者直接依据官方 PDF 编写问题，不读取系统答案。
2. 对每个必要事实记录主体、期间、口径、规范值、单位、容差和推导式。
3. 每个事实绑定精确的 `document_version_id`、chunk、页码、章节和最短充分原文；页面视觉核验通过后才将 `pdf_visual_verified` 设为 `true`。
4. 参考答案只能陈述已绑定事实，并逐项引用；外部知识不得补足 PDF 中不存在的内容。
5. 不可回答题不得携带 gold evidence，并必须记录明确的拒答或澄清理由。

## 双人审核

每题需要两名非出题者独立批准。两人都必须分别核对问题语义、证据、参考答案和问法变体；只填写 `human_verified` 标签不算审核。任何一人给出 `revise` 或 `reject`，该题都不能进入冻结集。

审核身份使用稳定匿名 ID。最终发布门禁为：`independent_gold=true`、`status=human_frozen`，治理审计无 blocker，且 frozen-test 至少 24 题。

## 防污染操作顺序

1. 在 calibration 上修正明显实现错误；
2. 在 dev 上选择检索、路由、停止条件和提示词；
3. 锁定代码提交、配置哈希、模型和三个 split 的 index ID；
4. 只运行一次 frozen test 并生成不可变报告；
5. 若依据 frozen 结果继续调优，必须重新分配未接触过的公司和文档作为下一版 frozen test。
