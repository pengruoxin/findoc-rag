# PDF Hard v2：复杂表格评测与标注协议

## 目标与边界

`pdf-hard-v2` 的评测单位是独立表格，不是问答题。它用于比较 native、OCR、表格结构
模型和局部视觉后端的提取结果，再把通过门禁的证据交给 RAG。旧的
`pdf-extraction-v1` 和 `pdf-cross-page-v1` 只作为种子回归材料，不填充新基准的正式
配额。

受控栅格化页面必须标为 `controlled_rasterization`，不得描述为真实扫描件。正式的
`genuine_scan` 样本必须来自原本就是扫描图像的 PDF，并保留来源和文件 SHA-256。

## 规模与分层

每个分层计划 10 张独立表格：calibration 3、development 3、frozen 4，总计 70 张。

1. `native_control`：可靠原生文本层；
2. `genuine_scan`：真实扫描 PDF；
3. `degraded_text_layer`：乱码、缺字、隐藏层错序或受控退化；
4. `borderless_table`：无框线或弱框线；
5. `merged_hierarchical_header`：多级表头和合并单元格；
6. `cross_page_continuation`：纵向或横向跨页续表；
7. `rotated_or_mixed_layout`：旋转、倾斜、水印或图文混合。

一个案例可以有多个困难标签，但只用 `primary_stratum` 计算配额，避免重复计数。

## Gold 字段

每张表至少标注：

- 表格在每页的区域；
- 单元格原文和规范化值；
- 行列索引；
- `row_span`、`column_span`；
- 行表头路径和列表头路径；
- 单元格页码和坐标；
- 跨页 segment 之间的 continuation edge；
- 标注者、复核者、来源哈希和区域哈希。

分层表头不能只保存最后一级。例如“2024 年 / 调整后”必须保存为
`["2024年", "调整后"]`。合并单元格不得把一个值无来源地复制成多个独立事实。

## 标注流程

1. 数据管理员只准备 PDF、页码、表格区域和来源清单，不提供候选单元格值；
2. 标注者根据渲染页创建 cell graph；
3. 第二人复核数值、表头路径、span 和坐标；
4. calibration/development 可在复核后向开发代码开放；
5. frozen 的 gold 与开发工作区分离，只暴露清单和哈希；
6. 阈值、路由和后端版本冻结后，由未参与调参的新会话或独立复核者运行 frozen；
7. 候选确认只能记为 candidate verification，不能写成 blind annotation。

本项目此前的冻结材料已在开发会话中出现过，因此只能继续作为回归集；新的正式
frozen 必须重新封存并由独立执行者完成最终评测。

## 解析器输出契约

所有后端先转换为 `TableEvidenceCandidate`：

- 每个 cell 保留后端、版本、置信度、页码、坐标和 region SHA；
- 跨页表由 page-local segment 加显式 continuation edge 表示；
- native、OCR、结构模型和 VLM 均产生候选，不直接覆盖现有证据；
- `TableExtractionDecision` 只能选择已列出的候选；
- 有冲突但无法证明哪个正确时必须输出 `manual_review`。

视觉模型只接收发生冲突的局部区域和候选结构，不接收 gold，不负责最终评分。

## 指标

每次消融至少报告：

- 路由准确率和不必要升级率；
- 表格检测 precision/recall 和 false-table 数；
- 单元格规范化值 exact/F1；
- 行列、表头路径与 span F1；
- continuation-edge F1；
- 坐标证据完整率；
- unsafe auto-accept rate；
- manual-review rate；
- 页面/区域数量、耗时、token、显存和失败率。

最终问答仅作为下游指标，并固定检索、回答模型、prompt 和索引，不用问答波动替代
提取层硬指标。

## 当前种子清单

`data/evaluation/pdf-hard-v2/manifest.json` 当前包含 5 个旧回归案例、4 个原生复杂页
候选，以及 4 个真实扫描财务表候选。原生候选页面保存在
`development-candidates.pdf`；扫描候选保存在
`genuine-scan-development-candidates.pdf`。两组均记录来源页、官方来源和 SHA-256。

这 8 页目前只是 `assistant_visual_reviewed_unannotated`，还没有完整 cell graph 和第二人
复核；扫描集另有 30 个 `assistant_curated_provisional` 开发探针，但不能替代正式标注。
所有案例仍为 `counts_toward_target=false`，所以正式完成度保持 0/70。这是刻意的诚实
边界。

候选发现器只扫描 source manifest 中的 calibration/dev 文档，报告固定记录
`frozen_documents_opened=false`。两轮视觉抽查分别发现“日期密集正文”和“两列金额
对齐的贷款说明”假阳性，现已增加财务数值 span、重复数值列和至少三列网格门禁。当前
1,597 页中保留 7 个高置信无框线候选，不为凑满 12 个而放宽门槛。

执行清单完整性审计：

```powershell
uv run python scripts/validate_pdf_hard_v2.py
uv run python scripts/discover_pdf_hard_v2_candidates.py
uv run python scripts/prepare_pdf_hard_v2_genuine_scans.py
uv run python scripts/evaluate_pdf_hard_v2_genuine_scans.py
```

只有准备正式冻结时才使用 `--require-complete`。常规开发审计在配额未满时仍成功，但
报告中的 `ready_for_formal_freeze` 保持 `false`。
