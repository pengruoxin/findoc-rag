# Agent P4-G 表格 Cell / Geometry Proof 日志

## 1. 问题

P4-F 的人工审核包已经能展示 claim、chunk 原文和 PDF 页码，但表格数值仍只能回答
“来自这个 chunk”，不能回答：

- 数字属于哪一张表；
- 是哪一行、哪一列；
- 数值在 PDF 页面的准确位置；
- 坐标或逻辑标签被改过后能否被检测。

代码审计发现，现有结构化 sidecar 虽有 `row/column/value`，坐标在
`reconstruct_cells → StructuredTableCell` 转换时被丢弃；AgentEvidence 也没有表格证明。
因此本阶段从 PDF IR 重建源头修复，不在审核层猜坐标。

## 2. 实现

### 2.1 Sidecar schema v2

`StructuredTableCell` 新增：

- `row_index` / `column_index`：1-based 逻辑行列序号；
- `page_number`；
- `value_bbox`；
- `coordinate_space=pymupdf_unrotated_page`。

`ExtractedCell` 从数值 token 保留 page 和 bbox；季度长标签修复、分部 section 修复都必须
继续携带该 provenance。新 sidecar 标记为 schema 2 / `coordinate-safe-v3-geometry`。

读取器同时接受历史 schema 1 / `coordinate-safe-v2`，旧索引不会因升级突然不可用；新建或
重建的索引才写 schema 2。结构化 sidecar 仍排除在 chunk identity 外，因此没有改写 chunk
或 benchmark index ID。

### 2.2 保守补坐标

坐标重建直接选中的单元格使用数值 token 的原始 bbox。文本 fallback 不默认宣称有坐标；
只有同一单元格的 `section + row + column + value` 在坐标候选中存在**唯一精确匹配**时，
才把 bbox 附到 text-selected cell。歧义、多匹配或无匹配继续标为 `text_only`。

### 2.3 防篡改 Cell Proof

每个单元格生成 `TableCellGeometryProof`，绑定：

- table ID/type/source；
- chunk ID 与 persisted chunk SHA-256；
- table page range、unit、section；
- row/column label 与 index；
- value；
- geometry status、page、bbox、coordinate space。

上述字段规范化后再次计算 `binding_sha256`。加载 proof 时本地重算，改 value、行列、页码、
bbox、table 或 chunk 任何一项都会失败。sidecar 的 chunk hash 与当前检索 chunk 不一致时，
proof 在进入 Agent Evidence Memory 前就 fail closed。

### 2.4 Agent 与人工审核

`add_evidence` 把表格 proof 写入 `AgentEvidence.table_cell_proofs`。P4-F 创建审核包时，按
requirement description、candidate claim、row/column/value 精确筛选相关 cell，不把整张表
无差别塞给 reviewer。`agent review inspect` 现在显示：

- table ID；
- `row[index]`、`column[index]`；
- value 与 unit；
- page 与 bbox，或明确的 `text-only`；
- cell proof SHA-256。

这些 proof 不进入 DeepSeek prompt，因此不增加回答 token。

## 3. 真实 PDF IR 评测

命令：

```powershell
uv run python scripts/evaluate_agent_table_cell_proofs.py
```

输入固定为 agent-hard-v3 的 calibration + dev：8 份年报、5,012 chunks。评测器显式只加载
`benchmark_split in {calibration, dev}`，不读取 frozen test。基线绑定升级前的固定 generation：

`data/indexes/agent-hard-v3/development/generations/20260820T055043-3f82e702`

| 指标 | schema 1 基线 | P4-G |
|---|---:|---:|
| 表格数 | 11 | 11 |
| 单元格数 | 111 | 111 |
| 单元格语义 exact | 111/111 | 111/111 |
| 逻辑行列 index | 0/111 | **111/111** |
| 有 PDF value bbox | 0/111 | **102/111（91.89%）** |
| 坐标路径 bbox 覆盖 | 0/72 | **72/72** |
| 文本 fallback 唯一匹配补坐标 | 0 | **30** |
| 防篡改 proof | 0/111 | **111/111** |
| 模型请求 / token | 0 / 0 | 0 / 0 |

按表型：

| 表型 | 单元格 | 有 bbox | text-only |
|---|---:|---:|---:|
| quarterly | 48 | 48 | 0 |
| note_cost | 60 | 54 | 6 |
| annual_data | 3 | 0 | 3 |
| **总计** | **111** | **102** | **9** |

9 个剩余格没有被强行赋坐标。它们是下一轮页面区域/表头锚定要解决的真实残差，而不是用
chunk bbox 冒充 cell bbox。

原始报告：`reports/agent/agent-p4g-table-cell-geometry-proof-v1.json`。

## 4. 生产状态

已仅用非冻结 split 重建本地活跃索引：

- calibration：schema 2，32 cells；
- development（calibration + dev）：schema 2，111 cells，其中 102 cells 有 bbox。

旧 generation 保留，作为可复现 schema 1 基线。chunk/index identity 未改变，sidecar generation
已更新。

## 5. 测试与安全用例

- 数值 proof 任一字段被改，binding SHA 校验失败；
- sidecar chunk hash 错误时，AgentEvidence 写入前拒绝；
- text fallback 明确为 `text_only`；
- PDF IR 数值 token 的 page/bbox 跨重建保留；
- sidecar schema 1 仍可读取；
- AgentEvidence JSON round-trip 后 proof 不丢失；
- 人工审核命令显示 row/column/page/bbox/proof hash；
- P4-G/PDF/Agent 定向：80 passed；
- 全仓：**413 passed**；
- Ruff 与 `git diff --check`：通过。

## 6. 结论边界

P4-G 提升的是**表格证据可验证性**，不是 DeepSeek 问答准确率；本阶段没有重新跑远程回答，
45/48 保持历史结果但不能声称因 P4-G 提升。

目前的 bbox 只精确绑定数值 token，行名和列表头仍是逻辑 label/index，没有各自独立 bbox；
也没有生成 PDF 区域截图或像素 hash。当前 8 份非冻结年报没有证明真实扫描表泛化，9 个
text-only 格也尚未闭环。

下一步建议做小而明确的 P4-H：根据 value bbox 加有限上下文裁剪，连同 row/header anchor
生成可视化 region proof；只对 9 个 text-only 和人工升级案例触发，不把整页或整份 PDF 发给
DeepSeek。评测应报告 9 格修复数、误绑定率、审核耗时和新增视觉 token。
