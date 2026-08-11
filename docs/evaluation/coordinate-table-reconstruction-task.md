# 任务书：FinDocRAG 坐标级表格重建（Coordinate-level Table Reconstruction）

> 交给更强模型执行前，请先通读本文件，再按需阅读下文列出的代码与数据。执行时必须遵守 [实验协议](./experiment-protocol-zh.md)。

## 0. 项目背景与仓库现状

仓库：`D:\202607\raglab`（git，当前 main，工作区干净）。项目是面向中国上市公司年报的可追溯 RAG：PDF Document IR（pymupdf）→ 结构感知 chunking → 词法检索（默认 lexical-only + 确定性同义词）→ 表格确定性抽取 → 生成（确定性表格优先，DeepSeek 远程兜底）。

当前关键基线（均为受控实验产物，见 [baseline-zh.md](./baseline-zh.md)）：

- 表格抽取：五类表型，`table-eval-v1` 146/149 + `table-eval-concentration-v1` 8/8；
- 生成：DeepSeek 三轨（oracle/retrieved/robustness）strict 1.0 / 0.8286 / 0.9545，行为 1.0 / 0.9583 / 0.9655（`concentration-v2` 基线，revision `e73412d`）；
- 全量测试 131 passed，ruff 干净。

## 1. 问题定义

现有表格抽取基于"线性化文本 + 正则"，有三个已知缺陷：

1. **阅读顺序破坏行列关系**：PDF 的 `sort=True` 文本顺序可能把行标签排在数值之后（茅台季度表扣非行），现有文本级 workaround 是针对具体格式的 hack，不通用；
2. **单元格几何信息丢失**：当前 IR 只保留 block 级 bbox（`DocumentElement.bbox`），line/span 级坐标没有入库，无法做行带聚类、列对齐、跨页合并；
3. **文本层丢字**：伊利 segment 表"其他地区"行在 PDF 文字层只有"其他"（已用 pymupdf 实测：页 20 该行 span bbox=(40.8,294.2,58.8,303.2)，全页无独立"地区" span）——**坐标合并无法修复，属 OCR/标注分歧**，不得通过硬编码改名掩盖。

## 2. 目标与验收标准

构建坐标级表格重建层，从 PDF 元素几何恢复表结构（表头、行标签、列、单位、跨页），产出结构化表格（单元格三元组 + 页码/bbox/element_id 溯源），作为 table extraction 与答案生成的更可靠输入。

必须满足：

1. 新增模块 `src/findoc_rag/table_reconstruction.py`（或等价命名），输入 ParsedDocument（或 pymupdf 原始 dict），输出结构化表格；
2. **扩展 IR 保留 line/span 级几何**：`documents/pdf.py` / `documents/models.py` 需保存 span/line 的 bbox、字体、字号、粗体（当前只有 block 级）。这是 schema 变更，先评估对 chunk_id / benchmark-v2 gold 引用 / 索引的影响；若 chunk 文本不变则 chunk_id 不变（chunk_id 依赖文档 content_sha256 + 文本），需实测确认；若必须重锚定，按协议做版本升级并记录；
3. table-eval 回归不降（146/149 + 8/8），并尽量修复"坐标可修复"的残差；"其他地区"要么实现 OCR 兜底（可选加分），要么明确标注为不可由坐标修复；
4. 接入 `scripts/evaluate_table_extraction.py`（提供坐标输入路径，评测规则为"有坐标用坐标、无坐标回退文本"）；
5. 单测 + 集成测试；全量 pytest 保持 131+ 全绿；ruff 干净；
6. 文档回填：baseline / improvement-list（P1-11）/ optimization-log / experiment-summaries / registry / handoff；
7. 按协议分步提交（每步先 commit，得到 clean revision 再跑实验；需要生成三轨对比时用 `compare_generation_runs.py --change "<单变量说明>"`）。

## 3. 已替你验证的事实与线索

- **数据位置**：
  - PDF：`data/artifacts/cninfo/600519_2024_1222993920.pdf`（茅台，143 页）、`600887_2024_1223421123.pdf`（伊利，270 页）；
  - ParsedDocument（block 级 bbox）：`data/catalog/versions/7961508deeffb5e66ae88808/document.json`（茅台）、`data/catalog/versions/e96cf669106c99e4e283ca45/document.json`（伊利）；`data/processed/documents/` 下只有茅台副本；
  - chunk 及其 element_references（element_id / page / bbox）：`data/catalog/versions/*/chunks.jsonl`；
  - 标注与评测：`data/evaluation/table-eval-v1.json`（8 表 149 格）、`data/evaluation/table-eval-concentration-v1.json`（2 表 8 格）、`scripts/evaluate_table_extraction.py`。
- **span 级信息可用**：`page.get_text("dict", sort=True)` → blocks → lines → spans（bbox / text / font / size / flags），当前 `pdf.py` 只取了 block 级；
- **伊利 segment 表在第 19–20 页**，页 20 最后一行"其他"无"地区"span（已实测）；
- **茅台季度表在第 5 页**，扣非行标签与数值 y 对齐但阅读顺序倒置（现有 `extract_quarterly` 用"4 数值一组 × 出现指标"规避）；
- **表型结构**（CSRC 监管格式）：quarterly（4 指标 × 4 季度）、note_cost（3 行 × 本期/上期×收入/成本）、segment（分行业/产品/地区/销售模式 子表，行=5–6 数值）、annual_data（年报/年末两段，4 数值含同比）、concentration（句子式，金额+占比）。

## 4. 设计约束

1. 保留 `extract_cells(text, table_type)` 文本接口（回归基线 + fallback）；坐标路径作为新入口，评测按"有坐标用坐标、无坐标回退"；
2. 不硬编码"其他地区"；诚实记录不可修复项；
3. 方案要能泛化到新公司/新年度（按监管表格格式设计，禁止只对 8 张表特判）；
4. 涉及 schema / chunk / 索引变更前，先评估对 benchmark-v2 gold 的影响并记录决策；
5. 每次改动：`pytest -q` + `ruff check src tests scripts`；记录模板照 `docs/history/optimization-log-zh.md`。

## 5. 交付物

1. 代码 + 测试（单测覆盖行带聚类、列对齐、跨页合并、文本回退）；
2. `reports/ranking/table-reconstruction-v1/`：analysis.md（坐标 vs 文本基线逐表对比）+ summary.json；
3. 文档回填（见 §2.6）；
4. 分步 commit（IR 扩展 → 重建层 → 接入评测 → 生成接入（如需）→ 记录）；
5. 明确的"不可由坐标修复"残差清单。

## 6. 建议实施顺序

- **P0**：span 级 IR 扩展 + 行带聚类原型 + 8+2 张标注表坐标重建 → table-eval 对比（目标：除"其他地区"外 100%）；
- **P1**：跨页表合并、列头/单位识别、接入 answer_generation（远程确定性表格优先路径）；
- **P2**：OCR 兜底（可选加分项）或"其他地区"正式标注为 divergence；扩展标注到更多公司/年度。

## 7. 常用命令

```powershell
.venv\Scripts\python.exe -m pytest tests -q --no-header -p no:cacheprovider
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\python.exe scripts\evaluate_table_extraction.py --data data/evaluation/table-eval-v1.json --output-dir reports/ranking/table-reconstruction-v1-text-baseline
.venv\Scripts\python.exe scripts\evaluate_table_extraction.py --data data/evaluation/table-eval-concentration-v1.json --output-dir reports/ranking/table-reconstruction-v1-concentration
git -C D:\202607\raglab commit -m "..."
```

生成三轨（如需验证端到端，需要 `DEEPSEEK_API_KEY`）与配对对比的用法见 `docs/DEVELOPMENT-HANDOFF-zh.md`；注意 key 只从 `data/raw/deepseek_key.txt` 读取且绝不提交。
