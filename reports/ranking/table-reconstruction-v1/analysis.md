# 坐标级表格重建：集成与真实数据评测（P0）

## 交付与集成

- 外部模型交付 `table_reconstruction.py` + `test_table_reconstruction.py` + `REPORT.md`，已集成进仓库：
  - `src/findoc_rag/table_reconstruction.py`：span→token→行带聚类→数值列聚类→标签-数值配对；季度/年度/本期-上期/segment 子表/单位识别；`merge_pages` 接口；文本回退；
  - `tests/test_table_reconstruction.py`：11 个用例（含"标签后置""跨行标签"两个合成夹具），全绿；
  - 代码做了最小适配：legacy 回退改 import `findoc_rag.table_extraction`；修复 ruff 11 处（import 排序、盲异常、getattr 常量等）。
- 新增回归尺子：`scripts/evaluate_coordinate_reconstruction.py`——把标注表 chunk 覆盖页的**完整 pymupdf blocks（无区域裁剪）**喂给 `reconstruct_cells`，按同一单元格三元组规则打分。

## 评测结果

### 文本基线（生产路径，未改动）

`table-eval-v1` 146/149 + `table-eval-concentration-v1` 8/8 = **154/157（98.1%）**。

### 坐标路径 P0（整页输入，无裁剪）：92/157（Recall 58.6%）

| 表型 | gold | hit | Recall | 观察 |
|---|---:|---:|---:|---|
| concentration | 8 | 8 | 1.00 | 句子式，文本回退即全对 |
| note_cost | 24 | 24 | 1.00 | Recall 满但 Precision 仅 0.27–0.32（整页 prose 行被当成数据行） |
| annual_data | 39 | 33 | 0.85 | 表内行基本命中，混入页面上其他内容行 |
| quarterly | 32 | 24 | 0.75 | 扣非行（标签后置）仍未配对成功 |
| segment | 54 | 3 | 0.06 | 最差：整页多子表 + 说明文字，行带/标签过滤失效 |

## 结论（诚实口径）

1. **坐标路径目前不可替代文本路径**：整页输入下精度/召回都低于文本基线；`extract_cells` 仍是生产默认。
2. 主要瓶颈是**表格区域定位缺失**：把整页 blocks 全量送入导致 prose 行被误当数据行（note_cost/annual 的 Precision 崩、segment 几乎全灭）。
3. 合成夹具全过不代表真实数据可用：真实 PDF 的 span 形态（整行单 span、标签与数值混合、页面上并存多个表格/说明）比夹具复杂得多。
4. 外部模型 REPORT 声称的 150/157 是其**内置文本回退**在独立环境的自评；接入仓库后文本回退走我们的 `extract_cells`（154/157），其数字未复现、也不作为坐标路径成绩。

## 下一步（按优先级）

1. **表格区域定位**：用表头/单位行/章节标记确定表格 y 范围，只送表格块（预期 note_cost/annual/quarterly 精度大幅回升）；
2. **segment 子表隔离**：按"主营业务分X情况"边界切块，丢弃说明文字；
3. **quarterly 扣非行**：检查真实 PDF 页 6/8 的 band 吸收阈值与标签过滤，修复"标签在数值后"配对；
4. 全部通过后再考虑接入 `answer_generation` 与生产链路。

> 已实测：伊利 segment"其他地区"在 PDF 文字层无"地区" span，坐标路径无法修复，维持 OCR/标注分歧结论。
