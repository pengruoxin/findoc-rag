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

### 坐标路径迭代：92/157 → **154/157（Recall 98.1%）**

| 表型 | gold | hit（v1 → v9） | 最终 Recall | 最终 Precision |
|---|---:|---:|---:|---:|
| quarterly | 32 | 24 → **32** | 1.00 | 0.94 |
| note_cost | 24 | 24 → **24** | 1.00 | 1.00 |
| segment | 54 | 3 → **51** | 0.94 | 0.85（茅台 21/21；伊利 30/33，仅"其他地区"不可修复） |
| annual_data | 39 | 33 → **39** | 1.00 | 1.00 |
| concentration | 8 | 8 → **8** | 1.00 | 1.00 |

迭代修复清单（每步都用 `scripts/evaluate_coordinate_reconstruction.py` 回归）：

1. **表格区域定位**：按章节标题（chunk.section_path 末段或表型 marker）锚定 y 起点；锚点取"最后一个匹配"并排除"报告期末公司前三年…说明"等尾部行；
2. **边界识别**：拆开的标题（"2" + ")营业收入…"）跨 span 识别；修复标题正则把小数"15.71"误判为"15."的问题；
3. **季度标签修复**：扣非行标签尾部（"后的净利润"）落入下一行带时，按规范指标名前缀/后缀拆分回填；
4. **散文标签过滤**：含"主要原因/说明/，/："或年份的行标签直接拒绝；
5. **segment 列分配**：表头位置被"营业收入比/上年增"碎片污染，改按固定行结构取前 3 个数值列；
6. **跨页隔离**（关键）：token 增加 page 字段，区域过滤与行带聚类按页隔离，修复多页表格 y 坐标混用导致的整页噪声。

## 结论（诚实口径）

1. **坐标路径已追平文本基线（154/157）**：真实整页输入 + 区域定位下，召回与文本路径一致；对"标签后置/跨行标签/跨页表格"的几何鲁棒性优于文本正则。
2. 唯一残差为伊利 segment"其他地区"（PDF 文字层无"地区" span，坐标无法修复，需 OCR 或标注分歧）。
3. 外部模型交付的初版（合成夹具全过）真实数据仅 92/157；经 6 项修复后达到 154/157——合成夹具与真实数据之间需要回归尺子持续校验。
4. 坐标路径暂未接入生产生成链路；下一步是复用该几何层替换/校验 `extract_cells` 的文本回退，并接入 `answer_generation`。

## 下一步（按优先级）

1. **表格区域定位**：用表头/单位行/章节标记确定表格 y 范围，只送表格块（预期 note_cost/annual/quarterly 精度大幅回升）；
2. **segment 子表隔离**：按"主营业务分X情况"边界切块，丢弃说明文字；
3. **quarterly 扣非行**：检查真实 PDF 页 6/8 的 band 吸收阈值与标签过滤，修复"标签在数值后"配对；
4. 全部通过后再考虑接入 `answer_generation` 与生产链路。

> 已实测：伊利 segment"其他地区"在 PDF 文字层无"地区" span，坐标路径无法修复，维持 OCR/标注分歧结论。
