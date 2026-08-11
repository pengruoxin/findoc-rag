# 任务：中文年报表格的坐标级重建（自包含版本）

你是一位资深 RAG / PDF 解析工程师。请在不访问原仓库的情况下，基于本 prompt 内嵌的接口、合成夹具，以及同目录的 `coordinate-table-data.json`，实现一个可直接放进现有代码库的表格重建模块。

## 1. 背景

现有系统把 PDF 表格线性化成文本后用正则抽取，有三个已知缺陷：

1. 阅读顺序破坏行列关系：行标签可能排在数值之后（"茅台季度表扣非行"）；
2. 单元格几何信息丢失：当前 IR 只有 block 级 bbox，没有 line/span 级坐标；
3. 文本层丢字：伊利 segment 表"其他地区"行在文字层只有"其他"（不可由坐标修复，需 OCR 或如实标注）。

你的目标是实现"行带聚类 + 列对齐 + 标签-数值配对"的坐标级重建算法，并让它能泛化到监管表格格式（CSRC），而不是只对样例特判。

## 2. 输入

### 2.1 `coordinate-table-data.json`（同目录）

10 张标注表：`quarterly`（4 指标×4 季度）、`note_cost`（3 行×本期/上期×收入/成本）、`segment`（分行业/产品/地区/销售模式子表）、`annual_data`（年报/年末两段）、`concentration`（句子式）。每条含：

- `text`：线性化 chunk 文本（你现有正则基线的输入）；
- `element_references`：block 级元素（`element_id` / `page_number` / `bbox`），值元素常为整行 block，行标签是独立窄 block；
- `gold_cells`：`{row, column, value}`，value 已归一化（去逗号、"−"→"-"）。

### 2.2 内嵌接口（必须与现有代码一致）

```python
from dataclasses import dataclass
from typing import Literal
import re

TableType = Literal["quarterly", "note_cost", "segment", "annual_data", "concentration"]
WHITESPACE = re.compile(r"\s+")

@dataclass(frozen=True)
class ExtractedCell:
    row: str
    column: str
    value: str
    section: str = ""

def normalize_value(raw: str) -> str:
    return raw.replace(",", "").replace("+", "").replace("−", "-")

def normalize_label(raw: str) -> str:
    return WHITESPACE.sub("", raw)

# 评测匹配规则：gold_key = (normalize_label(row), column, normalize_value(value))
# precision = |预测 ∩ gold| / |预测|；recall = |预测 ∩ gold| / |gold|
```

现有文本抽取器入口（`table_extraction.py`）为 `extract_cells(text, table_type) -> list[ExtractedCell]`，你必须保持它不变；新增坐标入口建议签名：

```python
def reconstruct_cells(
    blocks: list[dict],          # pymupdf get_text("dict") 的 blocks，或你自定义的 Span/Line/Block
    table_type: TableType,
) -> list[ExtractedCell]:
    ...
```

推荐自建几何模型（并在模块内提供从 pymupdf dict 构建它的函数）：

```python
@dataclass(frozen=True)
class Span:
    text: str
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1
    font: str
    size: float
    bold: bool

@dataclass(frozen=True)
class Line:
    spans: list[Span]

@dataclass(frozen=True)
class Block:
    lines: list[Line]
```

## 3. 核心算法要求

1. **行带聚类**：按 y 区间重叠/相邻把元素分组为行带（容忍同一行内数值与标签的 y 中心偏差）；
2. **列对齐**：按 x 起点/中心聚类出列位置，识别"行标签列"与"数值列"；
3. **标签-数值配对**：数值归属其行带内的行标签（无论阅读顺序）；支持换行拆开的标签（如"液体乳及乳制品"+"制造业"拼成一行）；
4. **表头/单位识别**：识别季度列头（第一至第四季度）、年度列头（2024年/2023年/2022年及同比）、本期/上期×收入/成本、分部子表（分行业/分产品/分地区/分销售模式）、单位（元/万元）；
5. **跨页合并接口**：提供 `merge_pages(tables) `签名占位，逻辑可留 TODO 但接口必须存在；
6. **回退**：坐标信息不足时调用文本正则基线；提供 `extract_cells` 的 import 兼容。

## 4. 合成夹具（必须通过）

### 夹具 A：茅台季度表扣非行（标签在数值后）

```python
blocks_a = [
    {"lines": [{"spans": [
        {"text": "归属于上市公司股东的扣除非经常性损益后的净利润",
         "bbox": (40.8, 200.0, 180.0, 209.0), "size": 9.0, "bold": False}]}]},
    {"lines": [{"spans": [
        {"text": "24,051,471,185.69", "bbox": (200.0, 200.0, 300.0, 209.0), "size": 9.0, "bold": False},
        {"text": "17,618,626,634.30", "bbox": (300.0, 200.0, 400.0, 209.0), "size": 9.0, "bold": False},
        {"text": "19,108,543,634.77", "bbox": (400.0, 200.0, 500.0, 209.0), "size": 9.0, "bold": False},
        {"text": "25,462,264,522.66", "bbox": (500.0, 200.0, 600.0, 209.0), "size": 9.0, "bold": False}]}]},
]
# 期望：4 个 ExtractedCell，row="归属于上市公司股东的扣除非经常性损益后的净利润"，
# column 依次为 第一季度..第四季度，value 为上面四个数值（归一化后）
```

### 夹具 B：跨行拆开的行标签

```python
blocks_b = [
    {"lines": [{"spans": [
        {"text": "液体乳及乳制品", "bbox": (40.8, 100.0, 150.0, 109.0), "size": 9.0, "bold": False}]}]},
    {"lines": [{"spans": [
        {"text": "制造业", "bbox": (40.8, 109.0, 90.0, 118.0), "size": 9.0, "bold": False},
        {"text": "113,399,011,137.27", "bbox": (200.0, 105.0, 300.0, 114.0), "size": 9.0, "bold": False},
        {"text": "74,731,457,627.92", "bbox": (300.0, 105.0, 400.0, 114.0), "size": 9.0, "bold": False}]}]},
]
# 期望：row="液体乳及乳制品制造业"，两个数值列为 营业收入/营业成本
```

## 5. 验收标准

1. 两个合成夹具通过；
2. `coordinate-table-data.json` 的文本回退路径：在 `extract_cells` 语义下尽量对齐 gold（你不需要把 146/149 提满，但必须报告每张表当前文本基线能做到多少、哪些只能靠坐标修）；
3. 提供 8–12 个 pytest 用例（fixtures 内嵌在测试文件里），`pytest -q` 全绿；
4. 不硬编码"其他地区"；说明它需要 OCR 或标注为 divergence；
5. 无新第三方依赖（仅标准库；pymupdf 仅作为可选输入来源）。

## 6. 输出（三个文件）

1. `table_reconstruction.py`：几何模型 + pymupdf dict 构建器 + 行带聚类/列对齐/配对 + `reconstruct_cells` + 文本回退；
2. `test_table_reconstruction.py`：8–12 个用例；
3. `REPORT.md`：<500 字，说明算法选择、对 10 张表的文本基线评估、坐标可修复项与不可修复项清单。

## 7. 硬约束

- 与 `ExtractedCell` / `normalize_value` / `normalize_label` / `TableType` 完全一致；
- 不要把"其他地区"改名硬凑 gold；
- 算法按监管表格格式泛化设计，禁止只对给定样例特判。
