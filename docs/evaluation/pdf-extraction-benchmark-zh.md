# PDF 提取基准：native-first 混合解析

## 结论

本项目不采用全量 OCR。生产路径是：可靠文本层走 PyMuPDF 原生提取；扫描页和
混合页按页升级到轻量 OCR；复杂表格或结构冲突预留视觉文档模型后端；最终只把
结构化的小块 Markdown/JSON 交给回答模型。

`deepseek-chat`（以及当前正式文本模型）用于理解结构化证据、计算和回答，不作为
图片解析器。DeepSeek-OCR/OCR-2 是另一套需要独立部署的视觉文档模型，不能把当前
DeepSeek Chat API key 当成图片输入接口。

## 页级路由

每页先生成低成本画像，输出以下稳定 route：

| route | 含义 | 行为 |
|---|---|---|
| `native` | 文本层可接受 | 不调用 OCR |
| `partial_ocr` | 大图上只有少量原生文本 | 保留原生文本，只补充不重叠 OCR 区域 |
| `full_ocr` | 图片页且没有原生文本 | 用 OCR 文本替换文本层，保留页面图像证据 |
| `manual_review` | 空页、可疑 Unicode 或重复文本 | 不静默降级，进入后续视觉/人工复核 |

Document IR 记录 `extraction_source`、OCR 置信度、实际 route、OCR 是否尝试/成功、
原生与 OCR 字符数和错误信息。OCR 配置进入 processing fingerprint，避免不同解析
策略共享同一个版本身份。

## v2 挑战集（沿用 `pdf-extraction-v1` 数据集 ID）

`data/evaluation/pdf-extraction-v1` 从一张真实官方年报页派生三个受控案例：

1. 保留文本层的原生对照页；
2. 将同页渲染为图片的扫描页；
3. 扫描图叠加少量原生文字的混合页。

参考文本来自派生前的原始页，和待测 challenge PDF 分离。v2 又增加 6 个固定资产
表格问题，覆盖章节、重复行标签、期初/期末列和合计列；每个问题都有行、列、值、单位
硬标签。当前标签状态为 `assistant_curated_provisional`，必须在对外宣称人工 gold 前由
第二人复核。它可以证明路由与 OCR 回退是否生效，但仍是合成退化集，不能代替真实
扫描件、多栏、旋转、水印、跨页表格和人工独立标注。

生成并评测：

```powershell
uv run --extra ocr python scripts/build_pdf_extraction_benchmark.py
uv run --extra ocr python scripts/evaluate_pdf_extraction.py
```

报告：`reports/pdf-extraction/pdf-extraction-v1-table-baseline.json`。

## 2026-08-19 首次消融

| lane | 文本相似度 | 数字召回 | route accuracy | 三页总耗时 |
|---|---:|---:|---:|---:|
| native | 0.3311 | 0.3333 | 1.0000 | 201.7 ms |
| hybrid（RapidOCR，180 DPI） | 0.8204 | 0.9948 | 1.0000 | 15,762.2 ms |
| 增量 | +0.4892 | +0.6615 | 0 | +15,560.5 ms |

逐类结果显示：原生页没有触发 OCR；扫描页和混合页的数字召回约 0.99，但文本相似度
只有约 0.73–0.75。这是把复杂页升级到视觉文档模型的候选依据。升级不能只看 OCR
置信度，还要通过表格结构完整性、阅读顺序、关键数字勾稽和下游问答共同判断。

## 表格关联评测

单纯数字召回无法发现“数字在，但挂错了行/列”的问题。评测器现在把 native span 和
OCR region 按页面坐标聚成视觉行，再对每个 gold 依次检查：章节存在、行标签存在、
目标值存在、行和值在同一视觉行、列头存在、值落在对应列的横向区间。只有六项同时
成立才算 `recoverable`。

| lane | 表格事实数 | recoverable | 扫描页 | 混合页 |
|---|---:|---:|---:|---:|
| native | 18 | 0.3333 | 0 | 0 |
| hybrid（RapidOCR，180 DPI） | 18 | **1.0000** | **1.0000** | **1.0000** |

这里的 18 是 6 个事实在原生、扫描、混合三种页面上的重复受控测量，不是 18 个独立
问题。结果证明本页上轻量 OCR 不但找回了数字，也保住了已标注事实的行列关联；它不
代表所有复杂表格都达到 100%。

## 按需 DeepSeek 表格解释 lane

`--deepseek-table` 会把同一页的 6 个问题合并为一次请求，只发送该页经过坐标排序的
文本行，不发送 PDF、图片、gold 值或整份报告。文本不足 300 字的页面直接跳过，避免
无证据调用。返回结果由本地代码按值、单位、章节、行、列做严格匹配；DeepSeek 不参与
评分。报告记录实际 provider、model、endpoint、prompt revision、prompt hash、token、
延迟与逐题输出。

```powershell
$env:DEEPSEEK_API_KEY = "..."  # 只设置在本机进程；不要提交
uv run --extra ocr python scripts/evaluate_pdf_extraction.py `
  --deepseek-table `
  --output reports/pdf-extraction/pdf-extraction-v1-deepseek-table.json
```

当前工作区没有 provider key，因此已生成
`reports/pdf-extraction/pdf-extraction-v1-deepseek-not-run.json`，并明确记录
`status=not_run` / `reason=missing_provider_api_key`，没有伪造远程分数。默认模型仍读取
项目的 `FINDOC_RAG_ANSWER_MODEL`，未配置时为 `deepseek-chat`。

## 真实跨页表格 v1

`data/evaluation/pdf-cross-page-v1` 使用此前未参与 PDF 调参的海尔智家 2024 年报
第 184-186 页。三页是同一张“使用权资产”表：前三列在第 184-185 页，
“办公设备 / 其他 / 合计”又横向续到第 185-186 页；“二、累计折旧”标题位于第 185 页
末，而对应期末合计位于第 186 页。这同时覆盖纵向跨页、横向拆列、重复表头、重复行名
和章节状态跨页继承。

挑战 PDF 有两个三页组：原生页组是官方 PDF 页面的无损复制；栅格页组由同页渲染得到，
只用于受控 OCR 消融，仍明确标记为 `rasterized`，不冒充真实扫描件。7 个问题在两个页组
上形成 14 次行列关联测量，其中 1 题强制要求从前页继承章节。

```powershell
uv run --extra ocr python scripts/build_pdf_cross_page_benchmark.py
uv run --extra ocr python scripts/evaluate_pdf_extraction.py `
  --dataset data/evaluation/pdf-cross-page-v1/benchmark.json `
  --output reports/pdf-extraction/pdf-cross-page-v1-baseline.json
```

| lane | 文本相似度 | 数字召回 | 表格关联恢复 | 原生跨页组 | 栅格跨页组 |
|---|---:|---:|---:|---:|---:|
| native | 0.4863 | 0.5000 | 0.5000 | 1.0000 | 0 |
| hybrid（RapidOCR，180 DPI） | 0.8404 | 1.0000 | **1.0000** | 1.0000 | **1.0000** |

首轮还暴露并修复了一个真实 span 边界问题：PDF 将 `5.` 与 `期末余额` 分成两个 span，
布局序列化的视觉分隔符曾破坏标签匹配。修复只在标签规范化时忽略 span 分隔符，值仍
必须与行同带并落入正确列区间，没有退化成全文搜数字。跨页章节继承题在 native 和
hybrid 的有效页组上均通过。

### 人工复核状态

工作区用户于 2026-08-20 确认 7 条候选标注全部正确，记录已写入
`blind-review-packet.json`。验证器结果为 `complete`、`7/7`，五字段硬一致率为 1.0，
benchmark 已升级为 `human_verified`，且 `independent_gold_ready=true`。

复核方法明确记为 `candidate_verification`：这是第二人对候选标注的人工核验，不是从
空白问题包开始的独立盲标。因此报告同时保留
`blind_reannotation_complete=false`，不能把 1.0 一致率解释为盲标者间一致率。

```powershell
uv run python scripts/validate_pdf_review_packet.py
```

## 真实扫描覆盖审计

对当前 10 份官方年报、2,614 页执行“原生文本少于 250 字且大图覆盖至少 45%”筛查，
只得到 3 页候选。逐页渲染复核后，两页是美的年报封面，一页是中国神华权益结构图，
没有真实扫描财务表。审计报告为
`reports/pdf-extraction/pdf-scan-coverage-audit-v1.json`，结论是
`no_eligible_genuine_scanned_table_in_current_corpus`。因此当前项目能主张“真实跨页表 +
受控栅格 OCR”，不能主张已经通过真实扫描表盲测。

## 后续评测 lane

保持同一个 benchmark 和参考答案，依次增加：

1. `native`：原生文本基线；
2. `hybrid`：页级路由 + 轻量 OCR；
3. `layout`：版面/表格结构恢复（单页与真实三页连续表尺子已完成，仍需扩公司）；
4. `deepseek_table`：文本证据足够时按页批量理解表格（接口与本地硬评分已完成）；
5. `vision_fallback`：仅对 hybrid 仍不合格的页面区域调用视觉文档模型；
6. `rag`：固定检索和回答配置，测最终事实、引用、成本和延迟。

任何视觉后端都必须记录 provider、模型真实名称、版本、页面/区域数量、输入输出 token、
延迟和错误率。不得把运行标签当成真实 API 模型名，也不得用同一模型自评覆盖硬指标。
