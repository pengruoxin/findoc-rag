# PDF 证据增强日志

## 1. 这轮真正解决的问题

上一轮留下 9 个 `text_only` 单元格，原计划只是“再想办法补 bbox”。逐页渲染并人工复核后发现，
问题不全是缺坐标：

- 宁德时代 3 个 chunk 被误判成“营业收入和营业成本”表。投资收益、公允价值变动收益和
  跨页非经常性损益中的数字，被文本 fallback 拼成了 3 张不存在的四列表，共 12 个伪 cell；
- 中国平安的设计型财务摘要把一个数值拆成多个 PDF span，并同时存在“调整前/调整后”层级表头。
  旧逻辑不仅没有 bbox，还把 `2022年调整后` 的值标成 `2021年`。

因此本轮没有强行把 9 个残差变成 9 个坐标，而是先修语义，再生成区域证据。

## 2. 实现

### 2.1 表型入口收紧

`note_cost` 不再只看“本期发生额 / 上期发生额 / 收入 / 成本”这些高频词，必须同时出现
监管表标题“营业收入和营业成本”。这一步删除 3 张人工确认的假表、12 个伪 cell。

### 2.2 设计型表格恢复

- 数字 span `360` + `,` + `403` 只在同一视觉基线、紧邻且满足数字格式时合并；
- 年度表使用中心线聚类，避免相邻密排行的字框轻微重叠后把整张表并成一行；
- `2022` + `年` 可恢复为年度表头；
- 只有显式出现“调整前”的年度才允许拥有“调整前 / 调整后”两个子列，避免把同比列的
  “调整后”误绑到最老年度；
- 输出列名保留为 `2022年调整前` / `2022年调整后`，不再静默选一个比较口径。

这一逻辑同时恢复中国平安 2023、2024 两张财务摘要：各 8 行 × 4 个年度/重述列，共 64 格，
全部带 PDF value bbox。

### 2.3 有界 Region Proof

`PdfRegionInspector.render_table_cell_region` 只接受已经通过 cell proof 的坐标单元格，并执行：

1. source manifest 路径约束与整份 PDF SHA-256 校验；
2. cell page/bbox 边界校验；
3. bbox 必须与 PDF 原生文字中的同值 token 相交；
4. 只渲染包含表头/行名上下文的单页小区域，区域面积最多为页面的 20%；
5. PNG、源 PDF、cell proof、页码、bbox 和裁剪框再次绑定 SHA-256。

人工审核包可选接收 source manifest；只为实际命中的 requirement/cell 生成 region proof。
失败原因进入不可变审核包，不会退化成整页截图或任意文件读取。

## 3. 可复现评测

命令：

```powershell
uv run python scripts/evaluate_pdf_evidence_enhancement.py
```

输入只包含 `calibration + dev` 8 份年报。人工 hard-case fixture 绑定 3 张假表、16 个复杂年度格
和 4 个区域证明样本；未打开 frozen test。

| 指标 | 上一轮 P4-G | PDF 证据增强 |
|---|---:|---:|
| 表格数 | 11 | 10 |
| 单元格数 | 111 | 176 |
| 精确 value bbox | 102/111（91.89%） | **176/176（100%）** |
| text-only | 9 | **0** |
| 人工确认的假表 | 3 | **0** |
| 人工确认的伪 cell | 12 | **0** |
| 未涉及的历史 cell 保持一致 | — | **96/96** |
| 复杂年度 hard cells | — | **16/16 exact + coordinate** |
| 有界 region proof | 0 | **4/4** |
| 最大裁剪面积占整页 | — | **14.14%** |
| 模型请求 / token | 0 / 0 | **0 / 0** |

这里 `111 → 176` 不是简单“多抽了 65 格”：中间先删了 12 个伪格，再恢复两张 32 格年度表，
并补出一张此前因门禁未接入的 16 格季度表。评测将“旧格不回归、假格删除、新 hard cell 精确”
分开检查，避免用数量增长冒充准确率。

原始报告：`reports/agent/pdf-evidence-enhancement-v1.json`。人工 fixture：
`data/evaluation/pdf-evidence-hard-cases-v1.json`。

## 4. 生产状态与边界

- calibration 活跃索引已重建：32 cells，generator `coordinate-safe-v4-pdf-validated`；
- development 活跃索引已重建：176 cells，全部有 geometry/proof；
- frozen-test 索引没有重建或打开；
- region proof 不进入 DeepSeek prompt，默认不产生视觉 token；只在人工审核显式配置 source
  manifest 时按需生成。

这轮证明的是 PDF 表格语义与证据可验证性，不是 DeepSeek 问答分数；没有把历史 45/48
重新归因给本轮。真实扫描件、复杂合并单元格和跨页续表仍需独立盲测。
