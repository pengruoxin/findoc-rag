# FinDocRAG

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

面向**中文上市公司年报**的可追溯 RAG 基础设施：把长 PDF 转成保留坐标的证据，建立版本化混合索引，让检索的每一步都可检查、可评测、可归因，并用真实模型基线回答"瓶颈到底在哪一层"。

## 目录

- [为什么不是普通 RAG](#为什么不是普通-rag)
- [功能](#功能)
- [目前的提升](#目前的提升)
- [架构](#架构)
- [快速开始](#快速开始)
- [试一下](#试一下)
- [评测体系](#评测体系)
- [文档导航](#文档导航)
- [当前局限与下一步](#当前局限与下一步)
- [License](#license)

---

## 为什么不是普通 RAG

| 普通 RAG Demo | FinDocRAG |
|--------------|-----------|
| 关键词相似度决定答案 | 口径路由：季度 / 分部 / 合并 / 母公司 / 附注 / 审计，同一个指标不同口径分得清 |
| 把整份文档塞进上下文 | 检索筛选最小充分证据集，平均只送约 1.5k token |
| 黑盒召回，不知道错在哪 | 阶段级 trace：解析 / 切片 / 检索 / 融合 / 路由 / 重排全程可回放 |
| 靠感觉调 BM25 / 向量权重 | 冻结评测集 + 权重扫描，用数据决定默认策略 |
| 没有评测或只报一个准确率 | 双层评测：检索侧 Hit/MRR/NDCG + 生成侧原子事实 / 行为 / RAGAS，三轨定位瓶颈 |
| 数据没有版本 | benchmark 绑定 corpus + chunk schema，gold 失效自动 fail-closed |

---

## 功能

- **真实复杂文档**：针对中文年报的多级章节、跨页内容、页眉页脚和复杂表格设计，而不是 toy QA
- **结构感知切片**：保留 `section_path`、页码、bbox 和源元素映射，切片质量可报告
- **版本化索引**：不可变 corpus generation + 原子切换，重导入同内容是无操作
- **混合检索可评测**：BM25 / 多语 E5 / RRF 在同一冻结集上对比，配 `avg_context_tokens` 等成本指标
- **口径路由与元数据过滤**：公司 / 年份 / 文档类型 + 可解释的年度 / 季度 / 分部 / 附注路由
- **证据门禁回答**：DeepSeek 只拿检索证据回答，强制引用，证据与问题不一致时拒答
- **五类表型抽取 + 坐标级重建**：季度 / 附注 / 分部 / 年度 / 集中度 157 个单元格三元组尺子，坐标重建召回追平文本基线
- **生产问答归一化**：相对时间、公司别名 / 股票代码路由、确定性 / LLM 改写（带缓存与质量门控）
- **双语评测体系**：48 题 + 96 个专业表达变体 + 53 个真实干扰 + 词表外改写（OOV）验证

---

## 目前的提升（截至 2026-08-12，全部为受控实验，含逐题配对）

### 检索侧

| 亮点 | 结果 |
|---|---|
| 口语 / 相对时间问法 Hit@5 | 0.730 → **0.919**（同义词改写，零回归） |
| 词表外口语改写 Hit@5 | 0.194 → **0.694**（LLM 查询改写） |
| 检索策略 | 权重扫描定论：纯关键词全面优于任何 BM25×Dense 融合；中文 dense 对照（bge-small-zh-v1.5）三种问法全部低于 E5，语义路暂不启用 |
| 生产路由 | 别名 / 股票代码 / 相对时间解析 + 改写质量门控，路由评测 **18/18 精确匹配** |
| 负向结论 | LLM 改写进入 canonical 检索链路为阴性（strict 持平、3 个证据回归），只用于生产自由文本 |

### 表格侧

| 亮点 | 结果 |
|---|---|
| 五类表型单元格抽取 | 28/149 → **146/149（98.0%）**，另集中度表 8/8 |
| 坐标级表格重建 | 92/157 → **154/157（R=0.981）**，追平文本基线（修复标签后置 / 跨行标签 / 跨页 / 散文污染） |
| 远程确定性表格优先 | Oracle strict **1.000**、Retrieved **0.829**（行为 0.958）、Robustness **0.955**（行为 0.966），零回归 |

### 生成侧（DeepSeek 三轨，48 题）

| 赛道 | strict | 行为准确率 |
|---|---:|---:|
| Oracle（生成上限） | **1.000** | 1.000 |
| Retrieved（端到端） | **0.829** | 0.958 |
| Robustness（抗干扰） | **0.955** | 0.966 |

说明：早前"Retrieved strict 0.57 → 0.80"一类提升属于**打分口径修正**（远程拒答检测：应拒答被如实计分、带数字的伪拒答不再刷分），不是模型能力提升，对外表述时不混用。RAGAS 四项随三轨输出（DeepSeek 自评，仅作语义诊断）。

---

## 架构

```text
官方 PDF
  → 保留坐标的 Document IR（页码 / 阅读顺序 / bbox / 排版）
  → 结构感知切片（标题层级 / 页眉页脚去除 / section path）
  → 事务化文档注册表（版本不可变）
  → 不可变词法 + 稠密索引（原子切换 current.json）
  → 元数据过滤（公司 / 年份 / 文档类型）
  → BM25 / 多语 E5 / RRF 混合检索
  → 可解释的口径路由
  → 可选 CrossEncoder 重排
  → 证据门禁回答（引用强制 / 不一致拒答）
```

前端工作台、证据审核与实验面板是静态页，由服务挂载在 `/ui`。

---

## 快速开始

### 1. 环境（Python 3.12 / 3.13 + uv，命令跨平台）

```bash
git clone https://github.com/pengruoxin/findoc-rag.git
cd findoc-rag
uv sync --extra dev --extra api --extra dense
uv run findoc-rag doctor
uv run pytest -q
```

### 2. 用任意本地 PDF 跑通最小链路

```bash
uv run findoc-rag parse-pdf path/to/report.pdf
uv run findoc-rag chunk-pdf path/to/report.pdf
uv run findoc-rag build-index data/processed/chunks/<sha256>.jsonl --output-dir data/indexes/my-report --dense
uv run findoc-rag search-index data/indexes/my-report "2024年营业收入是多少" --top-k 5
```

首次 dense 命令会下载 `intfloat/multilingual-e5-small`，之后复用本地缓存。

### 3. 复现官方两份年报的完整链路

```bash
uv run findoc-rag fetch-annual-report --company 贵州茅台 --year 2024
uv run findoc-rag fetch-annual-report --company 伊利股份 --year 2024
uv run findoc-rag ingest-document data/artifacts/cninfo/<茅台pdf>.pdf --document-key cninfo:600519:annual:2024
uv run findoc-rag ingest-document data/artifacts/cninfo/<伊利pdf>.pdf --document-key cninfo:600887:annual:2024
uv run findoc-rag build-corpus-index --dense
uv run python scripts/validate_benchmark_dataset.py   # 输出 VALID 才说明 gold 与语料对齐
```

### 4. 启动服务

```bash
export FINDOC_RAG_INDEX_DIR="$(pwd)/data/indexes/corpus"
uv run findoc-rag serve
```

端点：`/health/live`、`/health/ready`、`/v1/index`、`/v1/search`、`/v1/traces/{trace_id}`、`/v1/metrics`。

---

## 试一下

真实年报检索（需要先完成第 3 步）：

```bash
uv run findoc-rag search-index data/indexes/corpus "贵州茅台2024年分季度营业收入是多少" --top-k 5
uv run findoc-rag search-index data/indexes/corpus "贵州茅台和伊利股份2024年前五名客户销售占比谁更高" --top-k 5
```

本地评测（不需要 API key）：

```bash
uv run python scripts/run_retrieval_variant_eval.py --output-dir reports/ranking/variant-regime-v2
uv run python scripts/evaluate_table_extraction.py --output-dir reports/ranking/table-eval-v2
```

DeepSeek 端到端（需要 key）：

```bash
export DEEPSEEK_API_KEY="your-token"   # 不进仓库
uv run python scripts/run_generation_eval.py --lane retrieved_context --model deepseek-chat --require-remote
```

---

## 评测体系

- **双层 8 指标**：检索侧 Recall@K / Precision@K / MRR@K / NDCG@K（另配候选池召回、干扰污染计数）；生成侧 Faithfulness / Answer Relevancy / Context Relevancy / Context Recall + 确定性事实 / 数值 / 单位 / 引用 / 行为门禁
- **受控实验协议**：每个 run 记录 `code_revision`；跨版本对比必须声明单变量（`--change`）；配对报告输出 fixed / regressed
- **数据门禁**：`scripts/validate_benchmark_dataset.py`（gold 存在性 / quote 匹配 / 变体一致性，fail closed）
- **检索评测**：`run_retrieval_variant_eval.py`（3 路 × 3 问法 × 2 过滤态）、`run_retrieval_fusion_sweep.py`（权重扫描）、`run_oov_eval.py`（词表外改写）
- **表格评测**：`evaluate_table_extraction.py`（单元格三元组 Precision / Recall）+ `evaluate_coordinate_reconstruction.py`（整页输入坐标重建回归）
- **生成评测**：`run_generation_eval.py` 三轨（oracle / retrieved / robustness）+ `run_ragas_generation_eval.py`
- **路由评测**：`evaluate_query_routing.py`（该过滤没过滤 / 错过滤 / 过滤过头）
- 每个实验同时报告：Hit@5 / MRR / strict / `avg_context_tokens` / `p95_latency_ms`

---

## 文档导航

完整索引与阅读顺序：[docs/README.md](docs/README.md)

- [总体路线图（瓶颈、战略问题、阶段）](docs/roadmap-zh.md)
- [当前基线（数字，唯一权威）](docs/evaluation/baseline-zh.md)
- [评测规则（指标定义、门禁、防泄漏）](docs/evaluation/benchmark-and-metrics-zh.md)
- [改进清单（P0–P3 状态）](docs/evaluation/improvement-list-zh.md)
- [实验结论索引](docs/evaluation/experiment-summaries.md)
- [术语处理方案（同义词可扩展性）](docs/architecture/term-normalization-design-zh.md)
- [跨设备开发交接（Windows / Mac）](docs/DEVELOPMENT-HANDOFF-zh.md)
- [面试说明](docs/interview/findoc-rag-interview-guide-zh.md)
- [分阶段成果摘要（简历 / 面试，含对外口径）](docs/interview/phase-summaries-zh.md)
- [控制变量实验协议](docs/evaluation/experiment-protocol-zh.md)
- [变更记录](docs/history/optimization-log-zh.md)

---

## 当前局限与下一步

局限（对外主张时必须声明）：

- 只有 2 家公司、1 个年度，`independent_gold=false`，未做第二人复核
- RAGAS 为 DeepSeek 自评（`independent_judge=false`）；Precision / NDCG 是部分判定下界
- 伊利 segment"其他地区"在 PDF 文字层无"地区" span（坐标无法修复，需 OCR 或标注分歧）
- 坐标几何层已追平文本基线但尚未接入生产生成链路；OOV / 96 变体实例未经人工审核

下一步：

1. **PDF 侧审计与改进（当前重点）**：审计报告见 [pdf-audit-2026-08-12.md](reports/processing/pdf-audit-2026-08-12.md)（可复现脚本 `scripts/audit_pdf_pipeline.py`）——文本层健康，问题集中在结构层：IR 只有 block 级 bbox、表格行阅读顺序反转、42% block 表格线性化、伊利"其他地区"文本层丢字；接下来扩展 span 级 IR、文本层质量门禁与 OCR 兜底
2. 坐标几何层接入生产生成链路（先全量三轨回归）
3. 行为拒答策略（可答题误拒答）+ 时间对齐实时模式
4. 公信力：多公司多年度 document-blind、第二人独立复核、独立 judge、置信区间

---

## License

本项目采用 [MIT License](./LICENSE)。
