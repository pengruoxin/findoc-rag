# FinDocRAG

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

面向**中文上市公司年报**的可追溯 RAG 基础设施：把长 PDF 转成保留坐标的证据，建立版本化混合索引，让检索的每一步都可检查、可评测、可归因，并用真实模型基线回答"瓶颈到底在哪一层"。

## 目录

- [为什么不是普通 RAG](#为什么不是普通-rag)
- [功能](#功能)
- [当前真实基线](#当前真实基线)
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
- **单元格级表格尺子**：8 张真实表格、149 个单元格三元组，让表格重建有精确分数
- **双语评测体系**：48 题 + 96 个专业表达变体 + 53 个真实干扰 + 词表外改写（OOV）验证

---

## 当前真实基线

### 检索（2026-08-07，公司 + 年份过滤，Hit@5）

| 问法 | 纯关键词 | 语义检索 | 同义词改写后 |
|---|---:|---:|---:|
| 原题（照年报原文问） | 0.838 | 0.216 | **0.892** |
| 代码 / 简称（600519 2024 年营收） | 0.811 | 0.676 | **0.892** |
| 口语 / 相对时间（去年营收、毛利水平） | 0.730 | 0.162 | **0.919** |

结论：当前语义分支在三种问法上都是负资产，**默认策略 = 纯关键词检索**；同义词改写（7 组映射，全部来自失败案例）零回归地救回口语问法。

### 生成（48 题，DeepSeek 三轨）

| 赛道 | 全对率（strict） | 行为准确率 | 平均上下文 | p95 延迟 |
|---|---:|---:|---:|---:|
| 直接给答案（生成上限） | **0.9714** | 1.0000 | 303 token | 1.7s |
| 真实检索（端到端） | **0.5429** | 0.8333 | 1536 token | 1.8s |
| 抗干扰（gold + 真实干扰） | **0.5455** | 0.7931 | 784 token | 2.4s |

核心结论：**只要证据给对，模型能答对 97%；走真实检索只剩 54%**——瓶颈在检索 / 路由 / 证据选择，不在生成。RAGAS 已随三轨输出（同模型自评，仅作诊断）。

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

- **数据门禁**：`scripts/validate_benchmark_dataset.py`（gold 存在性 / quote 匹配 / 变体一致性，fail closed）
- **检索评测**：`run_retrieval_variant_eval.py`（3 路 × 3 问法 × 2 过滤态）、`run_retrieval_fusion_sweep.py`（权重扫描）、`run_oov_eval.py`（词表外改写）
- **表格评测**：`evaluate_table_extraction.py`（单元格三元组 Precision / Recall）
- **生成评测**：`run_generation_eval.py` 三轨（oracle / retrieved / robustness）+ `run_ragas_generation_eval.py`
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
- [变更记录](docs/history/optimization-log-zh.md)

---

## 当前局限与下一步

局限（对外主张时必须声明）：

- 只有 2 家公司、1 个年度，`independent_gold=false`，未做第二人复核
- 表格仍以线性文本为主，单元格级尺子已就位但抽取器只实现了季度表基线
- 语义检索（dense）当前是负资产，等待"表格结构化 + 更强中文模型"后重新验证
- 术语处理方案已定（LLM 改写 → 指标知识库 → dense 终局），OOV 验证进行中

下一步：

1. **阶段 0：dense 对照实验**——用中文专用向量模型跑 OOV，回答"更强的 dense 能不能解决同义词、是否与 LLM 改写重复"
2. 查询侧 MVP：OOV 集跑 `--rewrite llm` 对比
3. 端到端 strict 的当前瓶颈：行为拒答（8 题）+ 表格抽取（5 题）

---

## License

本项目采用 [MIT License](./LICENSE)。
