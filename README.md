# FinDocRAG

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

把 **26 万 token 的中文上市公司年报**，变成每次提问只送 **1.5k token**、答案带页码与坐标证据的可追溯 RAG——检索、表格、生成三层都能量化，每个数字背后都是一次受控实验。

## 目录

- [亮点速览](#亮点速览)
- [为什么不是普通 RAG](#为什么不是普通-rag)
- [三层评测体系](#三层评测体系)
- [技术亮点](#技术亮点)
- [目前的提升](#目前的提升)
- [架构](#架构)
- [快速开始](#快速开始)
- [试一下](#试一下)
- [文档导航](#文档导航)
- [局限与下一步](#局限与下一步)
- [License](#license)

---

## 亮点速览

| 层 | 亮点 |
|---|---|
| **检索** | 口语/相对时间问法 Hit@5 **0.73 → 0.92**；词表外口语改写 **0.19 → 0.69**；融合权重与中文 dense 对照均为负收益 → 默认纯关键词是数据定的，不是拍脑袋 |
| **表格** | 五类表型 **157 个单元格**尺子，抽取准确率 **98.1%**；坐标级重建召回 **92/157 → 154/157**，追平文本基线 |
| **生成** | DeepSeek 三轨 strict **1.00 / 0.83 / 0.96**（Oracle / Retrieved / Robustness），行为准确率 **1.00 / 0.96 / 0.97** |
| **成本** | 每问平均上下文 **1.5k token**，相对整份语料压缩 **99.4%（约 168 倍）**；Oracle 只需 **303 token**；p95 延迟约 2s |
| **工程** | 全部受控实验：单变量声明、逐题配对、`code_revision` 追溯；**测量修正与能力提升分开记账** |

一句话：**证据给对时模型能答对 97%——问题从来不在生成，而在"找到并读懂证据"这两层，这两层恰好都能被精确评测。**

---

## 为什么不是普通 RAG

| 普通 RAG Demo | FinDocRAG |
|---|---|
| 关键词相似度决定答案 | 口径路由：季度 / 分部 / 合并 / 母公司 / 附注 / 审计，同一指标不同口径分得清 |
| 把整份文档塞进上下文（这里物理上不可能：26 万 token） | 最小充分证据：每问约 1.5k token，平均压缩 168 倍 |
| 黑盒召回，不知道错在哪 | 阶段级 trace：解析 / 切片 / 检索 / 融合 / 路由 / 重排全程可回放，另有 PDF 审计脚本 |
| 靠感觉调 BM25 / 向量权重 | 冻结评测集 + 权重扫描 + OOV + dense 对照，默认策略由数据决定 |
| 表格是"能召回的线性文本" | 坐标级表格重建 + 五类表型确定性抽取，数字与行列关系可验证 |
| 只报一个准确率 | 三层评测：检索（4 指标 + 归因）/ 生成（RAGAS + 确定性门禁）/ 效率（token / 延迟） |
| 优化没有记录 | 受控实验协议：单变量、配对报告、测量与能力分离 |

---

## 三层评测体系

**检索侧（保证能找到）**：Recall@K / Precision@K / MRR@K / NDCG@K，另配两个归因指标——候选池召回（区分"没捞到"与"排序砸了"）与干扰污染计数。

**生成侧（保证用得好）**：RAGAS 四指标（Faithfulness / Answer Relevancy / Context Relevancy / Context Recall）+ 确定性门禁（事实召回 / 数值 / 单位 / 引用 / 行为：答·拒答·澄清）。

**效率侧（保证划得来）**：

| 送进模型的内容 | token | 相对整份语料 |
|---|---:|---:|
| 整份语料（茅台 + 伊利年报） | ≈ **259,742** | 100%（多数模型放不下） |
| 真实检索上下文（top-5） | ≈ **1,542** | **-99.4%（≈168×）** |
| Oracle 证据（gold） | ≈ **303** | -99.9% |

检索侧最新示例（canonical，纯关键词）：Recall@5 0.81 / Precision@5 0.22（部分判定下界）/ MRR@5 0.69 / NDCG@5 0.71。生成侧 RAGAS 随三轨输出。边界：RAGAS 为 DeepSeek 自评（`independent_judge=false`）；NDCG 当前二元相关；Precision/NDCG 为部分判定；冻结集 48 题 / 2 家公司 / 1 个年度。

---

## 技术亮点

1. **坐标级表格重建（从 58.6% 到 98.1%）**——真实中文年报的表格同时踩中阅读顺序反转、跨行标签、跨页表格、文本层丢字四类问题；用"区域定位 → 行带聚类 → 列对齐 → 标签修复 → 跨页隔离"的几何管线把整页输入召回从 92/157 提到 154/157，并诚实标注唯一不可修复项（PDF 文字层丢失"地区"）。
2. **五类表型确定性答案**——季度 / 附注 / 分部 / 年度 / 集中度 157 个单元格三元组尺子；远程模式下表格题优先走确定性抽取（受控开关），Oracle strict 0.94 → 1.00、Retrieved 行为 0.90 → 0.96、Robustness strict 0.86 → 0.96，零回归。
3. **查询归一化 + 改写质量门控**——相对时间、公司别名/股票代码路由（18/18 精确匹配）；LLM 改写带持久化缓存，检索劣化自动回退 deterministic；并留下一个反直觉的负向结论：LLM 改写进 canonical 检索链路会伤 3 题，因此只用于生产自由文本。
4. **检索策略由数据决定**——fusion sweep 证明纯关键词优于任何融合权重；bge-small-zh-v1.5 对照证明问题不在"中文模型能力"；OOV 评测证明 LLM 改写是词表外问法的杠杆。三个结论互相独立、都可复现。
5. **受控实验文化**——每个 run 记录 `code_revision`；跨版本对比必须声明单变量；配对报告输出 fixed/regressed；把"打分口径修正"（拒答检测）与"模型能力提升"分开记账，避免伪提升进简历。
6. **版本化索引与全链路可观测**——不可变 corpus generation + 原子切换；trace 覆盖检索每一阶段；PDF 处理有可复现审计（文本层/几何/字体/跨页）。

---

## 目前的提升

| 领域 | 改动 | 结果 |
|---|---|---|
| 检索 | 同义词改写（7 组，来自失败案例） | 口语问法 Hit@5 0.73 → **0.92**，零回归 |
| 检索 | LLM 改写 + 持久化缓存 + 质量门控 | OOV Hit@5 0.194 → **0.694**；劣化自动回退 |
| 表格 | 五类表型抽取器 | 28/149 → **146/149（98.0%）**，集中度 8/8 |
| 表格 | 坐标级重建（6 项几何修复） | 92/157 → **154/157（R=0.981）** |
| 生成 | 远程确定性表格优先（受控开关） | Oracle strict **1.000**、Retrieved **0.829**（行为 0.958）、Robustness **0.955**（行为 0.966），零回归 |
| 评测 | 拒答检测（打分口径修正） | 应拒答被如实计分、伪拒答不再刷分（⚠️ 这是测量修正，不是能力提升） |
| 生产 | `/v1/query` 查询归一化 | 相对时间 / 别名 / 代码路由 + 改写门控，路由 **18/18** |

---

## 架构

```text
官方 PDF
  → 保留坐标的 Document IR（页码 / 阅读顺序 / bbox）
  → 结构感知切片（标题层级 / 页眉页脚去除 / section path）
  → 事务化文档注册表（版本不可变）→ 不可变语料索引（原子切换 current.json）
  → 元数据过滤（公司 / 年份 / 文档类型）
  → 纯关键词检索（默认，数据定论）→ 可选 dense / RRF / 重排
  → 可解释的口径路由 + 改写质量门控
  → 坐标级表格重建 / 五类表型确定性抽取
  → 证据门禁回答（引用强制 / 不一致拒答 / 拒答检测）
```

服务端点：`/health/live`、`/health/ready`、`/v1/index`、`/v1/search`、`/v1/query`、`/v1/traces/{trace_id}`、`/v1/metrics`。

---

## 快速开始

```bash
git clone https://github.com/pengruoxin/findoc-rag.git
cd findoc-rag
uv sync --extra dev --extra api --extra dense
uv run findoc-rag doctor
uv run pytest -q
```

官方年报全链路：

```bash
uv run findoc-rag fetch-annual-report --company 贵州茅台 --year 2024
uv run findoc-rag fetch-annual-report --company 伊利股份 --year 2024
uv run findoc-rag ingest-document data/artifacts/cninfo/<茅台pdf>.pdf --document-key cninfo:600519:annual:2024
uv run findoc-rag ingest-document data/artifacts/cninfo/<伊利pdf>.pdf --document-key cninfo:600887:annual:2024
uv run findoc-rag build-corpus-index --dense
uv run python scripts/validate_benchmark_dataset.py   # VALID 才说明 gold 与语料对齐
```

启动服务并问答：

```bash
export FINDOC_RAG_INDEX_DIR="$(pwd)/data/indexes/corpus"
uv run findoc-rag serve
curl -X POST localhost:8000/v1/query -H 'Content-Type: application/json' \
  -d '{"query": "600519 2024 年营收是多少", "top_k": 5}'
```

---

## 试一下

本地评测（不需要 API key）：

```bash
uv run python scripts/run_retrieval_variant_eval.py --output-dir reports/ranking/variant-regime-v3
uv run python scripts/evaluate_table_extraction.py --output-dir reports/ranking/table-eval-v3
uv run python scripts/evaluate_coordinate_reconstruction.py --output-dir reports/ranking/coordinate-smoke-v10
uv run python scripts/evaluate_query_routing.py --output-dir reports/ranking/query-routing-v2
uv run python scripts/audit_pdf_pipeline.py --output-dir reports/processing/pdf-audit-v2
```

DeepSeek 端到端（需要 key，仅当前终端）：

```bash
export DEEPSEEK_API_KEY="your-token"   # 不进仓库
uv run python scripts/run_generation_eval.py --lane retrieved_context --model deepseek-chat --require-remote
uv run python scripts/run_ragas_generation_eval.py reports/generation/runs/<run>/items.jsonl --output reports/generation/ragas-<lane>.json
```

---

## 文档导航

- [总体路线图](docs/roadmap-zh.md) · [当前基线（唯一权威数字）](docs/evaluation/baseline-zh.md) · [评测规则](docs/evaluation/benchmark-and-metrics-zh.md)
- [改进清单 P0–P3](docs/evaluation/improvement-list-zh.md) · [实验结论索引](docs/evaluation/experiment-summaries.md) · [控制变量实验协议](docs/evaluation/experiment-protocol-zh.md)
- [PDF 处理审计（2026-08-12）](reports/processing/pdf-audit-2026-08-12.md) · [跨设备开发交接](docs/DEVELOPMENT-HANDOFF-zh.md)
- [分阶段成果摘要（简历/面试，含对外口径）](docs/interview/phase-summaries-zh.md) · [面试说明](docs/interview/findoc-rag-interview-guide-zh.md) · [变更记录](docs/history/optimization-log-zh.md)
- 完整索引：[docs/README.md](docs/README.md)

---

## 局限与下一步

对外主张必须声明：2 家公司 / 1 个年度、`independent_gold=false`、RAGAS 为自评、Precision/NDCG 为部分判定、OOV 实例未人工审核、伊利"其他地区"存在 PDF 文本层丢字（需 OCR 或标注分歧）。

下一步：

1. **PDF 侧改进（当前重点）**：审计已定位 P0 问题——IR 只有 block 级 bbox、表格行阅读顺序反转、42% block 表格线性化、文本层丢字；按 P0→P2 计划推进 span 级 IR、文本层质量门禁与 OCR 兜底
2. 坐标几何层接入生产生成链路（先全量三轨回归）
3. 行为拒答策略 + 时间对齐实时模式
4. 公信力：多公司多年度 document-blind、第二人独立复核、独立 judge、置信区间

---

## License

本项目采用 [MIT License](./LICENSE)。
