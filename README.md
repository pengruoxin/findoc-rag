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
- [当前真实基线（完整数字）](#当前真实基线完整数字)
- [架构](#架构)
- [模型交代](#模型交代)
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
| **表格** | 五类表型 **157 个单元格**尺子；坐标级重建 **92/157 → 154/157**，安全选择后预测 165 → 157、Precision/Recall 均 **98.1%** |
| **生成** | 同一迁移索引、同一代码指纹的 DeepSeek 最终三轨 strict / 行为均 **1.00 / 1.00 / 1.00**（Oracle / Retrieved / Robustness），远程错误率 0 |
| **成本** | 每问平均上下文 **1.5k token**，相对整份语料压缩 **99.4%（约 168 倍）**；Oracle 只需 **303 token**；p95 延迟约 2s |
| **工程** | migration + index + artifact SHA + dirty-worktree `code_fingerprint` 四重绑定；**测量修正与能力提升分开记账** |

一句话：**历史基线证明瓶颈在“找到并读懂证据”；把 PDF IR、索引绑定表格 sidecar、结构路由和财务勾稽接成闭环后，最终三轨在同一代码指纹下全部达到 strict / 行为 1.00。**

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

1. **坐标级表格重建（从 58.6% 到 98.1%）**——真实中文年报的表格同时踩中阅读顺序反转、跨行标签、跨页表格、文本层丢字四类问题；用"区域定位 → 行带聚类 → 列对齐 → 标签修复 → 跨页隔离"的几何管线把整页输入召回从 92/157 提到 154/157，再用表头、完整行束、单位与原文一致性门禁把预测从 165 降到 157；PDF 与持久化 IR v2 结果完全一致。唯一残差是文字层丢失"地区"，没有硬编码掩盖。
2. **五类表型确定性答案**——季度 / 附注 / 分部 / 年度 / 集中度 157 个单元格三元组尺子；远程模式下表格题优先走确定性抽取（受控开关），Oracle strict 0.94 → 1.00、Retrieved 行为 0.90 → 0.96、Robustness strict 0.86 → 0.96，零回归。
3. **查询归一化 + 改写质量门控**——相对时间、公司别名/股票代码路由（18/18 精确匹配）；LLM 改写带持久化缓存，检索劣化自动回退 deterministic；并留下一个反直觉的负向结论：LLM 改写进 canonical 检索链路会伤 3 题，因此只用于生产自由文本。
4. **检索策略由数据决定**——fusion sweep 证明纯关键词优于任何融合权重；bge-small-zh-v1.5 对照证明问题不在"中文模型能力"；OOV 评测证明 LLM 改写是词表外问法的杠杆。三个结论互相独立、都可复现。
5. **受控实验文化**——每个 run 记录 `code_revision`；跨版本对比必须声明单变量；配对报告输出 fixed/regressed；把"打分口径修正"（拒答检测）与"模型能力提升"分开记账，避免伪提升进简历。
6. **版本化索引与全链路可观测**——不可变 corpus generation + 原子切换；trace 覆盖检索每一阶段；PDF 处理有可复现审计（文本层/几何/字体/跨页）。
7. **Agent-ready 证据契约**——`/v1/query` 返回稳定 outcome、路由、过滤与 claim→citation；`/v1/capabilities` 动态声明真实能力；`/v1/evidence:resolve` 按 index ID 解析完整证据并校验 SHA-256。五类表型已落为 index-bound sidecar：不改 benchmark chunk/index identity，启动时逐层验摘要，命中后才注入在线回答。
8. **显式授权的摄取状态机**——上传默认不改语料；Agent 明确启动后才经历 `validating → ingesting → indexing → ready/failed`，任务跨重启持久化并回写 document version / index ID，重复启动、伪 PDF、OCR 未解决均 fail-closed。

---

## 目前的提升

| 领域 | 改动 | 结果 |
|---|---|---|
| 检索 | 同义词改写（7 组，来自失败案例） | 口语问法 Hit@5 0.73 → **0.92**，零回归 |
| 检索 | LLM 改写 + 持久化缓存 + 质量门控 | OOV Hit@5 0.194 → **0.694**；劣化自动回退 |
| 表格 | 五类表型抽取器 + index-bound sidecar | 28/149 → **146/149（98.0%）**，集中度 8/8；真实两份年报自动发现 15 表 / 195 cells，12 表坐标路径、3 表安全回退文本 |
| 表格 | 坐标级重建 + chunk-grounded 安全选择 | 92/157 → **154/157**；raw P 0.933 → **safe P 0.981**，Recall 不降 |
| 评测 | 外部 SHA 锁 + 38 个最小源证据块 | 干净 clone 可验证 48 题 / 35 gold / 53 hard negative；错索引一票否决 |
| Agent | query / capabilities / evidence resolve | 索引绑定、证据哈希、结构化 outcome 与 claim-citation 可机读 |
| 生成 | index-bound 结构证据路由 + 自适应候选池 + 确定性财务勾稽 | 最终 Oracle / Retrieved / Robustness strict 与行为均 **1.000**；Retrieved gold context **37/37**，远程错误率 0 |
| 评测 | 拒答检测（打分口径修正） | 应拒答被如实计分、伪拒答不再刷分（⚠️ 这是测量修正，不是能力提升） |
| 生产 | `/v1/query` 查询归一化 | 相对时间 / 别名 / 代码路由 + 改写门控，路由 **18/18** |

---

## 当前真实基线（完整数字）

> 当前主结果为 2026-08-14 的 `deepseek-index-bound-final`。三轨共享 migration `benchmark-v2-to-e5-c3f157-v1`、目标索引 `9898c95e13d01c51c156` 与代码指纹 `5f02074f...aff06`。历史受控实验仍保留用于解释增量；历史→最终包含多项工程变化和 DeepSeek 随机性，**不是严格单变量实验**。

### 检索：按问法（Hit@5，公司 + 年份过滤）

| 问法 | 纯关键词 | 语义检索 | 同义词改写后 |
|---|---:|---:|---:|
| 原题（照年报原文问） | 0.838 | 0.216 | **0.892** |
| 代码 / 简称（600519 2024 年营收） | 0.811 | 0.676 | **0.892** |
| 口语 / 相对时间（去年营收、毛利水平） | 0.730 | 0.162 | **0.919** |

### 检索：排序四指标明细（canonical，纯关键词，query_parser 过滤，top5 / candidate20）

| Recall@5 | Precision@5 | MRR@5 | NDCG@5 | 候选池召回 | 前 5 干扰数 |
|---:|---:|---:|---:|---:|---:|
| 0.8108 | 0.2216（部分判定下界） | 0.6937 | 0.7089 | 0.8919 | 0.1081 |

### 生成：DeepSeek 三轨（48 题，含 RAGAS）

| 赛道 | strict | 行为准确率 | 平均上下文 | p95 延迟 | Faithfulness | Answer Relevancy | Context Relevancy | Context Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle（生成上限） | **1.0000**（35） | **1.0000** | 303 token | 1.63s | 0.7612 | 0.9077 | 1.0000 | 0.9865 |
| Retrieved（端到端） | **1.0000**（35） | **1.0000** | 1533 token | 2.08s | 0.8843 | 0.9066 | 1.0000 | **1.0000** |
| Robustness（抗干扰） | **1.0000**（22） | **1.0000** | 784 token | 2.23s | 0.7963 | 0.8938 | 0.9444 | **1.0000** |

strict 括号内是符合确定性数值评分条件的题数；叙述类题不进入 strict 分母，因此不能表述为“48 题全部数值 strict”。RAGAS 四项均为 DeepSeek 自评（`independent_judge=false`），每项 coverage 与完整行 coverage 均为 100%，只作语义诊断。Retrieved 的确定性 gold-context 检查为 37/37。

### 效率：证据预算三路对比（`avg_evidence_tokens_top5`）

| 检索方式 | 证据 token | 结论 |
|---|---:|---|
| 纯关键词 | **307** | 默认路径 |
| 语义检索 | 386 | 更贵 |
| 混合 2:1 | 348 | 更贵且更差 |

语义分支不仅命中率低，送进下游的证据量还更大——用更贵的上下文换更差的结果，这是"纯关键词默认"的另一个证据。

---

## 架构

```text
官方 PDF
  → 保留坐标的 Document IR v2（页 / block / line / span / 字体 / 旋转）+ 质量门禁
  → 结构感知切片（标题层级 / 页眉页脚去除 / section path）
  → 事务化文档注册表（版本不可变）→ 不可变语料索引（原子切换 current.json）
  → 元数据过滤（公司 / 年份 / 文档类型）
  → 纯关键词检索（默认，数据定论）→ 可选 dense / RRF / 重排
  → 可解释的口径路由 + 改写质量门控
  → 五类表型 index-bound sidecar（坐标优先 / 安全回退文本 / 不改变 chunk identity）
  → 证据门禁回答（引用强制 / 不一致拒答 / 拒答检测）
```

服务端点：`/health/live`、`/health/ready`、`/v1/index`、`/v1/search`、`/v1/query`、`/v1/capabilities`、`/v1/evidence:resolve`、`/v1/uploads`、`/v1/uploads/{job_id}:process`、`/v1/traces/{trace_id}`、`/v1/metrics`。

---

## 模型交代

默认离线可运行：检索走 BM25 关键词（`rank-bm25`），表格题走确定性抽取，不需要下载模型，也不需要 API key。

| 环节 | 模型 | 说明 |
|---|---|---|
| 检索（默认） | BM25 关键词 | `default_mode=lexical`，纯本地 |
| Dense 检索（可选） | `intfloat/multilingual-e5-small` | 仅 dense / hybrid 模式启用 |
| 重排（可选） | `BAAI/bge-reranker-v2-m3` | 默认关闭 |
| 生成 / 查询改写 | `deepseek-chat` | 默认关闭，temperature=0 |
| RAGAS 评测 | judge / embedding 可配置 | 仅评测使用 |

远程生成和查询改写需要 `DEEPSEEK_API_KEY`，通过被 `.gitignore` 忽略的 `local-keys.env` 注入。

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
uv run python scripts/validate_benchmark_migration.py \
  --manifest data/evaluation/benchmark-v2-e5-migration-v1.json \
  --target-index-root data/indexes/corpus   # VALID 才能在迁移索引上正式评测
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
set -a && source local-keys.env && set +a   # 文件已被 .gitignore 忽略；不要打印或提交
uv run python scripts/run_generation_eval.py \
  --lane retrieved_context \
  --model deepseek-index-bound-next \
  --index-root data/indexes/corpus \
  --migration-manifest data/evaluation/benchmark-v2-e5-migration-v1.json \
  --require-remote
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

1. **独立裁判与统计公信力**：当前回答和 RAGAS judge 都使用 DeepSeek，下一轮需引入不同 provider 的独立 judge、第二人盲审和题目层/文档层置信区间
2. **OCR / 扫描件**：为真实无文字层、低文本和缺字页补 OCR，并增加独立盲测；当前两份年报不能证明扫描 PDF 能力
3. **行为与新鲜度**：扩充近失拒答、多轮澄清和语料时效策略
4. **覆盖外推**：新增多公司、多年度 document-blind 测试；最终三轨 1.00 只代表当前冻结集，不宣称通用 SOTA

---

## License

本项目采用 [MIT License](./LICENSE)。
