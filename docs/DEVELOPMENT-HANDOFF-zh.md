# 跨设备开发交接（Windows → Mac）

> 目标：在另一台 Mac 上 clone 后能继续本项目的评测、检索与生成实验。本文档以本地文件为准，所有命令跨平台（用 `uv run`，不依赖 PowerShell）。

## 1. 快速开始

```bash
git clone https://github.com/pengruoxin/findoc-rag.git
cd findoc-rag
uv sync --extra dev --extra api --extra dense
uv run pytest -q
```

依赖组说明：

- `dev`：pytest / ruff（必装）
- `api`：FastAPI 服务
- `dense`：sentence-transformers + E5（检索评测必装；首次运行会下载 `intfloat/multilingual-e5-small`）
- `evaluation`：RAGAS / langchain（只在跑语义评测时安装；RAGAS 0.3.1）

## 2. 本地数据重建（Mac 上没有 PDF / 索引 / chunks）

仓库只提交评测 gold、报告和代码；PDF、解析结果、索引、模型缓存都在 `.gitignore` 里。Mac 首次需要重建，顺序如下：

### 2.1 下载两份年报（官方 CNInfo）

```bash
uv run findoc-rag fetch-annual-report --company 贵州茅台 --year 2024
uv run findoc-rag fetch-annual-report --company 伊利股份 --year 2024
```

输出会打印 PDF 与 manifest 路径（在 `data/artifacts/cninfo/` 下）。

### 2.2 摄取（document-key 必须固定，保证版本一致）

```bash
uv run findoc-rag ingest-document <茅台-pdf路径> --document-key cninfo:600519:annual:2024
uv run findoc-rag ingest-document <伊利-pdf路径> --document-key cninfo:600887:annual:2024
```

### 2.3 构建语料索引（含 dense）

```bash
uv run findoc-rag build-corpus-index --dense
```

### 2.4 验证版本一致性（关键）

如果 PDF 与 chunking 版本一致，重建后的 `index_id` 应为 `10fb50419145d56720c9`（与 benchmark-v2 绑定一致）：

```bash
cat data/indexes/corpus/current.json
uv run python scripts/validate_benchmark_dataset.py
```

`validate_benchmark_dataset.py` 输出 `VALID` 才说明评测 gold 与本地 corpus 对齐。若 index_id 不同（chunking 或 PDF 有变化），必须重新锚定 benchmark 或升级版本，**不能静默继续**。

## 3. DeepSeek key（不进仓库）

```bash
export DEEPSEEK_API_KEY="你的key"
```

仅当前终端有效。评测脚本从环境变量读取；`data/raw/*` 已被 gitignore，不要把 key 写进仓库。

## 4. 常用命令

### 数据门禁

```bash
uv run python scripts/validate_benchmark_dataset.py   # benchmark-v2 完整性（fail closed）
```

### 检索评测

```bash
# 变体 regime（canonical / ticker / 相对时间）
uv run python scripts/run_retrieval_variant_eval.py --output-dir reports/ranking/variant-regime-vX

# 融合权重 sweep
uv run python scripts/run_retrieval_fusion_sweep.py --output-dir reports/ranking/fusion-sweep-vX

# 词表外改写（OOV）评测：--rewrite none | deterministic | llm
uv run python scripts/run_oov_eval.py --rewrite llm --output-dir reports/ranking/oov-eval-llm-v1
```

### 表格抽取评测（单元格级尺子）

```bash
uv run python scripts/evaluate_table_extraction.py --output-dir reports/ranking/table-eval-vX
```

### 生成评测三轨（DeepSeek）

```bash
uv run python scripts/run_generation_eval.py --lane oracle_context --model deepseek-chat --require-remote
uv run python scripts/run_generation_eval.py --lane retrieved_context --model deepseek-chat --require-remote
uv run python scripts/run_generation_eval.py --lane robustness --model deepseek-chat --require-remote
```

`--model` 只是 run 标签；API 模型名用 `--api-model`（默认 deepseek-chat）。三轨完成后跑 RAGAS：

```bash
uv run python scripts/run_ragas_generation_eval.py \
  reports/generation/runs/<run>/items.jsonl --output reports/generation/ragas-<lane>.json
```

## 5. 当前进度与下一步

### 已完成（2026-08-07）

- P0：canonical benchmark-v2（48 题 / 96 变体 / 53 干扰）、数据集级 integrity gate、config 快照、as_of_date 注入
- P1：变体 regime 检索评测、融合权重 sweep（结论：lexical-only 默认）、表格抽取评测层（149 cells）、同义词查询改写（7 组映射，semantic 0.73 → 0.92）
- 真实基线：DeepSeek 三轨（Oracle strict 0.9714 / Retrieved 0.5429 / Robustness 0.5455）+ RAGAS
- 文档重组：docs/README.md 为索引；docs/architecture、evaluation、history、interview、ui 分类

### 已完成（2026-08-11）

- 阶段 0 收口：构建 bge-small-zh-v1.5 语料索引 `6a951f4e8b7bd913d918`（不覆盖 E5 索引），跑完 OOV ×（none / deterministic / llm）×（E5 / bge-zh）与变体矩阵；
- 结论：LLM 改写是 OOV 决定性杠杆（0.194 → 0.694），bge-zh dense 全面弱于 E5，默认 lexical-only 不变；
- `_dense_text` 增加 BGE v1.5 查询指令前缀（含单测）；新增 4 个实验目录与 2 份 analysis.md。
- B 阶段第一步：四类表型单元格抽取器（quarterly / note_cost / segment / annual_data）全部实现，table-eval **28/149 → 146/149（98.0%）**，8 个单测；唯一残差为 PDF 文字层丢"地区"（伊利 segment），需坐标级重建。
- B 阶段第二步：`extract_cells` 已接入 answer_generation 确定性表格路径（季度/附注成本/分部毛利率/年度营收/跨公司/合并-母公司，带引用）；no-LLM 三轨 strict **0.3143→0.6571 / 0.0857→0.2571 / 0.2273→0.3636**，零回归；A/B 开关 `FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES=1`。
- DeepSeek 三轨重跑完成（2026-08-11，`deepseek-chat-table-v2` + RAGAS）：Oracle 0.9714（持平）、Retrieved **0.5714**、Robustness **0.6364**；RAGAS faithfulness 全轨提升；修复 `api_model` 元数据（新产物 `independent_judge=false` 正确标注自评）。
- 远程拒答检测完成（2026-08-11，`deepseek-chat-abstain-v2`）：伪拒答不再刷分，应拒答被如实计分；真实基线 Retrieved strict **0.8000** / 行为 0.8958，Robustness strict **0.8636** / 行为 0.8276，Oracle 0.9429（单题波动）。
- 远程确定性表格优先完成（2026-08-11，`deepseek-chat-table-remote-v1`，受控开关）：Oracle strict 1.0、Retrieved strict **0.8286** / 行为 **0.9583**、Robustness 行为 0.8621，零回归；新 run 已带 `code_revision`。

### 下一步（方案已定，待执行）

1. **LLM 改写落地生产**：接入 `/v1/query`；改写缓存持久化到 run 目录（当前跨 run 不稳定，36 条中 8 条不同）；改写后叠加 deterministic 词表兜底；
2. 行为拒答策略：剩余误拒答集中在 concentration 类表格（moutai_concentration / yili_concentration，未被四类抽取器覆盖）与非表格波动；坐标级表格重建修复 PDF 文字层丢字；
3. 96 变体与 OOV 实例人工审核；多轮评测；公信力工程（独立 judge、多公司多年度）。

方案文档：[term-normalization-design-zh.md](./architecture/term-normalization-design-zh.md)

## 6. 文档导航

- 入口：[docs/README.md](./README.md)
- 总览：[roadmap-zh.md](./roadmap-zh.md)（项目水平、瓶颈、路线）
- 数字：[evaluation/baseline-zh.md](./evaluation/baseline-zh.md)
- 规则：[evaluation/benchmark-and-metrics-zh.md](./evaluation/benchmark-and-metrics-zh.md)
- 纪律：[evaluation/experiment-protocol-zh.md](./evaluation/experiment-protocol-zh.md)（控制变量实验协议，所有对比必须遵守）
- 清单：[evaluation/improvement-list-zh.md](./evaluation/improvement-list-zh.md)
- 实验结论：[evaluation/experiment-summaries.md](./evaluation/experiment-summaries.md)
- 变更记录：[history/optimization-log-zh.md](./history/optimization-log-zh.md)

## 7. 注意事项

- Windows 下的 `.venv\Scripts\python.exe` 在 Mac 是 `uv run python` 或 `.venv/bin/python`；
- `scripts/validate-artifacts.ps1` 是 PowerShell 门禁，Mac 可安装 pwsh 运行，或用等价 python 脚本（`validate_benchmark_dataset.py`）；
- 跑 RAGAS 需要 `--extra evaluation`；首次会下载 embedding 模型；
- 报告/run 目录已提交（含 DeepSeek 逐题回答），可作为回归对照，不要覆盖（新实验用新目录）。
