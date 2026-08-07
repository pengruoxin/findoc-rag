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

### 下一步（方案已定，待执行）

1. **阶段 0：dense 对照实验**——在现有输入上用中文专用向量模型（bge-small-zh-v1.5）跑 OOV 对比，回答"更强的 dense 能不能解决同义词、是否与 LLM 改写重复"；
2. 查询侧 MVP：OOV 集跑 `--rewrite llm` 对比（36 条已生成）；
3. 行为拒答（8 题）+ 表格抽取（5 题）是端到端 strict 的当前瓶颈。

方案文档：[term-normalization-design-zh.md](./architecture/term-normalization-design-zh.md)

## 6. 文档导航

- 入口：[docs/README.md](./README.md)
- 总览：[roadmap-zh.md](./roadmap-zh.md)（项目水平、瓶颈、路线）
- 数字：[evaluation/baseline-zh.md](./evaluation/baseline-zh.md)
- 规则：[evaluation/benchmark-and-metrics-zh.md](./evaluation/benchmark-and-metrics-zh.md)
- 清单：[evaluation/improvement-list-zh.md](./evaluation/improvement-list-zh.md)
- 实验结论：[evaluation/experiment-summaries.md](./evaluation/experiment-summaries.md)
- 变更记录：[history/optimization-log-zh.md](./history/optimization-log-zh.md)

## 7. 注意事项

- Windows 下的 `.venv\Scripts\python.exe` 在 Mac 是 `uv run python` 或 `.venv/bin/python`；
- `scripts/validate-artifacts.ps1` 是 PowerShell 门禁，Mac 可安装 pwsh 运行，或用等价 python 脚本（`validate_benchmark_dataset.py`）；
- 跑 RAGAS 需要 `--extra evaluation`；首次会下载 embedding 模型；
- 报告/run 目录已提交（含 DeepSeek 逐题回答），可作为回归对照，不要覆盖（新实验用新目录）。
