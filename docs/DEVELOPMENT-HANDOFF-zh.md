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

仓库提交评测 gold、报告、外部哈希锁和覆盖所有 gold/hard-negative 的 38 个最小源证据块；完整 PDF、解析结果、索引和模型缓存仍在 `.gitignore`。因此干净 clone 能先验证题库真实性，正式 Retrieved 评测仍需按下述步骤重建绑定索引。

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

历史 `10fb50419145d56720c9` 的原 snapshot 已不可恢复，不能再声称“同 PDF/chunking 必然重建为该 ID”。当前可复现迁移候选为 `9898c95e13d01c51c156`，运行前必须校验 `data/evaluation/benchmark-v2-e5-migration-v1.json`，不得手改 ID：

```bash
uv run python scripts/validate_benchmark_migration.py \
  --manifest data/evaluation/benchmark-v2-e5-migration-v1.json \
  --target-index-root tmp/benchmark-migration-e5-v1
```

canonical benchmark 门禁仍需单独运行；它验证历史 `10fb...` 身份，不会因为迁移存在而自动通过。即使尚无本地 canonical 索引，也可用外部锁中的 corpus 身份与提交的最小证据目录完成数据集资产门禁；Retrieved runner 则必须看到真实且 ID 一致的 canonical index，或显式提供已验证 migration，否则一票否决。

迁移校验器输出 `VALID` 才说明 source benchmark、judged evidence 与目标索引制品完整对齐。若 index ID、snapshot、模型或任一 artifact digest 不同，必须新建 migration 版本，**不能静默继续**。

## 3. DeepSeek key（不进仓库）

```bash
# 仓库根目录 local-keys.env（已被 .gitignore 忽略）
DEEPSEEK_API_KEY="你的key"

# 评测前加载到当前终端
set -a
source local-keys.env
set +a
```

不要读取、打印或提交该文件；评测脚本只从环境变量读取。API key 必须与 endpoint host 对应，否则 provider credential gate 会拒绝启动。

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
- B 阶段第一步（历史）：四类表型单元格抽取器（quarterly / note_cost / segment / annual_data）全部实现，table-eval **28/149 → 146/149（98.0%）**，8 个单测；当时唯一残差为 PDF 文字层丢"地区"（伊利 segment），后续坐标路径已在 157-cell 尺子上完成双轨验证但也不凭空补字。
- B 阶段第二步：`extract_cells` 已接入 answer_generation 确定性表格路径（季度/附注成本/分部毛利率/年度营收/跨公司/合并-母公司，带引用）；no-LLM 三轨 strict **0.3143→0.6571 / 0.0857→0.2571 / 0.2273→0.3636**，零回归；A/B 开关 `FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES=1`。
- DeepSeek 三轨重跑完成（2026-08-11，`deepseek-chat-table-v2` + RAGAS）：Oracle 0.9714（持平）、Retrieved **0.5714**、Robustness **0.6364**；RAGAS faithfulness 全轨提升；修复 `api_model` 元数据（新产物 `independent_judge=false` 正确标注自评）。
- 远程拒答检测完成（2026-08-11，`deepseek-chat-abstain-v2`）：伪拒答不再刷分，应拒答被如实计分；真实基线 Retrieved strict **0.8000** / 行为 0.8958，Robustness strict **0.8636** / 行为 0.8276，Oracle 0.9429（单题波动）。
- 远程确定性表格优先完成（2026-08-11，`deepseek-chat-table-remote-v1`，受控开关）：Oracle strict 1.0、Retrieved strict **0.8286** / 行为 **0.9583**、Robustness 行为 0.8621，零回归；新 run 已带 `code_revision`。
- concentration 表型完成（2026-08-11，`deepseek-chat-concentration-v2`）：第五类表型 8/8 单元格；Robustness strict **0.9545** / 行为 0.9655；期间受控对比抓到"负例前置取错公司"bug 并修复。
- LLM 改写 retrieved lane 实验为**阴性**（2026-08-11，`rewrite-llm-v1`）：strict 持平、3 个证据回归，不落地；retrieved lane 默认保持 deterministic 词表。改写缓存持久化基础设施已就绪，生产 `/v1/query` 接入需 paraphrase 门控（待做）。
- 坐标级表格重建 P0（2026-08-11，外部模型交付集成）：合成夹具 11/11 过，真实整页输入 92/157（< 文本基线 154/157），**暂不接入生产**；回归尺子 `scripts/evaluate_coordinate_reconstruction.py` 已就位。
- 坐标重建迭代完成（2026-08-11，v1→v9）：**92/157 → 154/157（R=0.981，追平文本基线）**；实现区域定位、拆开标题边界、季度标签修复、散文过滤、segment 列结构、跨页隔离；唯一残差"其他地区"。
- 生产 `/v1/query` 查询归一化完成（2026-08-11）：相对时间解析、别名/代码路由、确定性同义词（默认）/LLM 改写（`FINDOC_RAG_QUERY_REWRITE=llm` + 缓存）、确定性表格优先；真实索引冒烟通过。
- A 阶段软收尾完成（2026-08-12）：**改写质量门控**（LLM 劣化自动回退 deterministic，`FINDOC_RAG_QUERY_GATE`）+ **路由错误率指标**（query-routing-v1，18/18 精确匹配）；过滤信号改为时间解析后推断。

### 已完成（2026-08-13，本工作区待提交）

- PDF IR v2 持久化 line/span bbox、字体、字号、粗体、旋转与坐标空间；两份年报重解析 chunk ID 全部不变；PDF 与 IR 重放均 154/157。
- 坐标安全选择：表头/单位/完整行束/原文一致性门禁，raw 165 pred / 154 hit → safe 157 pred / 154 hit（P/R 0.981）。
- benchmark 外部 SHA 锁 + 38 个最小证据块；clean clone 可验证 48 题 / 35 gold / 53 hard negative；正式 Retrieved/fusion/variant runner 均拒绝错 index ID。
- Agent API v1：结构化 outcome/route/filter/trace/index，动态 capabilities，index-bound evidence resolve + SHA-256，claim-citation 映射。
- 答案层不再用数值大小猜合并/母公司；索引按章节边界传播 statement scope；季度合计真实比较；年度表头年份动态解析。
- 五类表型已落为可选的 index-bound structured-table sidecar：独立 schema/generator/index/source/content/chunk SHA 与计数校验；runtime 命中后注入，在线答案优先读取 cells、缺失时回退文本；不改变 chunk serialization/index identity，旧索引兼容。
- 两份真实年报迁移演练：958 chunks、15 表、195 cells；IR v2 中 12 表采用坐标结果，3 表触发安全回退。冻结 157-cell 仍为 154 hit / 157 pred（P=R=0.9809），PDF 与 IR 完全一致。
- IR v2 重摄取时发现并修复 metadata 继承缺口：调用方未重复传 metadata 时继承当前 active version，避免公司/年份过滤字段在处理升级中静默丢失。
- `/v1/uploads` 已从内存占位升级为默认关闭、显式启动、文件持久化的 ingestion job；strict PDF 质量门禁后进入 registry 和 immutable corpus generation，ready 回写 version/index ID，重复启动与重启中断 fail-closed。
- benchmark migration v1：新 E5 索引 `9898...` 绑定精确模型 revision、模型/embedding/BM25/sidecar/snapshot SHA；38/38 judged chunks 语义核心一致；111 条 lexical paired Hit/MRR 与历史 `10fb...` 完全相同，0 fixed / 0 regressed。
- 生产路由已复用于 retrieval 评测，修复预测年份误作报告年份和简称识别分叉；迁移内 deterministic rewrite 为 12 fixed / 0 Hit@5 回归；dev 融合扫描继续证明 lexical-only 最优。
- no-remote Retrieved 在迁移索引上完成：48 items、strict 0.6000、行为 0.5000、error 0；无远程 key，未生成新 DeepSeek/RAGAS 分数。

### 已完成（2026-08-14，最终远程收口）

- 用户已明确授权将 benchmark 问题和公开财报片段发送给 DeepSeek API；密钥仍只经忽略的 `local-keys.env` 注入，未读取、展示或提交。
- 正式三轨目录：`reports/generation/runs-e5-migration-remote-final`。Oracle / Retrieved / Robustness 共用 migration `benchmark-v2-to-e5-c3f157-v1`、目标 index `9898c95e13d01c51c156` 和 code fingerprint `5f02074f...aff06`。
- strict / 行为三轨均为 1.0000，strict 分母 35 / 35 / 22，p95 1.63s / 2.08s / 2.23s，error rate 0；Retrieved 37/37 gold context 完整。
- 最终 RAGAS 位于 `reports/generation/ragas-index-bound-final-{oracle,retrieved,robustness}.json`；每项 coverage 与 complete-row coverage 均 100%，behavior mismatch 为空。`independent_judge=false`，必须标注 DeepSeek 自评。
- 历史→最终配对：Retrieved 以 table-remote-v1 为历史最佳，6 strict fixed / 0 regressed；Robustness 以 concentration-v2 为历史最佳，见 `paired-historical-best-to-index-bound-final-robustness`，1 / 0；Oracle 持平。旧的 Robustness table-remote 3 / 0 报告保留阶段审计，但不再称为历史最佳。跨度包含多项工程变化和 DeepSeek 随机性，不是严格单变量实验。
- `runs-e5-migration-remote-v1` 中 Oracle 43.75% error 是沙箱网络权限失败的无效 run，只保留审计，不纳入结论。

### 下一步（方案已定，待执行）

1. 用不同 provider 的 key 跑独立 judge；不要用同一 DeepSeek 自评替代独立性。
2. C 阶段：扩充近失拒答、多轮澄清、实时时间/语料新鲜度。
3. 公信力：96 变体人工审核、第二人 gold 盲审、多公司多年度 document-blind、题目层/文档层置信区间。
4. PDF 泛化：增加真实扫描件 OCR 与视觉复核；当前 3 个“其他地区”残差保留，不做 benchmark 特判。

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
