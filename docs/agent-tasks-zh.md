# FinDocRAG Agent 任务与困难评测 v1

## 先说明边界

现在有两条明确分开的运行路径：

1. `deepseek`：默认路径。真实调用 DeepSeek Chat Completions，由模型在任务允许的范围内
   选择检索、页面窗口、页面区域检查、受控计算或提交工具；没有 provider key 就拒绝运行。
2. `deterministic-baseline`：只用于离线对照。它按固定规则逐目标检索，不是模型 Agent，
   不能再把它的轨迹或分数标成“DeepSeek Agent”。

当前模型 Agent 实现只读 `compare`、`extract` 和两类受控 `calculate`。输入解析和公司/
年份范围由本地门禁确定，DeepSeek 在边界内选择检索词、证据锚点、关系或操作数；本地代码
验证工具参数、文档/页范围、证据归属和引用 ID，并执行精确计算。运行上限为 3 轮、8 次
函数调用，持久化模型名、endpoint、prompt hash、token、延迟和结构化轨迹，不记录自由
形式思维链。

## 命令

真实 DeepSeek 运行：

```powershell
$env:DEEPSEEK_API_KEY = "..."  # 仅当前终端；不要提交或打印
uv run findoc-rag agent run `
  --task compare `
  --index-dir data/indexes/benchmark-v3 `
  "比较海尔智家和长江电力2024年营业收入"

uv run findoc-rag agent run `
  --task extract `
  --index-dir data/indexes/benchmark-v3 `
  --source-manifest data/evaluation/benchmark-v3-source-manifest.json `
  "海尔智家2024年使用权资产表中，累计折旧的期末余额合计是多少？"

# 高风险审计/多事实抽取：额外启用独立证据复核和最多一次修复
uv run findoc-rag agent run `
  --task extract `
  --index-dir data/indexes/agent-hard-v3/calibration `
  --source-manifest data/evaluation/agent-hard-v3-source-manifest.json `
  "工商银行2023年关于预期信用损失的关键审计风险与审计应对是什么？"

uv run findoc-rag agent run `
  --task calculate `
  --index-dir data/indexes/benchmark-v3 `
  "根据中国神华2023年报权益结构图，H股股东、国家能源投资集团有限责任公司、其他A股股东的持股比例分别是多少，合计是否为100%？"

uv run findoc-rag agent inspect <task-id>
uv run findoc-rag agent inspect <task-id> --json

# support proof 无法验证时，任务会自动进入相邻 reviews 队列
uv run findoc-rag agent review list
uv run findoc-rag agent review inspect <review-id>
uv run findoc-rag agent review resolve <review-id> approve --reviewer reviewer-a
uv run findoc-rag agent review resolve <review-id> reject --reviewer reviewer-a

# 修正答案必须显式绑定审核包内证据
uv run findoc-rag agent review resolve <review-id> correct `
  --reviewer reviewer-a `
  --corrected-answer "修正后的答案" `
  --evidence-chunk-id <chunk-id>
```

若 key 已写入被 Git 忽略的 `local-keys.env`，新 PowerShell 会话可安全载入：

```powershell
. .\scripts\import_local_keys.ps1
```

显式离线对照：

```powershell
uv run findoc-rag agent run `
  --runtime deterministic-baseline `
  --task compare `
  --index-dir data/indexes/benchmark-v3/frozen_test `
  "比较海尔智家和长江电力2024年营业收入"
```

默认轨迹写入 `data/agent/tasks/<task-id>.json`。`inspect` 只读已有轨迹。
`--source-manifest` 是 PDF 视觉/版面工具的来源白名单：运行前复核文件 SHA-256，
`extract` 遇到关键审计事项时才按页渲染并分栏 OCR，不会把整份 PDF 发给模型。
P4-C 起，抽取任务的 `--verifier-policy` 默认为 `auto`：审计事项、不少于 4 个原子事实，
或 claim/evidence 通用语言支持异常时才增加一次独立复核。P4-D 又把简单 claim 偏离原子
requirement contract、必要数字缺失纳入复核路由；显式负号/会计括号与正数 contract 冲突
则在模型调用前安全拒答。`off` 可关闭，`always` 可强制全部回答复核。旧
`--evidence-verifier` 仍兼容。不完整答案最多修复一次并再次复核。
P4-E 起，非审计复核还会为 claim/requirement contract 最弱的一项生成逐字 support proof；
proof 无法落回当前 cited evidence 时暂停自动回答并标记 `manual_review`。低语言对齐 proof
最多增加一次窄范围 challenge，结构 partition 错误最多纠正一次。可用
`--no-verifier-support-proof` 做显式消融；关键审计事项保留紧凑完整性复核，避免长 proof
输出挤占预算。
P4-F 起，人工升级会保留被拦截的候选 `AgentTaskResult`，自动生成绑定 task trace SHA-256
的不可变审核包。reviewer 可批准、修正或驳回；修正只能引用包内 chunk，trace 变化后旧审批
失效，每条 review 只能追加一个 resolution，原始模型轨迹不被覆盖。历史人工任务可用
`agent review enqueue <task-id>` 补建队列。工作流契约评测为 10/10、模型请求和 token 均为 0；
这不等于问答准确率或人工审核准确率。
P4-G 又把结构化表格的逻辑单元格和 PDF 数值 bbox 写入 `AgentEvidence`：审核时可直接看到
table、row/column label 与 index、value/unit、page/bbox 和防篡改 proof hash。8 份非冻结
年报的 111 格语义保持 111/111，proof 111/111；102/111 有精确 bbox，剩余 9 格明确显示
text-only。cell proof 只进入本地轨迹/审核包，不注入模型 prompt，额外 token 为 0。
复核前先执行 P4-B 零 token claim
风险闸门：明确的跨公司主体、事实期间或引用范围冲突直接安全拒答；无法对齐的数字/单位
及低语言支持只升级给 Verifier，不由本地规则武断拒答。正常严格分仍为 45/48；自动路由
没有增加正常模型请求，不等于所有普通查询默认多 Agent。

## 本地安全门禁

- 模型只能使用计划中枚举的 target ID；metadata filter 由本地 target 映射产生；
- 工具参数用严格 schema 验证，模型给出的 JSON、chunk ID 和目标归属都不可信；
- `answer` 必须覆盖全部目标，每条 claim 只能引用本次工具返回的 chunk；普通事实严格绑定
  目标，重述/同比按显式同公司证据组共享，跨公司引用还必须同时出现公司名和比较词；
- 零证据缺口达到限定重试后由本地门禁确定性拒答，不能用预测、目标或外部知识补全；
- 引用不合法、目标不完整或证据不足时，把错误回传给模型；预算耗尽仍未通过则安全拒答；
- `answer`、`evidence_only`、`clarify`、`abstain` 分开记录；
- 少于两个可表示的比较目标时，本地输入门禁直接澄清，不消耗模型 token。
- 页面窗口只能读取锚点所在文档的有限相邻页；视觉检查页必须落在检索锚点页范围；
- PDF 页面区域工具只读取来源 manifest 白名单文件并复核 SHA-256；
- 计算操作数必须原样出现在所引 chunk，实际减法和求和由本地 `Decimal` 执行。
- 高风险复核路径在调用模型前检查 claim 的主体、事实期间、数值、单位、会计正负号和
  引用文档范围；主体/期间/引用/明确正负号硬冲突本地拒答，数字、单位、必要数字遗漏或
  requirement contract 偏离只触发模型复核。

## 两套评测

### 1. 轨迹烟雾集

```powershell
# 默认真实 DeepSeek；没有 key 时输出 status=not_run
uv run python scripts/evaluate_agent_compare.py

# 只测离线规则控制器
uv run python scripts/evaluate_agent_compare.py --runtime deterministic-baseline
```

对应报告：

- `reports/agent/agent-compare-v1-deepseek-not-run.json`
- `reports/agent/agent-compare-v1-deterministic-baseline.json`

离线对照的计划、工具和 filter 门禁通过，证据目标覆盖 1.0，但 grounded 目标覆盖仍为 0。
这只能证明控制逻辑，不是 DeepSeek 回答能力。

### 2. PDF Agent 困难集

`agent-hard-v1` 刻意选择首轮应当失败的任务，gold 来自官方 PDF 原页，不由模型生成或
评分。8 题覆盖：真实跨页表、重复行名与章节跨页、图形连线关系、合并/母公司口径冲突、
重述前后值、事实年度与证据文档年度错位、四目标完整覆盖和未来实际数据拒答。

```powershell
# 没有 key 时只生成诚实的 not_run 报告
uv run python scripts/evaluate_agent_hard.py

# 当前能力下限
uv run python scripts/evaluate_agent_hard.py --runtime deterministic-baseline

# CI/正式运行要求必须真实调用 provider
uv run python scripts/evaluate_agent_hard.py --require-remote
```

### P0 阶段结果

2026-08-20 已完成 P0-A 循环控制和 P0-B provenance target 改造。最终 P0-B8 及同代码
复现实验结果一致：5 个当前可执行 `compare` 难例全部通过，计划、行为、可执行事实、
可执行题通过和安全拒答均为 100%；折算全 8 题端到端为 5/8，另外 3 题仍明确标记为
不支持的 `extract` / `calculate`。

| 指标 | 首次真实基线 | P0-B8 | P0-B8 复现 |
|---|---:|---:|---:|
| task coverage | 0.6250 | 0.6250 | 0.6250 |
| 可执行题 plan exact | 0.6000 | **1.0000** | **1.0000** |
| 可执行题事实准确率 | 0.0000 | **1.0000** | **1.0000** |
| 可执行题通过率 | 0.0000 | **1.0000** | **1.0000** |
| 全 8 题端到端通过率 | 0.0000 | **0.6250** | **0.6250** |
| 安全拒答准确率 | 0.0000 | **1.0000** | **1.0000** |
| 模型请求 | 12 | **10** | **10** |
| 输入 token | 48,290 | **22,144** | **22,144** |
| 实际检索调用 | 25 | **13** | **13** |

每次增量、负结果、成本和原始报告见
[Agent P0 逐增量评测日志](./evaluation/agent-p0-improvement-log-zh.md)。最终报告为
`reports/agent/agent-hard-v1-deepseek-p0b8-provenance-citation-repair.json`，复现报告为
`reports/agent/agent-hard-v1-deepseek-p0b8-replication.json`。

### P1 PDF 工具阶段结果

P1 在同一 8 题上依次加入跨页 `extract`、`get_page_window`、带来源 `Decimal` 计算器和
`inspect_page_region`。最终任务覆盖、plan exact、行为、事实、引用覆盖、题级通过和安全
拒答全部为 1.0000，即 8/8 通过；这不代表任意 PDF 泛化能力，视觉样本也不是独立盲标。

| 指标 | P0-B8 | P1-A1 | P1-B1 | P1-C |
|---|---:|---:|---:|---:|
| task coverage | 0.6250 | 0.7500 | 0.8750 | **1.0000** |
| 可执行事实准确率 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 全 8 题端到端通过率 | 0.6250 | 0.7500 | 0.8750 | **1.0000** |
| 模型请求 | 10 | 13 | 16 | 19 |
| 输入 token | 22,144 | 27,602 | 33,760 | 36,769 |

逐次失败、修复、成本和原始报告见
[Agent P1 PDF 工具逐增量评测日志](./evaluation/agent-p1-pdf-tools-log-zh.md)。最终报告为
`reports/agent/agent-hard-v1-deepseek-p1c-pdf-geometry-relationships.json`。

### P2-A 整份文档盲测基线

P2-A 冻结 P1 代码，改用 5 份开发阶段未见年报的 34 题候选集。严格自动通过 17/34；
逐一核对失败答案、引用 excerpt 和 PDF 原页后为 24/34，但该来源复核仍非独立人工双审，
不能替换原始硬分。10 个真实失败为：4 个通用计算未支持、4 个文档范围/事实期间规划错误、
2 个多事实遗漏。运行共 73 次模型请求、167,700 token、71 次工具调用。

完整构造、两套分数、争议页依据和复现命令见
[Agent P2-A 整份文档盲测](./evaluation/agent-p2a-document-blind-baseline-zh.md)。原始报告和失败
归因分别为 `reports/agent/agent-hard-v2-deepseek-p2a-baseline.json` 与
`reports/agent/agent-hard-v2-p2a-failure-analysis.json`。

P2-B1 随后只修改报告年份/事实期间与唯一文档范围规划：严格通过 **17/34 → 21/34**，
4 fixed / 0 regressed，行为与安全拒答均为 100%；来源复核为 28/34。中间一次仅修 3 题的
“YYYY年年报”语法缺口也保留在汇总中。见
[P2-B1 单变量日志](./evaluation/agent-p2b1-document-scope-log-zh.md)。

### 首次真实基线（历史冻结）

2026-08-20 使用 `deepseek-v4-flash` 完成首次真实困难集运行。严格复核后，预算耗尽导致
的框架级拒答不计为正确拒答：

| 指标 | 规则下限 | DeepSeek 首次真实运行 |
|---|---:|---:|
| 总题数 | 8 | 8 |
| 现有命令可执行 | 5 / 8 | 5 / 8 |
| task coverage | 0.6250 | 0.6250 |
| 可执行题 plan exact | 0.6000 | 0.6000 |
| 可执行题事实准确率 | 0.0000 | 0.0000 |
| 端到端通过率 | 0.1250 | **0.0000** |
| 可执行拒答准确率 | 1.0000 | **0.0000** |

DeepSeek 运行共 12 次模型请求、25 次实际检索调用，输入 48,290 token、输出 2,183
token。3 个回答任务在模型轮次预算内没有提交结果，四目标题耗尽工具预算；合并/母公司
口径题在模型调用前就被旧目标规划器错误澄清。完整报告为
`reports/agent/agent-hard-v1-deepseek.json`。

规则下限唯一通过的是“2025 实际数据未披露”的安全拒答；DeepSeek 同题只是反复检索至
模型预算耗尽，因此严格计为失败。后续改造必须在同一数据集、同一索引和同一本地硬评分器
上提升，不能删除失败题或让 DeepSeek 充当裁判。评分规则变化可从已存轨迹离线重算，避免
重复产生模型费用：

```powershell
uv run python scripts/evaluate_agent_hard.py `
  --rescore-from reports/agent/agent-hard-v1-deepseek.json
```

## 首次基线暴露的关键缺陷与当前状态

1. **文档年度不等于事实年度**：已在 P0-B 修复。target 分离事实年度与报告年份，并由
   真实困难集的跨报告重述题验证。
2. **目标模型只有公司 × 年份**：已在 P0-B 修复。现在可表示合并/母公司口径和同一事实
   的调整前/调整后版本。
3. **扁平文本丢失视觉关系**：已在 P1-C 修复当前样本。按需读取 PDF 原生文字坐标和
   连接线，DeepSeek只确认关系，本地求和；扫描图和其他图表仍未证明。
4. **任务覆盖过窄**：已在 P1 增加 `extract` 和受控 `calculate`；尚未成为任意任务计算器。
5. **证据充足不等于答案正确**：旧规则基线可以达到 evidence coverage 1.0，同时事实
   准确率仍为 0；因此不能再用“找到 chunk”代替端到端回答评测。

## 下一步改造顺序

1. ✅ P2-B1 已完成：分离 `document_year`、`fact_period` 和唯一文档范围，+4 / 0 回退；
2. P2-B2 为多事实任务增加原子事实 checklist，不与通用计算混在同次实验；
3. 将现有来源绑定 Decimal 扩成白名单通用算术，而不是开放任意代码执行；
4. 为扫描页、混合页、复杂合并单元格和更多图表类型扩充独立 document-blind 难例；
5. 给 P1/P2 gold 补独立人工双审；任何升级继续报告正确率、拒答、引用、token 和失败率。
