# Agent hard-v3 P4-D：高语言重叠关系与会计符号风险

## 结论

P4-C 能发现“claim 使用了证据之外的新语言”，但无法判断大量复用原文词语后是否偷偷换了
指标、关系或正负号。P4-D 为这个已知盲区补了 15 个高语言重叠故障，字符覆盖率为
0.824–1.000，全部不会触发 P4-C 的低覆盖信号。

最终采用轻量方案：简单抽取的 claim 偏离原子 requirement contract 时只升级 DeepSeek
Verifier；claim 擅自把正数改成负数或会计括号时，本地确定性拒答；requirement 中的必要
数字没有进入回答时升级修复。结果是：

- 高重叠故障安全处理 **5/15 → 14/15**，unsafe accept **9 → 1**；
- 标签偷换 5/5、会计符号翻转 5/5、关系方向偷换 4/5；
- 正常组合严格通过保持 **45/48 → 45/48**，模型请求保持 18；
- 正常 Verifier 成本 81,812 → 84,198 token，增加 2,386（+2.92%）；
- 修正评测器后，已知故障由 13/15 提升到 15/15，细项遗漏修复 3/5 → 5/5；
- P4-C 开放风险复跑的路由均为 15/15，但同模型 Verifier 的安全处理出现 **14–15/15**
  波动，不能把单次 15/15 写成稳定能力。

仍未识别的是 `v3_601398_y23_asset_quality:relation_swap`：把“不良贷款率同比下降”改成
“同比上升”后，DeepSeek Verifier 接受了错误回答。没有为这一题增加“上升/下降”专用
关键词规则，因为那只会把 benchmark 答案写进系统。

## 评测设计

从 5 条已保存正常轨迹各注入 3 类错误，共 15 条：

1. `label_swap`：数值和页面不变，把指标或业务标签替换成同页另一标签；
2. `relation_swap`：把母公司/少数股东、增长/下降、收入/成本等关系换成相邻关系；
3. `accounting_sign_flip`：保留数字字符，只用会计括号把正数改成负数。

它们都保留正确公司、年份、数字、单位、引用文档和绝大多数原文词语。因此该集合不是再测
P4-B 的主体/时期硬冲突，也不是再测 P4-C 的低语言覆盖，而是专门测“高词面相似、事实
关系错误”。评测脚本为
`scripts/evaluate_agent_high_overlap_faults.py`，支持分别关闭 contract 与会计符号信号。

## 逐次结果

| 版本 | 核心变化 | 安全处理 | unsafe | 请求 | token | 结论 |
|---|---|---:|---:|---:|---:|---|
| P4-C 基线 | 只有复杂度路由与低语言覆盖 | 5/15 | 9 | 9 | 37,959 | 暴露盲区 |
| contract 关闭 | 同一版 prompt，只关 contract 信号 | 6/15 | 9 | 9 | 39,332 | 对照臂 |
| contract 开启 | 简单 claim 偏离 requirement 时升级 | 11/15 | 4 | 15 | 62,289 | 标签 5/5，符号仍弱 |
| focused evidence | 重复拼接聚焦证据 | 13/15 | 2 | 10 | 82,906 | 效果和成本都更差，删除 |
| 轻量 contract + sign | 本地会计符号冲突门禁 | **14/15** | **1** | 10 | 41,839 | 采用 |
| + 数值完整性 | 必要数字遗漏升级修复 | **14/15** | **1** | 10 | **41,804** | 最终版本 |

`focused evidence` 不是只在故障集上昂贵：正常 48 题的 Verifier token 从 P4-C 的 81,812
上升到 **166,273（+103.23%）**，高重叠结果却只有 13/15，因此实现已撤回，报告保留为
负实验。

## 三个风险信号的边界

### Requirement contract 偏离

仅对少于 4 个 requirement、且不是审计事项的简单抽取生效。规范化 claim 与对应原子
requirement 的相似度低于 0.99 时记录 `requirement_claim_divergence`，状态为 `review`，只能
触发 Verifier，不能由字符串相似度直接定罪。复杂任务原本就会复核，无需重复增加请求。

### 会计符号冲突

若 requirement 只包含正数，而 claim 明确为同一数字增加负号或会计括号，记录
`accounting_sign_conflict` 并本地拒答。这是可确定验证的数值语义，不依赖模型。5 个符号
翻转全部在调用 Verifier 前安全关闭。

### 必要数字完整性

每个 requirement 中受支持的数字都必须出现在对应 claims 中；缺失时记录
`missing_requirement_numeric` 并进入最多一次修复。它不直接拒答，避免因为合法改写或单位
表达差异误杀正常答案。当前 109 条正常 claim 没有新增误报。

## 正常集与回归

| 范围 | P4-C | P4-D | P4-D 请求 | P4-D token |
|---|---:|---:|---:|---:|
| calibration | 22/24 | 22/24 | 10 | 46,408 |
| dev（post-hoc） | 23/24 | 23/24 | 8 | 37,790 |
| 合计 | **45/48** | **45/48** | **18** | **84,198** |

正常请求没有增加；+2.92% token 来自更明确的原子事实 contract 提示和远程输出波动，不是
额外多 Agent 调用。

已知故障评测同时修复了一个评测器缺陷：旧 omission mutation 把数字中的 ASCII 千分位
逗号当成分句符，两条所谓“遗漏”实际把 `33,521,174` 截断成 `33`。旧结果不能与真正的
遗漏修复率比较。修正后：

| 评测器/门禁 | 安全处理 | omission 修复 | 请求 | token |
|---|---:|---:|---:|---:|
| 旧逗号切分评测器 | 15/15 | 3/5 | 16 | 66,452 |
| 修正评测器，完整性门禁前 | 13/15 | 3/5 | 16 | 67,123 |
| 修正评测器，完整性门禁后 | **15/15** | **5/5** | 20 | 84,533 |

第一行标为不可比，只保留审计记录。真正可比的是后两行。

P4-C 开放风险在 P4-D 当前代码上复跑时，15 条均稳定进入 Verifier；第一次安全处理
15/15，当前代码复跑为 14/15，漏过一条 `semantic_negation`。这说明路由信号有效，但
同一 DeepSeek 同时充当主 Agent 与 Verifier 时仍存在相关性和采样波动。

## 决策与下一步

采用轻量 contract、会计符号和必要数字完整性门禁；不采用证据重复拼接；不再针对剩余
“上升/下降”样本继续添加词表规则。当前更有信息量的下一步是：

1. 用不同 provider/model 做真正独立 Verifier 消融，比较错误相关性和成本；
2. 对仍无法确定的高风险结果输出结构化 support proof，或升级人工复核；
3. 冻结当前代码和阈值后，才一次性打开 frozen test 做无偏确认。

frozen test 本阶段始终封存；dev 仍为 post-hoc，15 个高重叠故障仍是小规模受控注入。

全仓验证结果：**396 passed**；Ruff、组合报告生成、报告一致性断言与
`git diff --check` 全部通过。

## 产物

- 汇总：`reports/agent/agent-hard-v3-p4d-high-overlap-improvement.json`
- calibration 组合：`reports/agent/agent-hard-v3-calibration-deepseek-p4d-composed.json`
- dev 组合：`reports/agent/agent-hard-v3-dev-deepseek-p4d-composed.json`
- 高重叠最终：`reports/agent/agent-hard-v3-deepseek-p4d-high-overlap-final-v3.json`
- 已知故障最终：`reports/agent/agent-hard-v3-calibration-deepseek-p4d-known-faults-fixed-v3.json`
- 开放风险复跑：`reports/agent/agent-hard-v3-deepseek-p4d-open-risk-regression-v2.json`
