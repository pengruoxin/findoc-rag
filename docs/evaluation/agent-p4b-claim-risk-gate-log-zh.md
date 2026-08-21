# Agent hard-v3 P4-B：确定性 Claim 风险闸门

## 结论

P4-B 在高风险 Evidence Verifier 前增加了一个**零模型调用**的本地风险闸门，解决 P4-A
仍会放过跨公司主体错误的问题，同时不把所有可疑数字直接拒掉。

- 正常集严格通过保持 **45/48（93.75%）→ 45/48（93.75%）**，0 提升、0 回退；
- 28 条已保存的正常抽取轨迹中，20 条进入闸门并全部通过，8 条不适用，误拒绝 0；
- 15 个 claim 故障的安全处理由 **14/15（93.33%）提升到 15/15（100%）**；
- 5 个跨公司主体篡改全部在 DeepSeek 调用前本地拦截；
- 故障实验请求数 **25 → 20**，token **100,941 → 80,882**，减少 20,059
  （19.87%）；
- 数值篡改 5/5 安全拒答，删除已支持细项 5/5 修复后通过二次复核。

因此多 Agent 策略不变：**普通查询仍走单 Agent；只有高风险复杂抽取显式开启独立
Verifier。** P4-B 负责在它之前处理便宜、明确、可解释的冲突。

## 为什么不是“正则看到不同就拒答”

PDF 财务数据存在合法的复杂表达：年报会引用以前年度、表格单位可能是千元/百万元而答案
转换成亿元、负数可能用括号或绝对值复述、同比比例也会发生舍入。初版规则在 28 条正常
轨迹上曾误报 5 条，包括百万元与亿元换算、千元到亿元舍入、比较期年份和负百分比。

最终策略分成两级：

| 级别 | 检查 | 动作 |
|---|---|---|
| 硬冲突 | 明确的错公司主体、事实期间不在 requirement/引用证据中、引用越出文档公司或报告年范围 | 本地拒答，不调用模型 |
| 风险提示 | claim 中的数字或单位无法在 requirement/引用证据中对齐 | 不本地定罪，强制交给 DeepSeek Verifier |

数字对齐支持正负号别名、百分号、千元/万元/百万元/千万元/亿元/万亿元换算和有限舍入
容差。规则只证明“存在明确冲突”或“需要复核”，不试图替代语义验证模型。

## 时间语义边界

闸门不会拿运行机器的 2026 年直接判断 2022 年文档里的“去年”。相对时间先由查询路由以
明确的 `question_time` 解析；进入 Agent 后，`document_year`（证据载体年份）和
`fact_period`（事实所属期间）分开保存。P4-B 检查的是 claim 年份是否出现在原子事实要求
或其引用证据中，因此年报中的比较期、上年同期等只要证据明确出现，就不会因当前系统
年份而被误判。

## 正常答案配对结果

| 范围 | P4-A | P4-B | 触发 Verifier | P4-B 新增 token |
|---|---:|---:|---:|---:|
| calibration | 22/24 | 22/24 | 10 | 44,518 |
| dev（post-hoc） | 23/24 | 23/24 | 8 | 36,042 |
| 合计 | **45/48** | **45/48** | 18 | **80,560** |

P4-A 对应正常集成本为 80,817 token。P4-B 没有减少正常复杂题的 Verifier 请求；257 token
差异来自远程输出长度波动，不能当作闸门节省。可以确定的是本地闸门自身增加 0 个模型
请求、0 token。

## 故障注入配对结果

| 故障 | P4-A | P4-B | P4-B 最终动作 |
|---|---:|---:|---|
| 数值篡改 | 5/5 | 5/5 | DeepSeek Verifier 全部拒答 |
| 错公司主体 | 4/5 | **5/5** | 5 个全部本地拒答，0 模型请求 |
| 删除已支持细项 | 5/5 | 5/5 | 5 个全部修复并通过二次复核 |
| 合计 | **14/15** | **15/15** | unsafe accept 1 → 0 |

| 成本 | P4-A | P4-B | 变化 |
|---|---:|---:|---:|
| 模型请求 | 25 | 20 | -5（-20%） |
| 输入 token | 95,772 | 76,478 | -19,294 |
| 输出 token | 5,169 | 4,404 | -765 |
| 总 token | 100,941 | 80,882 | **-20,059（-19.87%）** |

## 实现和轨迹

- `AgentTaskTrace.claim_risk_gate` 持久化规则版本、状态、检查数和逐 requirement finding；
- `reject` 使用 `claim_risk_gate_rejected` 停止原因，provider 明确记为
  `claim-risk-gate`；
- `review` 会覆盖原来的简单任务快路径，确保可疑数字或单位必须接受模型复核；
- 一次修复后重新执行本地闸门，再进行第二次独立 Verifier；
- CLI 仍使用显式 `--evidence-verifier`，没有把多 Agent 默认打开；
- 公司词表由当前索引/评测轨迹中的公司元数据提供，未访问外部知识。

全仓验证结果：**388 passed**；Ruff、报告组合脚本和差异空白检查全部通过。

## 限制与下一步

- 15 个故障样本仍然小，且来自受控合成退化，不能代表线上自然错误率；
- 未出现在公司词表中的别名、简称或错别字可能绕过主体规则；
- 数字存在于证据不等于语义关系正确，所以数字/单位异常仍需模型复核；
- dev 是 post-hoc，frozen test 始终封存，当前结果不是无偏最终确认；
- 主 Agent 和 Verifier 仍同为 DeepSeek，独立的是上下文和职责，不是 provider。

下一单变量建议是 P4-C 自动高风险路由：只依据任务复杂度、审计类型和本地风险信号决定
是否开启 Verifier，并评测相同安全率下能减少多少正常请求。完成路由后，再决定是否一次性
打开 frozen test；在此之前不增加更多自由角色。

## 产物

- 汇总：`reports/agent/agent-hard-v3-p4b-claim-risk-gate-improvement.json`
- calibration 组合：`reports/agent/agent-hard-v3-calibration-deepseek-p4b-composed.json`
- dev 组合：`reports/agent/agent-hard-v3-dev-deepseek-p4b-composed.json`
- 正常校准：`reports/agent/agent-hard-v3-calibration-deepseek-p4b-claim-risk-gate-v1.json`
- 正常开发：`reports/agent/agent-hard-v3-dev-deepseek-p4b-claim-risk-gate-posthoc-v1.json`
- 故障注入：`reports/agent/agent-hard-v3-calibration-deepseek-p4b-claim-risk-gate-faults-v1.json`
