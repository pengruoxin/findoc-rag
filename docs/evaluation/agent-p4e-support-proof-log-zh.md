# Agent hard-v3 P4-E：有界原子事实 Support Proof

## 结论

P4-D 剩余错误的直接原因不是没有调用 Verifier，而是 Verifier 只返回了 4 个
`supported_requirement_ids`，用 76 个输出 token 完成整体“盖章”，没有说明每个 claim 被哪段
证据支持。P4-E 把最弱的一条原子事实契约改成可机器核验的
`requirement → exact claim → cited evidence quote` support proof；claim 与 proof 语言对齐较弱时，
再做一次只看最小证据和有限表头上下文的对抗挑战。

同代码、同 DeepSeek、只开关 support proof 的最终结果：

- 高语言重叠故障安全处理 **14/15 → 15/15**，unsafe accept **1 → 0**；
- 修复的是 P4-D 明确保留的
  `v3_601398_y23_asset_quality:relation_swap`：初审仍判 supported，但 proof quote 明确写
  “同比下降”，挑战器据此拒绝 claim 的“同比上升”；
- 正常组合严格通过保持 **45/48 → 45/48**；
- 正常 Verifier 请求 **18 → 22**，token **84,578 → 100,686**，增加 16,108
  （+19.05%）；
- 高重叠故障成本 45,416 → 52,358 token（+15.29%），请求 10 → 11；
- 已知故障 15/15、遗漏修复 5/5；开放风险本轮 15/15。

最终 15 条高重叠故障的处理构成为：模型拒答 9、本地会计符号拒答 5、结构证明不完整转
人工 1。`manual_review` 只计安全处理，不算自动回答成功。

## 输出契约

Verifier 不再只为选中 requirement 返回 supported ID，而要提交：

```json
{
  "requirement_id": "r1",
  "claim": "必须逐字复制当前原子 claim",
  "evidence_quotes": [
    {
      "evidence_chunk_id": "当前 claim 实际引用的 chunk",
      "quote": "证据中的原文片段"
    }
  ]
}
```

本地代码独立校验：

1. proof 必须覆盖被选中且被判 supported 的 requirement；
2. claim 必须逐字等于 `current_claims`，不能在证明阶段悄悄改答案；
3. chunk 必须来自该 requirement 当前 cited evidence，candidate evidence 不能替代引用；
4. quote 必须能在证据中按空白归一化或多行有序锚定；
5. claim 数值必须能在 quote 中直接或按已有单位换算得到；
6. claim 与 requirement + quote 的语言覆盖低于 0.90 时，不直接定罪，而是进入 proof
   challenge。

challenge 只接收该 claim、proof quote 和同一 cited chunk 的有限上下文。上下文只能解释表头、
年份、单位和相邻列，主体、关系与数值仍必须由 quote 绑定。它是同一 DeepSeek 的窄任务二次
调用，不是假装成跨模型独立裁判。

## 为什么只证明一项

第一版要求每个 requirement 都输出 proof。故障集虽然达到 15/15，但 calibration 正常抽取从
12/14 降到 **8/14**：PDF 表格经文字层抽取后，标签、数值和表头经常不是一个连续字符串；
8–13 项任务还会让工具参数过长并偶发截断。

最终策略按 claim 与原子 requirement contract 的相似度排序，只证明最低的一项：

- 高重叠故障的修改目标在 5 条轨迹中都是最低相似项；
- 每条轨迹最多增加一个 proof，不让输出随 requirement 数线性膨胀；
- 关键审计事项继续使用原有完整性 Verifier，不增加 proof；它们本身 requirement 多、输出长，
  本阶段也没有关系偷换故障；
- 如果结构化 JSON 已解析但 requirement partition 不完整，只允许一次纠错重试；语义矛盾、
  证据不足和 proof 不可信都不重试。

## 同代码消融

| 指标 | proof 关闭 | proof 开启 | 变化 |
|---|---:|---:|---:|
| 高重叠安全处理 | 14/15 | **15/15** | +1 |
| 高重叠 unsafe accept | 1/15 | **0/15** | -1 |
| 高重叠人工升级 | 0 | 1 | +1 |
| 高重叠模型错误 | 0 | 0 | 0 |
| 高重叠请求 | 10 | 11 | +1 |
| 高重叠 token | 45,416 | 52,358 | +6,942（+15.29%） |
| 正常组合严格通过 | 45/48 | **45/48** | 0 |
| 正常请求 | 18 | 22 | +4 |
| 正常 token | 84,578 | 100,686 | +16,108（+19.05%） |

正常新增 4 次调用来自低 proof 语言覆盖的表格 claim challenge。它们使用表头上下文后均通过，
没有正常 `manual_review`、`abstain` 或 error。audit 任务使用紧凑旧契约，所以正常输出事实分也
保持 P4-D：calibration 0.8519、dev post-hoc 0.9474。

## 逐轮实验

| 轮次 | 改动 | 高重叠 | 正常结果 | 结论 |
|---|---|---:|---:|---|
| v1 | 每个 requirement 都要逐字 proof | 15/15 | calibration 8/14 | 正常误升级过多，否决 |
| v2 | 空白/多行锚定 + 低覆盖 challenge | 13/15，另 2 error | calibration 11/14 | 长输出仍不稳定 |
| v3 | 只选最低 contract 相似项 | 7/15，另 8 error | calibration 12/14 | 模型同时报 proof 与 finding，校验未归一 |
| v4 | actionable finding 优先于 supported/proof | 14/15，另 1 error | 未做完整正常集 | 安全但结构噪声仍存在 |
| v5 | proof 结构错误转人工 | 15/15 | dev 11/14 | 最小表格行缺表头，挑战器误判证据不足 |
| v6 | challenge 补有限表头上下文；audit 保持紧凑 | 15/15 路径保持 | dev 13/14；calibration 一次 11/14 | 仅剩 partition 随机缺项 |
| v7 | partition 一次纠错重试 | **15/15** | calibration 12/14、dev 13/14 | 最终采用 |

所有中间报告均保留在最终汇总的 `iterations`，失败版本没有被覆盖。

## 回归

- 已知主体/数字/遗漏故障：15/15 安全处理，unsafe 0，error 0，遗漏修复 5/5；20 次请求，
  92,818 token；
- P4-C 开放风险：15/15 安全处理，unsafe 0，error 0；15 次请求，77,252 token；
- P4-D 曾观察到开放风险 14–15/15 的同模型波动，因此本轮单次 15/15 仍不能写成稳定 100%；
- frozen test 始终未打开，dev 仍为 post-hoc。

## 人工升级边界

proof 缺失、claim 非原样、引用不属于当前 requirement、quote 无法落回证据或结构纠错重试仍
失败时，轨迹写入：

- `final_decision=manual_review`；
- `human_review_required=true`；
- 可审计的 `human_review_reasons`；
- 用户侧自动回答暂停，结果保持 `abstain`。

当前实现只产生结构化人工升级状态，**尚未连接工单队列、审批 UI 或人工结论回写**，不能宣称
已经完成端到端 human-in-the-loop。

## 决策与限制

采用“非 audit Verifier 路径只证明最低 contract 相似项”，不采用每项都证明。理由是它在正常
严格分不回退的前提下修复唯一剩余高重叠关系错误，但正常 Verifier token 增加 19.05%，不适合
扩大到所有查询。

边界仍然明确：主 Agent、初审和 proof challenge 都是 `deepseek-v4-flash`，只有上下文/职责
隔离，没有模型独立性；15 个故障样本很小；最低相似项选择和 0.90 阈值使用了 calibration 与
post-hoc dev；一次纠错重试处理的是结构噪声，不提升事实推理上限。

下一步优先级：接入不同 provider/model 做独立 challenge 消融；若仍只有 DeepSeek，则先完成
人工复核队列/审批回写和结构化表格 cell/geometry proof，再考虑 frozen test 一次性确认。

全仓验证结果：**400 passed**；Ruff、CLI 参数加载、组合报告生成、SHA/指标一致性断言与
`git diff --check` 全部通过。

## 产物

- 汇总：`reports/agent/agent-hard-v3-p4e-support-proof-improvement.json`
- calibration 组合：`reports/agent/agent-hard-v3-calibration-deepseek-p4e-composed.json`
- dev 组合：`reports/agent/agent-hard-v3-dev-deepseek-p4e-composed.json`
- 高重叠 proof-off：`reports/agent/agent-hard-v3-deepseek-p4e-support-proof-ablation-off-high-overlap-v1.json`
- 高重叠 proof-on：`reports/agent/agent-hard-v3-deepseek-p4e-bounded-proof-context-high-overlap-v7.json`
- 已知故障：`reports/agent/agent-hard-v3-calibration-deepseek-p4e-known-faults-v2.json`
- 开放风险：`reports/agent/agent-hard-v3-deepseek-p4e-open-risk-regression-v1.json`
