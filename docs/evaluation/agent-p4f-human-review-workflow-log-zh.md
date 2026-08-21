# Agent P4-F 人工复核闭环日志

## 1. 为什么这一阶段必要

P4-E 已能在 support proof 无法独立验证时返回 `manual_review`，但那只是一个安全状态，
不是可执行流程。阶段开始前存在四个明确缺口：

1. 自动回答被替换成拒答文本后，原始候选答案没有单独保留，复核人无法判断要批准什么；
2. 没有持久化待审队列，也没有 Agent 命令查看 claim、页码、chunk 和原文；
3. 没有批准、修正、驳回的结果对象，`manual_review` 不能形成最终业务结论；
4. 没有防过期和防篡改绑定，旧证据生成的批准可能错误套到后来变化的任务上。

因此 P4-F 不修改 DeepSeek 回答能力，也不把人工升级计作模型答对；它只把“安全暂停”
变成可审计的人机协作闭环。

## 2. 实现

### 2.1 保留被拦截的候选结果

`EvidenceVerificationTrace.candidate_result` 在进入人工复核前保存当前完整
`AgentTaskResult`，包括：

- 原始候选答案与 citation；
- 每个 atomic requirement 的 claim；
- requirement→chunk 证据绑定；
- 已完成的 scope validation 状态。

对用户的自动结果仍保持 `abstain`，所以候选答案不会绕过门禁直接返回。

### 2.2 不可变审核包

`HumanReviewStore` 为每条人工升级生成确定性的 review ID，并保存：

- task ID、index ID、query；
- 完整 task trace 的规范化 SHA-256；
- 人工升级原因；
- 候选答案；
- 原子 claim 对应的 chunk 内容 SHA-256、文档、页码、章节与 excerpt。

审核包与 resolution 分目录保存。写入使用同目录临时文件、刷新磁盘后创建不可覆盖的硬链接；
同一 trace 重复 enqueue 是幂等的，不同内容不能覆盖已有包。

### 2.3 三种人工决策

- `approve`：恢复原候选答案，但 citation 必须仍落在审核包证据中；
- `correct`：要求明确提交修正答案，并至少绑定一个审核包内 chunk；
- `reject`：保持 `abstain`，不带证据 ID。

每个 review 只能生成一个 resolution。resolution 记录 reviewer、时间、comment、最终 outcome、
答案与证据 ID；原始模型 trace 不被改写。

### 2.4 两个关键安全门禁

1. **stale trace guard**：审批时重新加载 task trace 并计算 SHA-256，和 enqueue 时不同就拒绝；
2. **outside evidence guard**：修正答案引用的 chunk 不在不可变审核包中就拒绝。

此外修复了一个持久化缺陷：Pydantic computed fields 曾被写进 Verifier JSON，严格 schema
重新加载会报 extra field。新文件不再持久化这些派生字段；加载器仍兼容旧文件中的
`request_count/input_tokens/output_tokens`。

## 3. Agent 命令

新任务一旦进入 `evidence_verifier_manual_review`，`agent run` 会在 task trace 旁自动创建
review packet。历史人工任务可手动 enqueue：

```powershell
uv run findoc-rag agent review enqueue <task-id>
uv run findoc-rag agent review list
uv run findoc-rag agent review inspect <review-id>

uv run findoc-rag agent review resolve <review-id> approve `
  --reviewer reviewer-a

uv run findoc-rag agent review resolve <review-id> correct `
  --reviewer reviewer-a `
  --corrected-answer "修正后的答案" `
  --evidence-chunk-id <chunk-id>

uv run findoc-rag agent review resolve <review-id> reject `
  --reviewer reviewer-a
```

自定义 `--task-dir` 时，`agent run` 默认把队列放到相邻的 `reviews` 目录；独立运行 review
命令时可显式传 `--task-dir` 和 `--review-dir`。

## 4. 评测

评测脚本：

```powershell
uv run python scripts/evaluate_agent_human_review_workflow.py
```

输入不是合成空对象，而是已保存的真实 `deepseek-v4-flash` 开发集 Agent trace
`v3_601318_y23_segments`。脚本只把该 trace 包装为人工升级状态，随后在临时目录执行闭环，
不调用 provider，不打开 frozen test。

| 契约 | 结果 |
|---|---:|
| 候选结果保留 | 通过 |
| claim、页码和证据可查看 | 通过 |
| pending 队列持久化 | 通过 |
| approve | 通过 |
| correct + packet evidence 绑定 | 通过 |
| reject | 通过 |
| 原始 trace 不变 | 通过 |
| resolution 不可覆盖 | 通过 |
| stale trace 拒绝 | 通过 |
| 审核包外证据拒绝 | 通过 |
| **合计** | **10/10** |

原始报告：
`reports/agent/agent-p4f-human-review-workflow-v1.json`。模型请求、输入 token、输出 token
均为 **0**。这 10/10 是工作流契约通过率，不是问答准确率，也不是人工审核准确率。

## 5. 回归验证

- P4-F 定向：55 passed；
- 全仓：**407 passed**；
- Ruff：通过；
- frozen test：未打开。

## 6. 面试边界与下一步

现在可以回答“模型不确定后怎么办”：系统不是简单打印一个 flag，而是保存候选、证据页和
hash，进入一次性审批，旧 trace 或包外证据均不能被批准。

仍不能声称的部分：

- 当前只有本地 CLI 队列，没有 Web 审核页面、鉴权、角色权限或通知；
- 10/10 只证明状态机和 provenance 安全，不证明人审判断质量；
- 没有第二名审核人、一致性指标或 SLA；
- resolution 是原始 trace 之外的追加记录，需要上层业务显式消费最终 outcome；
- 同一 DeepSeek 作为回答与 Verifier 的独立性问题没有改变。

下一单变量应回到 PDF 证据质量：为表格回答保存 cell 坐标、表头/行头路径与 PDF 区域 hash，
让 reviewer 能检查“数字在哪个格子、属于哪一行哪一列”，再用复杂合并单元格和扫描页难例
评测，而不是继续增加 Agent 角色。
