# Agent hard-v3 P4-A：独立 Evidence Verifier 实验

## 结论

P4-A 完成，但结论不是“项目应该全面升级为多 Agent”。

在 P3-B 的 calibration + dev 48 题上，增加独立证据审计后严格通过仍为
**45/48（93.75%）→ 45/48（93.75%）**，0 提升、0 回退。18 个复杂抽取答案被单独复核，
10 个简单或非回答任务不触发；18 次新增模型请求消耗 80,817 token，正常答案全部原样
通过，没有发生真实修复。

在 15 个存量轨迹故障注入上，最终版本安全处理 14/15：

- 数值篡改：5/5 拦截；
- 跨公司主体篡改：4/5 拦截，仍有 1 个错误放行；
- 删除证据已经支持的答案细项：5/5 发现，并在一次修复后再次通过审计；
- Verifier 结构解析错误：0。

因此当前证据支持的产品决策是：**保留可选的高风险 Evidence Verifier，不把多 Agent 设为
默认路径，也不继续扩成多个自由讨论的角色。**

## 研究依据

- [Self-RAG](https://arxiv.org/abs/2310.11511)把检索、生成和反思作为可控制动作，并报告
  事实性与引用准确率收益；
- [CRAG](https://arxiv.org/abs/2401.15884)使用检索评价器决定是否触发纠正动作，而不是
  无条件增加检索；
- [Anthropic 的 agent 工程建议](https://www.anthropic.com/engineering/building-effective-agents)
  建议从最简单可工作的组合模式开始，并明确说明 agentic 系统通常以延迟和成本换取效果；
  evaluator–optimizer 适合存在明确验收标准、且允许定向修复的任务。

本项目采用的是 evaluator–optimizer 工作流，不是群聊式多 Agent。

## 实现

### 独立上下文

Verifier 不接收主 Agent 的消息历史，只接收：

- 原始问题与文档范围；
- 原子事实 requirement；
- 当前 requirement → claim；
- 当前引用和有限候选 evidence；
- 主体、期间、集团/分部口径及证据类型。

它不能检索外部信息，也不能使用参数记忆替代证据。

### 条件路由

以下任务才触发：

- 包含审计风险或审计应对；
- 原子事实 requirement 数量不少于 4。

简单抽取、拒答和澄清继续走单 Agent 快路径。

### 结构化判定

每项 requirement 只能被分类一次：

- `supported`：当前引用直接、完整支持；
- `incomplete`：证据包含问题要求的细项，但回答遗漏；
- `contradicted`：主体、期间、数值或口径与证据冲突；
- `insufficient_evidence`：当前证据无法支持。

`contradicted` 和 `insufficient_evidence` 直接安全拒答；`incomplete` 最多触发一次修复。修复
结果必须通过原有本地 requirement/引用门禁，并再经过一次独立 Verifier；第二次仍不过就
拒答。产品轨迹只记录结构化 finding、token、延迟和校验错误，不保存自由形式思维链。

## 正常答案配对结果

| 范围 | P3-B | P4-A | 触发复核 | 新增 token |
|---|---:|---:|---:|---:|
| calibration | 22/24 | 22/24 | 10 | 44,518 |
| dev（post-hoc） | 23/24 | 23/24 | 8 | 36,299 |
| 合计 | 45/48 | 45/48 | 18 | 80,817 |

18 个正常复杂答案全部为 `accept_original`，没有真实修复。现有 3 个严格失败主要是同义词
exact 差异或完整枚举评分差异；逐项证据审计认为当前 claim 已被引用支持，因此没有理由
为了提高 benchmark 分数强行改写。

## 故障注入结果

最终故障集固定使用 5 个 calibration 正常复杂答案，每题注入三种 claim 级错误：

| 故障 | 安全处理 | 结果 |
|---|---:|---|
| 数值篡改 | 5/5 | 全部拒绝输出冲突数字 |
| 错公司主体 | 4/5 | 1 个比亚迪产品任务错误放行 |
| 删除已支持细项 | 5/5 | 全部完成一次修复并通过二次复核 |
| 合计 | **14/15（93.33%）** | unsafe accept 1/15 |

最终故障实验使用 25 次模型请求、100,941 token。故障注入证明了 Verifier 能发现一部分
主答案评分不会主动暴露的安全问题，但它是受控合成退化，不等于自然线上错误率。

## 负实验

1. calibration v1 只检查主 Agent 已规划的 requirement：10 个任务全部原样通过，分数
   不变，却增加 43,223 token。它无法发现 requirement 清单本身之外的规划漏项。
2. 第一次故障脚本在简单 claim 上找不到可截断标点，已发生至少 2 次远程请求后异常退出；
   因报告尚未持久化，token 不可追溯。该次明确记为无效运行，不计入正式成本。
3. 初版故障集对“错误主体公司”和重复出现数值的删除不够真实；随后改成真实跨公司主体，
   并删除同一数值的全部出现位置。旧报告保留，但不能与最终故障集直接比较。
4. dev post-hoc v1 中，DeepSeek 把两个 `supported` 项放进 `findings`，内容等价但字段位置
   不符，安全关闭导致 13/14 → 12/14。最终解析器仅将 `verdict=supported` 归并回支持 ID；
   未知 verdict、漏 requirement、未知引用和其他非法结构仍拒绝。dev v2 恢复 13/14。

全部已持久化的 P4-A 实验新增成本为 599,776 token；另有上述一次无法计量的无效运行。

## 限制与决策

- Verifier 与主 Agent 当前都使用 `deepseek-v4-flash`；独立的是上下文和角色，不是模型
  provider，因此两者可能共享盲点。
- 15 个故障样本不足以证明通用安全性，且主体篡改仍有 1/5 漏检。
- dev 是 post-hoc，不是无偏确认；frozen test 始终未打开。
- 当前 benchmark 的正常答案已经较强，Verifier 没有带来事实或严格分提升，默认开启的
  成本收益比不成立。

产品策略：CLI 提供显式 `--evidence-verifier`；审计事项、高风险多事实抽取可以开启，普通
查询默认关闭。下一次值得做的实验是使用不同 provider/model 作真正独立复核，或增加本地
确定性公司/期间冲突门禁；在这些方案超过当前 14/15 且正常集不回退前，不扩成更多 Agent。

## 产物

- 汇总：`reports/agent/agent-hard-v3-p4a-evidence-verifier-improvement.json`
- calibration 组合：`reports/agent/agent-hard-v3-calibration-deepseek-p4a-composed.json`
- dev 组合：`reports/agent/agent-hard-v3-dev-deepseek-p4a-composed.json`
- 最终正常校准：`reports/agent/agent-hard-v3-calibration-deepseek-p4a-evidence-verifier-v3.json`
- 最终正常开发：`reports/agent/agent-hard-v3-dev-deepseek-p4a-evidence-verifier-posthoc-v2.json`
- 最终故障注入：`reports/agent/agent-hard-v3-calibration-deepseek-p4a-evidence-verifier-faults-v4.json`
