# FinDocRAG 面试说明

## 一句话介绍

FinDocRAG 是面向中国上市公司年报的结构感知、可追溯 RAG 系统。系统不仅返回答案，还记录 PDF 处理、chunk 来源、检索版本、引用证据和失败原因，支持离线评测与持续优化。

## 项目亮点

1. **真实复杂文档**：针对中文年报中的多级章节、跨页内容、页眉页脚、脚注和复杂表格设计，而不是 toy QA 数据集。
2. **结构感知 chunking**：按照标题层级、页面元素和 token 预算切分，保留 `section_path`、页码、bbox 和源元素映射。
3. **混合检索评测**：同时支持 BM25、multilingual-e5 dense retrieval 和 RRF hybrid，并统一记录 Hit@K、MRR、索引版本和失败案例。
4. **metadata routing**：利用公司、年份和文档范围过滤，避免多家公司年报模板相似导致串文档。
5. **证据门禁与拒答**：证据与问题公司、年份或核心指标不一致时拒绝生成，避免 LLM 在错误证据上编造流畅答案。
6. **可替换回答层**：默认支持 extractive / deterministic-table；配置 `DEEPSEEK_API_KEY` 后接入 DeepSeek OpenAI-compatible API。
7. **可观测性**：保存 query、候选、最终证据、trace、处理问题和实验配置，使错误可以定位到 PDF、chunk、召回、重排或生成阶段。
8. **三轨生成评测**：同一冻结集分别运行 Oracle、真实检索和受控干扰；48 条问题拆成 120 个原子事实，并用 53 个真实年报 hard negatives 测错公司、错期间、错口径、部分证据和因果拼接。

## 技术路线

```text
年报发现/下载 -> provenance manifest -> PDF Document IR
-> 结构感知 chunking -> BM25 + Dense -> RRF/重排
-> 公司/年份路由 -> 证据门禁 -> 表格或 LLM 回答
-> citation/trace -> Oracle/Retrieved/Robustness 评测与失败分析
```

## 面试回答要点

### 为什么不用纯向量检索？

财务问题包含公司名、年份、会计科目和数字，BM25 对专有名词和精确词匹配更稳；dense 对自然语言改写更有帮助，因此用混合检索并通过同一 holdout 对比，而不是凭感觉选型。

### 为什么要做 metadata routing？

不同公司的年报存在大量相同章节名称。没有公司和年份过滤，系统可能召回另一家公司的“审计委员会”段落，LLM 仍会生成看似合理的答案。路由过滤和证据门禁共同降低这一风险。

### PDF 表格怎么处理？

先建立带页码、坐标和元素映射的 Document IR，保证覆盖率；发现表格线性化导致行列关系丢失后，单独记录 processing issue，并对季度指标增加确定性行列抽取、单位校验和回归测试。

### 如何证明项目不是 demo？

每次实验固定数据集版本、索引版本和配置，检索侧记录 Recall/Precision/MRR/NDCG，生成侧记录事实、单位、引用、行为和 RAGAS 指标。数据门禁会验证“数值确实位于绑定原文”，前后版本做逐题配对并列出修复与回归，而不是只展示成功回答。

## 90 秒介绍模板

“我做的是 FinDocRAG，一个针对中国上市公司年报的可追溯 RAG。难点不在调用大模型，而在中文复杂 PDF 的结构恢复、跨页表格、公司年份混淆，以及如何证明答案真的有证据支持。我把 PDF 转成带页码和坐标的 Document IR，做结构感知 chunking，再建立 BM25、multilingual-e5 和加权 RRF 检索链。生成评测不是只看一条答案：我把 48 个问题拆成 120 个原子事实，并分别跑 Oracle、真实检索和带 53 个真实干扰块的 Robustness 轨道。每次修改都做逐题配对，能定位问题来自解析、chunk、召回、口径判断还是生成。”

## 当前结果与边界

当前 holdout v2 的 BM25、Dense、Hybrid 结果位于 `reports/ranking/`，生成数据卡位于 `reports/generation/generation-eval-v1-dataset-card.md`。面试时应说明两者都是 assistant-reviewed provisional 回归集：当前只有两家公司、一个年度，旧 32 条 DeepSeek 分数也不能代表新版 48 条数据；指标用于工程迭代，不宣称通用 SOTA。后续重点是 document-level blind test、人工独立复核、OCR/跨页表格、强 reranker 和失败回放。
