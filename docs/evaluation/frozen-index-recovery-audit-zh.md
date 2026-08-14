# 冻结索引 `10fb...` 恢复审计

日期：2026-08-13

## 结论

`benchmark-v2` 绑定的 `10fb50419145d56720c9` 当前**不可按原身份精确重建**。应继续保留原 benchmark 与历史分数，不得把当前索引 ID 改写成旧 ID；后续正式复跑必须新建显式迁移版本，并做逐题 paired comparison。

## 可核验证据

- 历史报告证明 `10fb...` 是 Windows 环境下使用 `intfloat/multilingual-e5-small` 的 dense 索引，embedding dimension 为 384。
- 当前仓库与本机只保留两个完整 snapshot：
  - `ec12df...`：处理升级时 metadata 丢失的中间快照；
  - `c3f157...`：恢复 company/year/type metadata 后的 958-chunk 快照。
- 索引 ID 的历史与当前公式一致：`sha256(source_snapshot_sha256:dense_model:index_format_version)[:20]`，format version 为 3。
- 用两个现存 snapshot 和历史 E5 model name 计算得到 `044eb11b093ca095fdbd` 与 `9898c95e13d01c51c156`，都不等于 `10fb...`。
- 本机没有任何 `10fb...` generation manifest、`dense_embeddings.npy`、`dense_chunk_ids.json` 或对应完整 snapshot；Git 历史也没有提交这些大文件。
- 当前 lexical snapshot 生成 `a66f0caef7dd29101861`，冻结门禁因此正确拒绝正式 Retrieved 运行。

## 为什么不能“把 ID 改回去”

index ID 是 source snapshot、dense model 和 format 的内容身份，不是展示标签。直接修改 manifest 或 benchmark ID 会让历史 Retrieved/RAGAS 分数与另一份语料/索引静默混合，破坏项目最重要的可复现性主张。

## 迁移方案

1. 保留 `benchmark-v2`、其题目/答案/主指标和全部历史报告为只读历史基线。
2. 以当前 `c3f157...` 958-chunk snapshot 构建新的 dense generation；记录模型精确 revision、依赖锁、snapshot SHA、manifest 与 embeddings checksum。
3. 创建新的 benchmark migration manifest（不是覆盖 v2），逐条证明 35 个 unique gold 与 53 个 hard negative 的 chunk ID/SHA 映射。
4. 在新 index 上跑同一 retrieval matrix 和三轨生成，使用 paired fixed/regressed 报告；不得把结果与旧 `10fb...` 当同一 index 重复 run。
5. 只有迁移门禁和 paired review 通过后，才把新版本设为当前正式 benchmark；旧数字继续带 `index_id=10fb...` 展示。

## 仍需外部条件

- 构建 E5 dense 索引本身不需要商业 API key，但首次模型下载或缺失缓存需要网络。
- DeepSeek 三轨需要对应 endpoint 的 `DEEPSEEK_API_KEY`。
- 独立 judge 最好使用与回答模型不同的 provider/model key；否则必须继续标注 `independent_judge=false`。
