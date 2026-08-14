# 表格抽取评测 v1（table-eval-v1）

- 数据集：`table-eval-concentration-v1` | 表数：2 | 标注单元格：8
- 匹配规则：单元格三元组 (行标签归一化, 列头, 数值归一化) 完全一致才算对。

| 表型 | 实现 | 表数 | gold cells | 正确 cells | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| concentration | ✅ | 2 | 8 | 8 | 1.0000 | 1.0000 |

## 逐表结果

| table | 类型 | gold | 预测 | 正确 | Precision | Recall | 错误行（归一化） |
|---|---|---:|---:|---:|---:|---:|---|
| moutai_concentration | concentration | 4 | 4 | 4 | 1.0000 | 1.0000 | - |
| yili_concentration | concentration | 4 | 4 | 4 | 1.0000 | 1.0000 | - |
