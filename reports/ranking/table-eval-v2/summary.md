# 表格抽取评测 v1（table-eval-v1）

- 数据集：`table-eval-v1` | 表数：8 | 标注单元格：149
- 匹配规则：单元格三元组 (行标签归一化, 列头, 数值归一化) 完全一致才算对。

| 表型 | 实现 | 表数 | gold cells | 正确 cells | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| annual_data | ✅ | 2 | 39 | 39 | 1.0000 | 1.0000 |
| note_cost | ✅ | 2 | 24 | 24 | 1.0000 | 1.0000 |
| quarterly | ✅ | 2 | 32 | 32 | 1.0000 | 1.0000 |
| segment | ✅ | 2 | 54 | 51 | 0.9444 | 0.9444 |

## 逐表结果

| table | 类型 | gold | 预测 | 正确 | Precision | Recall | 错误行（归一化） |
|---|---|---:|---:|---:|---:|---:|---|
| moutai_quarterly | quarterly | 16 | 16 | 16 | 1.0000 | 1.0000 | - |
| yili_quarterly | quarterly | 16 | 16 | 16 | 1.0000 | 1.0000 | - |
| moutai_note_cost | note_cost | 12 | 12 | 12 | 1.0000 | 1.0000 | - |
| yili_note_cost | note_cost | 12 | 12 | 12 | 1.0000 | 1.0000 | - |
| moutai_segment | segment | 21 | 21 | 21 | 1.0000 | 1.0000 | - |
| yili_segment | segment | 33 | 33 | 30 | 0.9091 | 0.9091 | 其他 |
| moutai_annual_data | annual_data | 21 | 21 | 21 | 1.0000 | 1.0000 | - |
| yili_annual_data | annual_data | 18 | 18 | 18 | 1.0000 | 1.0000 | - |
