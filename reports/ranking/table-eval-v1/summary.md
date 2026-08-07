# 表格抽取评测 v1（table-eval-v1）

- 数据集：`table-eval-v1` | 表数：8 | 标注单元格：149
- 匹配规则：单元格三元组 (行标签归一化, 列头, 数值归一化) 完全一致才算对。

| 表型 | 实现 | 表数 | gold cells | 正确 cells | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| annual_data | ⬜ 未实现 | 2 | 39 | 0 | 0.0000 | 0.0000 |
| note_cost | ⬜ 未实现 | 2 | 24 | 0 | 0.0000 | 0.0000 |
| quarterly | ✅ | 2 | 32 | 28 | 0.8750 | 0.8750 |
| segment | ⬜ 未实现 | 2 | 54 | 0 | 0.0000 | 0.0000 |

## 逐表结果

| table | 类型 | gold | 预测 | 正确 | Precision | Recall | 错误行（归一化） |
|---|---|---:|---:|---:|---:|---:|---|
| moutai_quarterly | quarterly | 16 | 16 | 12 | 0.7500 | 0.7500 | 归属于上市公司股东的扣除非经常性损益后的净利润 |
| yili_quarterly | quarterly | 16 | 16 | 16 | 1.0000 | 1.0000 | - |
| moutai_note_cost | note_cost | 12 | 0 | 0 | 0.0000 | 0.0000 | - |
| yili_note_cost | note_cost | 12 | 0 | 0 | 0.0000 | 0.0000 | - |
| moutai_segment | segment | 21 | 0 | 0 | 0.0000 | 0.0000 | - |
| yili_segment | segment | 33 | 0 | 0 | 0.0000 | 0.0000 | - |
| moutai_annual_data | annual_data | 21 | 0 | 0 | 0.0000 | 0.0000 | - |
| yili_annual_data | annual_data | 18 | 0 | 0 | 0.0000 | 0.0000 | - |
