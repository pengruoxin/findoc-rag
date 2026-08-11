# 坐标级表格重建评测（整页块输入，无区域裁剪）

- 数据集：table-eval-v1.json, table-eval-concentration-v1.json
- 输入：chunk 覆盖页的完整 pymupdf blocks；无表格区域裁剪（P0 最严苛口径）

| table | 类型 | 页 | gold | pred | hit | P | R |
|---|---|---|---:|---:|---:|---:|---:|
| moutai_quarterly | quarterly | 6-6 | 16 | 31 | 12 | 0.3871 | 0.7500 |
| yili_quarterly | quarterly | 8-8 | 16 | 29 | 12 | 0.4138 | 0.7500 |
| moutai_note_cost | note_cost | 108-108 | 12 | 44 | 12 | 0.2727 | 1.0000 |
| yili_note_cost | note_cost | 206-206 | 12 | 38 | 12 | 0.3158 | 1.0000 |
| moutai_segment | segment | 9-9 | 21 | 23 | 0 | 0.0000 | 0.0000 |
| yili_segment | segment | 19-20 | 33 | 37 | 3 | 0.0811 | 0.0909 |
| moutai_annual_data | annual_data | 5-5 | 21 | 34 | 18 | 0.5294 | 0.8571 |
| yili_annual_data | annual_data | 7-7 | 18 | 31 | 15 | 0.4839 | 0.8333 |
| moutai_concentration | concentration | 10-11 | 4 | 4 | 4 | 1.0000 | 1.0000 |
| yili_concentration | concentration | 22-22 | 4 | 4 | 4 | 1.0000 | 1.0000 |

合计：gold=157 hit=92 Recall=0.5860
