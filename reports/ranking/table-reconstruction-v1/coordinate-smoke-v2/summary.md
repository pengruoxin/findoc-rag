# 坐标级表格重建评测（整页块输入，无区域裁剪）

- 数据集：table-eval-v1.json, table-eval-concentration-v1.json
- 输入：chunk 覆盖页的完整 pymupdf blocks；无表格区域裁剪（P0 最严苛口径）

| table | 类型 | 页 | gold | pred | hit | P | R |
|---|---|---|---:|---:|---:|---:|---:|
| moutai_quarterly | quarterly | 6-6 | 16 | 21 | 8 | 0.3810 | 0.5000 |
| yili_quarterly | quarterly | 8-8 | 16 | 21 | 16 | 0.7619 | 1.0000 |
| moutai_note_cost | note_cost | 108-108 | 12 | 13 | 12 | 0.9231 | 1.0000 |
| yili_note_cost | note_cost | 206-206 | 12 | 38 | 12 | 0.3158 | 1.0000 |
| moutai_segment | segment | 9-9 | 21 | 22 | 0 | 0.0000 | 0.0000 |
| yili_segment | segment | 19-20 | 33 | 10 | 0 | 0.0000 | 0.0000 |
| moutai_annual_data | annual_data | 5-5 | 21 | 6 | 6 | 1.0000 | 0.2857 |
| yili_annual_data | annual_data | 7-7 | 18 | 18 | 18 | 1.0000 | 1.0000 |
| moutai_concentration | concentration | 10-11 | 4 | 4 | 4 | 1.0000 | 1.0000 |
| yili_concentration | concentration | 22-22 | 4 | 4 | 4 | 1.0000 | 1.0000 |

合计：gold=157 hit=80 Recall=0.5096
