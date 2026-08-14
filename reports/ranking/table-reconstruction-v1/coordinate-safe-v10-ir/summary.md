# 坐标级表格重建评测（整页块输入，无区域裁剪）

- 数据集：table-eval-v1.json, table-eval-concentration-v1.json
- 输入：chunk 覆盖页的持久化 Document IR v2 line/span geometry；无表格区域裁剪（P0 最严苛口径）
- IR 版本：`7ba72cacdd30eec11da4fd94` / `sha256:5299f4940e2ce4e91084b73dc457d558b9d335fa76fbfee6227e4254eb7f4a30` / processing `8134f2a7b6de38b871e307f6a268b2dba63bebe3e65fdcee7ef46ba35fdc8ef6` / document SHA `f334437dff01af10ddff2ea3f5f8c6897a3cca8591fceeb1fd09c89b0ac63bef`
- IR 版本：`b288fd8c6467f1757516abb7` / `sha256:a82a81e52f52da3cd1b7f38ded08625dc18e3d4522b15d3ef76bf921e54c1f43` / processing `8134f2a7b6de38b871e307f6a268b2dba63bebe3e65fdcee7ef46ba35fdc8ef6` / document SHA `9ae0cd84b172f4bb667ecd8590054cf10dbd17ed72e3977b40bec253a8520a78`

| table | 类型 | 页 | gold | pred | hit | P | R |
|---|---|---|---:|---:|---:|---:|---:|
| moutai_quarterly | quarterly | 6-6 | 16 | 16 | 16 | 1.0000 | 1.0000 |
| yili_quarterly | quarterly | 8-8 | 16 | 16 | 16 | 1.0000 | 1.0000 |
| moutai_note_cost | note_cost | 108-108 | 12 | 12 | 12 | 1.0000 | 1.0000 |
| yili_note_cost | note_cost | 206-206 | 12 | 12 | 12 | 1.0000 | 1.0000 |
| moutai_segment | segment | 9-9 | 21 | 21 | 21 | 1.0000 | 1.0000 |
| yili_segment | segment | 19-20 | 33 | 33 | 30 | 0.9091 | 0.9091 |
| moutai_annual_data | annual_data | 5-5 | 21 | 21 | 21 | 1.0000 | 1.0000 |
| yili_annual_data | annual_data | 7-7 | 18 | 18 | 18 | 1.0000 | 1.0000 |
| moutai_concentration | concentration | 10-11 | 4 | 4 | 4 | 1.0000 | 1.0000 |
| yili_concentration | concentration | 22-22 | 4 | 4 | 4 | 1.0000 | 1.0000 |

合计：gold=157 hit=154 Recall=0.9809
