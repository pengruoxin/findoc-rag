# REPORT

算法：span→token→行带聚类→数值列聚类→标签-数值配对；识别季度/年度、本期-上期、segment 子表和单位。年度表跳过同比列，segment 保留营业收入/成本/毛利率。坐标不足时回退 `extract_cells`；`merge_pages` 提供跨页去重接口。

当前 JSON 共 157 个 gold，prompt 的 146/149 为旧口径。内置文本基线命中：茅台季度 12/16，伊利季度 16/16；两张 note_cost 各 12/12；茅台 segment 21/21，伊利 segment 30/33；茅台 annual 21/21，伊利 annual 18/18；两张 concentration 各 4/4。总计预测 153、命中 150，P=98.04%，R=95.54%。

坐标可修复茅台季度“扣非净利润”4 值先于标签的问题，也可拼接跨行标签。伊利 segment 文本层只有“其他”而 gold 为“其他地区”，坐标无法恢复缺字，需 OCR 或标记 divergence；代码未硬编码补字。
