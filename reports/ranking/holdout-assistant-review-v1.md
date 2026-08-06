# Assistant-reviewed holdout v1

This report records semantic review of the 16-item holdout pack. It is a quality-control artifact, not an independently created gold set.

| Status | Count | Benchmark use |
|---|---:|---|
| approved | 12 | eligible |
| edited | 4 | eligible with reviewed chunk IDs |
| rejected | 0 | not applicable |

The review corrected two recurring errors: confusing扣非净利润 with扣非每股收益/ROE, and selecting an adjacent statement-analysis table instead of the actual segment industry/product table. Net assets were bound to the annual accounting-data row for归属于上市公司股东的净资产.

Because the evidence is from the same corpus and pipeline, results must be labelled assistant-reviewed provisional holdout evidence, not independent human-annotated gold.

Source decisions: `data/diagnostics/holdout-assistant-review-v1.json`.
