# Holdout evaluation v2

Runtime: local index `10fb50419145d56720c9`, 16 assistant-reviewed queries, top_k=5, dense model loaded offline.

| Pipeline | Hit@5 | MRR@5 |
|---|---:|---:|
| BM25 / lexical | 0.6875 | 0.4531 |
| Dense multilingual-e5-small | 0.2500 | 0.0802 |
| Hybrid + adaptive budget | 0.5000 | 0.3615 |

## Interpretation

On this provisional holdout, lexical retrieval is strongest. Hybrid fusion underperforms BM25, so the next optimization target is fusion/routing calibration rather than increasing dense model size blindly. The result is not an independent benchmark: questions and evidence were assistant-reviewed, and the corpus/index are shared with development diagnostics.

Raw outputs:

- `holdout-eval-v2-bm25.json`
- `holdout-eval-v2-dense.json`
- `holdout-eval-v2-hybrid.json`
