#!/usr/bin/env python3
import json
import pathlib
import sys

REQUIRED_PATHS = [
    "schema_version","report_id","created_at",
    "dataset.dataset_id","dataset.dataset_sha256","dataset.split","dataset.query_regime",
    "index.index_id","index.corpus_index_id","index.index_build_id",
    "retrieval.mode","retrieval.top_k","retrieval.candidate_k",
    "retrieval.rrf_weights.lexical","retrieval.rrf_weights.dense",
    "retrieval.metadata_filters.enabled","retrieval.metadata_filters.source",
    "retrieval.metadata_filters.fields","retrieval.metadata_filters.oracle",
    "retrieval.scope","retrieval.adaptive",
    "retrieval.reranker.enabled","retrieval.reranker.model","retrieval.reranker.top_n",
    "models.dense_embedding_model","models.generation_model","models.judge_model",
]
FILTER_SOURCES = {"none","query_parser","user_explicit","gold_metadata","oracle","other"}
QUERY_REGIMES = {"canonical","ticker_or_finance_shorthand","semantic_or_relative_time","all"}
MODES = {"lexical","dense","hybrid","hybrid_rerank","other"}

def get_path(obj, path):
    cur=obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur=cur[part]
    return cur

def validate(report):
    errors=[]
    for p in REQUIRED_PATHS:
        try:
            get_path(report,p)
        except KeyError:
            errors.append(f"missing required field: {p}")

    if errors:
        return errors

    if report["schema_version"] != 1:
        errors.append("schema_version must be 1")

    if report["dataset"]["query_regime"] not in QUERY_REGIMES:
        errors.append(f"dataset.query_regime must be one of {sorted(QUERY_REGIMES)}")

    mode=report["retrieval"]["mode"]
    if mode not in MODES:
        errors.append(f"retrieval.mode must be one of {sorted(MODES)}")

    mf=report["retrieval"]["metadata_filters"]
    if mf["source"] not in FILTER_SOURCES:
        errors.append(f"retrieval.metadata_filters.source must be one of {sorted(FILTER_SOURCES)}")
    if mf["enabled"] and mf["source"] == "none":
        errors.append("metadata filters enabled but source='none'")
    if not mf["enabled"] and mf["source"] not in {"none","other"}:
        errors.append("metadata filters disabled but a non-none source was recorded")
    if mf.get("oracle") and mf["source"] not in {"gold_metadata","oracle"}:
        errors.append("metadata_filters.oracle=true requires source gold_metadata/oracle")
    if mf["source"] in {"gold_metadata","oracle"} and not mf.get("oracle"):
        errors.append("gold/oracle metadata source must set metadata_filters.oracle=true")

    r=report["retrieval"]
    if not isinstance(r["top_k"], int) or r["top_k"] < 1:
        errors.append("retrieval.top_k must be a positive integer")
    if not isinstance(r["candidate_k"], int) or r["candidate_k"] < r["top_k"]:
        errors.append("retrieval.candidate_k must be an integer >= top_k")

    rr=r["reranker"]
    if rr["enabled"]:
        if not rr["model"]:
            errors.append("reranker enabled but retrieval.reranker.model is empty")
        if not isinstance(rr["top_n"], int) or rr["top_n"] < r["top_k"]:
            errors.append("reranker enabled: top_n must be an integer >= top_k")

    dense_used=mode in {"dense","hybrid","hybrid_rerank"}
    if dense_used and not report["models"]["dense_embedding_model"]:
        errors.append("dense retrieval is used but models.dense_embedding_model is empty")

    if mode in {"hybrid","hybrid_rerank"}:
        w=r["rrf_weights"]
        if w["lexical"] < 0 or w["dense"] < 0 or (w["lexical"]+w["dense"] <= 0):
            errors.append("hybrid RRF weights must be non-negative and not both zero")

    # Relative-time runs must record their frozen anchor.
    if report["dataset"]["query_regime"] == "semantic_or_relative_time":
        time_cfg = report.get("time") or {}
        if not time_cfg.get("as_of_date"):
            errors.append(
                "dataset.query_regime=semantic_or_relative_time requires time.as_of_date"
            )

    # Explicit leakage warning promoted to validation error for primary reports.
    if mf["source"] in {"gold_metadata","oracle"} or mf.get("oracle"):
        errors.append("oracle/gold metadata filtering is diagnostic-only; do not treat this report as a primary benchmark score")

    return errors

def main():
    if len(sys.argv) != 2:
        print("usage: python validate_eval_report.py <report.json>", file=sys.stderr)
        return 2
    path=pathlib.Path(sys.argv[1])
    report=json.loads(path.read_text(encoding="utf-8"))
    errors=validate(report)
    if errors:
        print("INVALID")
        for e in errors:
            print(f"- {e}")
        return 1
    print("VALID")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
