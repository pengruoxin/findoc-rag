import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RankedHitSnapshot(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    original_rank: int | None = None
    rerank_score: float | None = None
    rank_delta: int | None = None
    retrieval_rank: int | None = None
    scope_score: int | None = None
    scope_rank_delta: int | None = None


class RetrievalStageTrace(BaseModel):
    stage: Literal["lexical", "dense", "rrf", "scope", "structured", "rerank"]
    duration_ms: float = Field(ge=0)
    candidate_count: int = Field(ge=0)
    hits: list[RankedHitSnapshot]


class RetrievalTrace(BaseModel):
    trace_id: str
    request_id: str
    index_id: str
    started_at: datetime
    completed_at: datetime
    query_sha256: str
    query_text: str | None = None
    mode: Literal["lexical", "dense", "hybrid"]
    requested_top_k: int
    candidate_k: int
    base_candidate_k: int | None = None
    candidate_budget_reason: str | None = None
    reranked: bool = False
    reranker_model: str | None = None
    inferred_scope: str | None = None
    scope_confidence: str | None = None
    status: Literal["success", "error"]
    total_duration_ms: float = Field(ge=0)
    result_count: int = Field(ge=0)
    stages: list[RetrievalStageTrace]
    error_type: str | None = None
    error_message: str | None = None


class RetrievalMetrics(BaseModel):
    request_count: int
    success_count: int
    error_count: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    requests_by_mode: dict[str, int]
    errors_by_type: dict[str, int]


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * percentile)))
    return sorted_values[index]


class TraceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    trace_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    index_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_duration_ms REAL NOT NULL,
                    error_type TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS trace_started_at_idx
                    ON retrieval_traces(started_at);
                CREATE INDEX IF NOT EXISTS trace_request_id_idx
                    ON retrieval_traces(request_id);
                CREATE INDEX IF NOT EXISTS trace_status_idx
                    ON retrieval_traces(status);
                """
            )
            connection.commit()

    def record(self, trace: RetrievalTrace) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO retrieval_traces (
                    trace_id, request_id, index_id, started_at, mode, status,
                    total_duration_ms, error_type, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.request_id,
                    trace.index_id,
                    trace.started_at.isoformat(),
                    trace.mode,
                    trace.status,
                    trace.total_duration_ms,
                    trace.error_type,
                    trace.model_dump_json(),
                ),
            )
            connection.commit()

    def get(self, trace_id: str) -> RetrievalTrace | None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT payload_json FROM retrieval_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return RetrievalTrace.model_validate_json(row[0]) if row else None

    def metrics(self) -> RetrievalMetrics:
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT mode, status, total_duration_ms, error_type FROM retrieval_traces"
            ).fetchall()
        latencies = sorted(float(row[2]) for row in rows)
        modes = Counter(str(row[0]) for row in rows)
        errors = Counter(str(row[3]) for row in rows if row[1] == "error" and row[3])
        success_count = sum(row[1] == "success" for row in rows)
        error_count = len(rows) - success_count
        return RetrievalMetrics(
            request_count=len(rows),
            success_count=success_count,
            error_count=error_count,
            error_rate=error_count / len(rows) if rows else 0.0,
            latency_p50_ms=_percentile(latencies, 0.50),
            latency_p95_ms=_percentile(latencies, 0.95),
            latency_max_ms=max(latencies, default=0.0),
            requests_by_mode=dict(modes),
            errors_by_type=dict(errors),
        )
